// ============================================================================
// REDUCTION PATTERNS — kernel3.cu
// Topic: Warp Shuffle Reduction  vs  Plain Tree Reduction (baseline)
//
// THREE kernels benchmarked side-by-side:
//   Kernel 1: treeReduction        — shared mem tree (8 rounds), atomicAdd
//   Kernel 2: warpShuffleReduction — shared mem tree (3 rounds) + shuffle (5 rounds)
//   Kernel 3: pureWarpReduction    — shuffle per warp, ONLY 8 values hit shared mem
//
// Compile:   .\compile.bat 05-reduction-patterns\kernel3.cu
// Profile:   .\compile.bat 05-reduction-patterns\kernel3.cu profile
//            Open kernel3-analysis.ncu-rep → Summary tab → 3 rows.
// Divergence:.\compile.bat 05-reduction-patterns\kernel3.cu divergence
//
// ─────────────────────────────────────────────────────────────────────────────
// WHAT IS __shfl_down_sync?
//
//   A "warp shuffle" instruction exchanges data BETWEEN THREADS IN THE SAME
//   WARP without touching shared memory or going through the cache hierarchy.
//   Data moves through a dedicated warp-level interconnect — ~1-2 cycles.
//   Compare: shared memory ~5 cycles, L1 ~30 cycles, DRAM ~600 cycles.
//
//   Syntax:
//     float result = __shfl_down_sync(mask, var, offset);
//
//   Parameters:
//     mask   — which threads participate. 0xffffffff = all 32 threads.
//     var    — the value THIS thread contributes to the exchange.
//     offset — thread (lane + offset) sends its var to thread lane.
//
//   What each thread receives:
//     thread lane  →  receives var from thread (lane + offset)
//     if lane + offset >= 32  →  thread keeps its OWN var (no wraparound)
//
//   Example: __shfl_down_sync(0xffffffff, val, 16)
//     thread 0  receives val from thread 16   → adds them: val = val[0]  + val[16]
//     thread 1  receives val from thread 17   → adds them: val = val[1]  + val[17]
//     ...
//     thread 15 receives val from thread 31   → adds them: val = val[15] + val[31]
//     thread 16 source = 16+16 = 32 >= 32    → keeps its own val[16]
//     thread 17 source = 33 >= 32            → keeps its own val[17]
//     ...
//     thread 31 source = 47 >= 32            → keeps its own val[31]
//
// ─────────────────────────────────────────────────────────────────────────────
// WHY ONLY THREAD 0 HAS THE CORRECT SUM AFTER THE LOOP:
//
//   Start: val[0..31] = sdata[0..31]   (32 partial sums from Stage 2)
//
//   After offset=16:
//     val[0]  = sdata[0]  + sdata[16]   ← thread 16 still has sdata[16]
//     val[1]  = sdata[1]  + sdata[17]
//     ...
//     val[15] = sdata[15] + sdata[31]
//     val[16] = sdata[16]  (unchanged — source 32 was out of range)
//     val[17..31] same
//
//   After offset=8:
//     val[0]  = (sdata[0]+sdata[16]) + (sdata[8]+sdata[24])  ✓
//     val[8]  = (sdata[8]+sdata[24]) + sdata[16]             ✗ (stale sdata[16])
//     Thread 8 is wrong — but we don't care about thread 8's value.
//
//   After offset=4, offset=2, offset=1:
//     val[0]  = sdata[0]+sdata[1]+...+sdata[31]  ✓  ← ONLY thread 0 is correct
//     val[1..31]: partial or garbage              ✗  ← we never use them
//
//   NO guard needed inside the loop — all 32 threads execute every instruction.
//   Threads whose results are "wrong" just never write to output.
//   → ZERO warp divergence in Stage 3.
//
// ─────────────────────────────────────────────────────────────────────────────
// WHAT WARP SHUFFLE REPLACES:
//
//   Plain tree reduction for the last 5 rounds (stride 16, 8, 4, 2, 1):
//     for stride=16..1:
//       if (tid < stride) sdata[tid] += sdata[tid+stride];
//       __syncthreads();   ← unnecessary (warp 0 is in lockstep anyway)
//     → 5 × (if-check + shared mem read + shared mem write + sync) = wasted
//     → "if (tid < stride)" causes divergence inside warp 0 for stride < 32
//
//   Warp shuffle for those same 5 rounds:
//     sum += __shfl_down_sync(0xffffffff, sum, offset)   for offset 16,8,4,2,1
//     → No shared memory   (~5 cycles saved per access)
//     → No __syncthreads() (warp executes in lockstep)
//     → No divergence      (all 32 threads run every line)
//
// ─────────────────────────────────────────────────────────────────────────────
// WHAT NSIGHT WILL SHOW:
//
//   [Section 3 — Shared Memory]
//     Both kernels: Bank Conflicts = 0  (sequential addressing)
//     Shuffle kernel: fewer Shared Mem ops (no shared memory in Stage 3)
//
//   [Section 4 — Warp State]
//     treeReduction:
//       Active Threads Per Warp < 32 for the last 5 reduction rounds
//       → divergence from "if (tid < stride)" when stride < 32
//     warpShuffleReduction:
//       Active Threads Per Warp should be closer to 32 in Stage 3
//       → no branch inside the shuffle loop
//
//   [Compare rows in Summary tab]
//     Duration should be similar or shuffle slightly faster.
//     The improvement is more visible at very large N where Stage 3
//     is a larger fraction of total time.
//
// ============================================================================

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE       256    // threads per block — must be a power of 2
#define WARP_SIZE         32    // threads per warp on all NVIDIA GPUs since Fermi
#define FIXED_BLOCKS      96    // 24 SMs × 4 blocks — saturates all SMs, no tail waste
#define WARPS_PER_BLOCK    8    // BLOCK_SIZE / WARP_SIZE = 256 / 32 = 8

// Participation mask for the second shuffle (only 8 threads, lanes 0-7)
// Bits 0-7 set = 0b00000000_00000000_00000000_11111111 = 0x000000ff
#define WARP_SUMS_MASK  0xffu

// ============================================================================
// KERNEL 1: treeReduction  (baseline, plain shared memory)
//
// This is the same sequential tree reduction from kernel2.cu for comparison.
// All 8 rounds (stride 128..1) go through shared memory.
// The last 5 rounds (stride 16..1) suffer from warp divergence.
//
// Memory path for all 8 reduction rounds: register → shared → shared → ... → atomicAdd
// ============================================================================
__global__ void treeReduction(float *input, float *output, int n) {
    __shared__ float sharedData[BLOCK_SIZE];

    int tid        = threadIdx.x;
    int idx        = blockIdx.x * blockDim.x + tid;
    int gridStride = blockDim.x * gridDim.x;

    // Grid-stride accumulation: each thread processes multiple elements
    float sum = 0.0f;
    for (int i = idx; i < n; i += gridStride)
        sum += input[i];

    sharedData[tid] = sum;
    __syncthreads();

    // All 8 rounds of tree reduction through shared memory
    // Strides 16, 8, 4, 2, 1 cause divergence inside warp 0:
    //   stride=16: threads 0-15 active, 16-31 idle → half-warp divergent
    //   stride=8:  threads 0-7 active                → 1/4 warp active
    //   etc.
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride)
            sharedData[tid] += sharedData[tid + stride];
        __syncthreads();
    }

    if (tid == 0)
        atomicAdd(output, sharedData[0]);
}

// ============================================================================
// KERNEL 2: warpShuffleReduction  (warp shuffle for final 5 rounds)
//
// Same as treeReduction for Stages 1 and most of Stage 2.
// The difference: once we've reduced to 32 elements in sharedData[0..31],
// we hand off to the warp shuffle for the last 5 rounds — no shared memory,
// no __syncthreads, no divergence.
//
// Memory path:
//   Stage 1: global → register (grid-stride, coalesced)
//   Stage 2: register → shared → shared (3 rounds: stride 128, 64, 32)
//   Stage 3: shared → register → register (5 shuffle rounds, no shared writes)
//   Final:   register → global (atomicAdd, thread 0 only)
//
// Three stages, then one atomic write.
// ============================================================================
__global__ void warpShuffleReduction(float *input, float *output, int n) {
    __shared__ float sharedData[BLOCK_SIZE];

    int tid        = threadIdx.x;
    int idx        = blockIdx.x * blockDim.x + tid;
    int gridStride = blockDim.x * gridDim.x;

    // ── Stage 1: Grid-stride accumulation (register only) ─────────────────
    // Identical to treeReduction Stage 1.
    float sum = 0.0f;
    for (int i = idx; i < n; i += gridStride)
        sum += input[i];

    sharedData[tid] = sum;
    __syncthreads();

    // ── Stage 2: Shared memory tree reduction — 256 elements → 32 ─────────
    //
    // Stop at stride = WARP_SIZE (32) inclusive.
    // Loop runs for: stride = 128, 64, 32
    //
    //   stride=128: threads 0-127 active (4 warps), add sdata[0..127] += sdata[128..255]
    //   stride=64:  threads 0-63  active (2 warps), add sdata[0..63]  += sdata[64..127]
    //   stride=32:  threads 0-31  active (1 warp),  add sdata[0..31]  += sdata[32..63]
    //
    // After stride=32:  sharedData[0..31] contains 32 partial sums.
    //                   Each value is the sum of 8 original elements.
    //
    // __syncthreads() is still required here because strides 128 and 64
    // involve multiple warps — they do NOT execute in lockstep.
    for (int stride = blockDim.x / 2; stride >= WARP_SIZE; stride >>= 1) {
        if (tid < stride)
            sharedData[tid] += sharedData[tid + stride];
        __syncthreads();
        // After this final __syncthreads() (stride=32 round), all 32 writes
        // to sharedData[0..31] are guaranteed visible to warp 0 in Stage 3.
    }

    // ── Stage 3: Warp-level reduction using __shfl_down_sync ──────────────
    //
    // Only the first warp (tid < 32) participates.
    // These 32 threads are in the same warp → they execute in lockstep (SIMT).
    // No __syncthreads() is needed between shuffle steps.
    //
    // Each thread loads ONE of the 32 partial sums into a register.
    // Then 5 rounds of shuffle-down reduce 32 values to 1 in thread 0's register.
    if (tid < WARP_SIZE) {
        // Load from shared memory into a register — LAST time we touch shared mem.
        float val = sharedData[tid];
        //   val[0]  = partial sum of elements block0 processed (elements 0, gridStride, 2*gridStride, ...)
        //   val[1]  = partial sum for block1's contribution
        //   ...
        //   val[31] = partial sum for block31's contribution

        // ── Shuffle-down reduction: 5 rounds ──────────────────────────────
        //
        // Round 1: offset=16
        //   thread 0  receives val from thread 16 → val[0]  = val[0]  + val[16]
        //   thread 1  receives val from thread 17 → val[1]  = val[1]  + val[17]
        //   ...
        //   thread 15 receives val from thread 31 → val[15] = val[15] + val[31]
        //   thread 16..31: source out of range    → keep their own val (unchanged)
        //
        // Round 2: offset=8
        //   thread 0  += thread 8's val  → val[0] = (val[0]+val[16]) + (val[8]+val[24])
        //   thread 8  += thread 16's val → val[8] = (val[8]+val[24]) + val[16]  ← WRONG
        //                                            (thread 16 kept stale value)
        //   → thread 8's value is incorrect from round 2 onward. That's fine.
        //   → thread 0 continues accumulating correctly.
        //
        // After all 5 rounds:
        //   val in thread 0 = sharedData[0] + sharedData[1] + ... + sharedData[31] ✓
        //   val in thread 1..31: partially correct or garbage — never used.
        //
        // Key advantage over tree reduction:
        //   No if-guard inside the loop → NO divergence (all 32 threads always active)
        //   No shared memory write/read  → ~5 cycle saving per round × 5 rounds
        //   No __syncthreads()           → 5 barrier cycles eliminated
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
            val += __shfl_down_sync(0xffffffff, val, offset);

        // Thread 0 now has the block's total sum in register val.
        // Write directly from register — no need to store back to shared memory.
        if (tid == 0)
            atomicAdd(output, val);
    }
}

// ============================================================================
// KERNEL 3: pureWarpReduction  (pure warp shuffle, minimal shared memory)
//
// DESIGN IDEA:
//   Instead of writing ALL 256 partial sums to shared memory first, let each
//   warp reduce its 32 threads' values entirely through shuffle instructions.
//   Only the winner of each warp (lane 0) ever touches shared memory — and
//   only to hand off 8 values to the final stage.
//
// MEMORY TRAFFIC COMPARISON (256 threads per block):
//
//   treeReduction:
//     Shared writes:  256 (initial store) + 128+64+32+16+8+4+2+1 = 256+255 = 511
//     Shared reads:   same pattern = 511
//     Total:          ~1022 shared mem ops per block
//
//   warpShuffleReduction:
//     Shared writes:  256 (initial) + 128+64+32 = 256+224 = 480
//     Shared reads:   480 (tree reads) + 32 (stage 3 load) = 512
//     Total:          ~992 shared mem ops per block
//
//   pureWarpReduction:
//     Shared writes:  8  (one per warp, lane 0 only)
//     Shared reads:   8  (one per thread in final stage, lanes 0-7 of warp 0)
//     Total:          16 shared mem ops per block  ← 64× fewer!
//
// HOW THE TWO SHUFFLES WORK:
//
//   First shuffle (per warp, 5 rounds, mask=0xffffffff):
//     All 8 warps do this independently and simultaneously.
//     Each warp reduces 32 thread values to 1 in lane 0's register.
//
//     Start (warp k): warpSum[lane] = partial sum from grid-stride loop
//     After offset=16: lane 0 has warpSum[0]+warpSum[16]
//     After offset=8:  lane 0 has warpSum[0]+warpSum[8]+warpSum[16]+warpSum[24]
//     After offset=4,2,1: lane 0 has sum of all 32 lanes of warp k
//
//   Shared memory bridge (8 writes, 1 sync):
//     warpSums[warpId] = warpResult   (only from lane 0 of each warp)
//     __syncthreads()                  (wait until all 8 values are written)
//
//   Second shuffle (threads 0-7, 3 rounds, mask=0xff):
//     Only threads 0-7 participate (they ARE lanes 0-7 of warp 0).
//     8 values → 3 rounds (offset 4, 2, 1) → thread 0 has block total.
//     Mask = 0xff means only bits 0-7 in the mask are set → lanes 0-7 only.
//
//     Start: val[lane] = warpSums[lane]  for lane 0..7
//     After offset=4: lane 0 has warpSums[0]+warpSums[4]
//     After offset=2: lane 0 has warpSums[0]+warpSums[2]+warpSums[4]+warpSums[6]
//     After offset=1: lane 0 has sum of all 8 warpSums → block total
//
//   Final write:
//     Thread 0 (lane 0) calls atomicAdd directly from register — no sync needed.
//
// ABOUT THE TEMPLATE'S TWO BUGS (fixed here):
//
//   Bug 1: __shared__ declared inside  if (laneId == 0) { ... }
//     In CUDA, __shared__ is a compile-time allocation — the shared memory slot
//     exists for ALL threads in the block regardless of which thread declares it.
//     Declaring inside an if-block is misleading (it appears conditional but isn't)
//     and technically undefined behavior. Moved to function scope.
//
//   Bug 2: output[blockIdx.x] = warpSums[0] after the final shuffle
//     After the second shuffle, thread 0 already has the correct block sum in
//     a REGISTER (val). Writing val back to warpSums[0] and then reading it
//     would be correct only for thread 0 (other threads must not overwrite it).
//     Simpler fix: thread 0 uses val directly for atomicAdd — zero overhead.
//
// NSIGHT — what to look for:
//   [Section 3 — Shared Memory]
//     Shared Mem Transactions: dramatically lower than kernels 1 and 2
//     16 ops/block × 96 blocks = 1,536 total  vs  ~100,000 for treeReduction
//   [Section 4 — Warp State]
//     Active Threads Per Warp: should be 32 in ALL shuffle stages
//     (no if-guards inside either shuffle loop)
//     Barrier stalls: only 1 __syncthreads() in the entire kernel
//   [Section 2 — Memory Workload]
//     DRAM Throughput: same as others (reading N floats once is unavoidable)
//     → pureWarpReduction is fastest when the reduction overhead is significant,
//       i.e. when N is small or the kernel is called in a tight loop.
// ============================================================================
__global__ void pureWarpReduction(float *input, float *output, int n) {

    // WARPS_PER_BLOCK = 8 slots — declared at function scope (not inside an if).
    // Only 8 floats = 32 bytes of shared memory per block. Negligible.
    __shared__ float warpSums[WARPS_PER_BLOCK];

    int tid    = threadIdx.x;
    int idx    = blockIdx.x * blockDim.x + tid;
    int laneId = tid % WARP_SIZE;   // lane 0..31 within this thread's warp
    int warpId = tid / WARP_SIZE;   // warp 0..7 within this block

    // ── Stage 1: Grid-stride accumulation (all 256 threads, register only) ─
    // Exactly like the other kernels. Each thread accumulates its share of the
    // input into a local register — NO shared memory involved here.
    float warpSum = 0.0f;
    for (int i = idx; i < n; i += blockDim.x * gridDim.x)
        warpSum += input[i];

    // ── Stage 2: Intra-warp reduction via shuffle (all 8 warps, parallel) ──
    //
    // All 8 warps run this simultaneously. Each warp independently reduces its
    // 32 thread values to 1 value in lane 0's register.
    //
    // Mask = 0xffffffff → all 32 lanes of THIS warp participate.
    // No divergence: all 32 threads in a warp execute every shuffle instruction.
    // After the loop: lane 0 of each warp has that warp's total sum.
    //                 lanes 1-31 have partial/garbage values (never used).
    float warpResult = warpSum;
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        warpResult += __shfl_down_sync(0xffffffff, warpResult, offset);

    // ── Stage 3: Lane 0 of each warp stores result to shared memory ─────────
    //
    // Only 8 writes happen (one per warp). All other lanes skip the store.
    // No race condition: each warp's lane 0 writes to a DIFFERENT warpSums slot.
    //   warp 0 → warpSums[0]
    //   warp 1 → warpSums[1]
    //   ...
    //   warp 7 → warpSums[7]
    if (laneId == 0)
        warpSums[warpId] = warpResult;

    // Barrier: wait until all 8 warp totals are visible in shared memory.
    // This is the ONLY __syncthreads() in the entire kernel.
    __syncthreads();

    // ── Stage 4: Second shuffle to combine the 8 warp sums ─────────────────
    //
    // Only threads 0-7 participate (the first warp's first 8 lanes).
    // They are all in WARP 0 → execute in lockstep → no __syncthreads needed.
    //
    // Mask = 0xff → only bits 0-7 set → lanes 0-7 participate.
    // This tells the hardware: "threads 8-31 of this warp are NOT involved."
    //
    // 8 values need ceil(log2(8))=3 rounds: offset = 4, 2, 1.
    // After the loop: lane 0 (= thread 0) has the sum of all 8 warp totals
    //                 = the entire block's contribution to the reduction.
    if (tid < WARPS_PER_BLOCK) {
        float val = warpSums[tid];   // load from shared → register (last touch of shmem)

        // 3 rounds with the restricted mask
        for (int offset = WARPS_PER_BLOCK / 2; offset > 0; offset >>= 1)
            val += __shfl_down_sync(WARP_SUMS_MASK, val, offset);

        // Thread 0 has the block total in register val — write directly.
        // No need to store to shared memory first: val is already the answer.
        if (tid == 0)
            atomicAdd(output, val);
    }
}

// ============================================================================
// CPU reference
// ============================================================================
float cpuSum(const float *arr, int n) {
    double acc = 0.0;
    for (int i = 0; i < n; i++) acc += (double)arr[i];
    return (float)acc;
}

// ============================================================================
// Error-check macro
// ============================================================================
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d — %s\n",                    \
                    __FILE__, __LINE__, cudaGetErrorString(err));               \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

// ============================================================================
// Timing helper (warmup + timed run, returns ms)
// ============================================================================
static float timeKernel(void (*fn)(float*, float*, int),
                        float *d_in, float *d_out, int n) {
    CUDA_CHECK(cudaMemset(d_out, 0, sizeof(float)));
    fn(d_in, d_out, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));
    CUDA_CHECK(cudaMemset(d_out, 0, sizeof(float)));
    CUDA_CHECK(cudaEventRecord(t0));
    fn(d_in, d_out, n);
    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
    CUDA_CHECK(cudaEventDestroy(t0));
    CUDA_CHECK(cudaEventDestroy(t1));
    return ms;
}

static void launchTree(float *d_in, float *d_out, int n) {
    treeReduction<<<FIXED_BLOCKS, BLOCK_SIZE>>>(d_in, d_out, n);
}
static void launchShuffle(float *d_in, float *d_out, int n) {
    warpShuffleReduction<<<FIXED_BLOCKS, BLOCK_SIZE>>>(d_in, d_out, n);
}
static void launchPureWarp(float *d_in, float *d_out, int n) {
    pureWarpReduction<<<FIXED_BLOCKS, BLOCK_SIZE>>>(d_in, d_out, n);
}

// ============================================================================
// main
// ============================================================================
int main() {
    printf("=== WARP SHUFFLE REDUCTION (kernel3.cu) ===\n\n");
    printf("Grid: %d fixed blocks × %d threads/block = %d total threads\n",
           FIXED_BLOCKS, BLOCK_SIZE, FIXED_BLOCKS * BLOCK_SIZE);
    printf("WARP_SIZE: %d\n\n", WARP_SIZE);

    printf("Stage breakdown:\n");
    printf("  treeReduction:\n");
    printf("    Stage 1: grid-stride → register\n");
    printf("    Stage 2: tree in shared memory   (8 rounds: stride 128..1)\n");
    printf("    Shared memory ops: ~1022/block\n");
    printf("    Final:   atomicAdd\n\n");
    printf("  warpShuffleReduction:\n");
    printf("    Stage 1: grid-stride → register  (same)\n");
    printf("    Stage 2: tree in shared memory   (3 rounds: stride 128, 64, 32)\n");
    printf("    Stage 3: warp shuffle             (5 rounds: offset 16, 8, 4, 2, 1)\n");
    printf("    Shared memory ops: ~992/block\n");
    printf("    Final:   atomicAdd from register\n\n");
    printf("  pureWarpReduction:\n");
    printf("    Stage 1: grid-stride → register  (same)\n");
    printf("    Stage 2: intra-warp shuffle       (5 rounds, all 8 warps parallel)\n");
    printf("    Stage 3: lane 0 → warpSums[]      (8 writes to shared memory)\n");
    printf("    Stage 4: second shuffle           (3 rounds on 8 values, mask=0xff)\n");
    printf("    Shared memory ops: 16/block  ← 64x fewer than treeReduction\n");
    printf("    Final:   atomicAdd from register\n\n");

    // ── Correctness: N=1024, all 1s → expected = 1024 ─────────────────────
    {
        const int    n  = 1024;
        const size_t B  = n * sizeof(float);
        float *h_in     = (float *)malloc(B);
        for (int i = 0; i < n; i++) h_in[i] = 1.0f;
        float ref = cpuSum(h_in, n);

        float *d_in, *d_out;
        CUDA_CHECK(cudaMalloc(&d_in,  B));
        CUDA_CHECK(cudaMalloc(&d_out, sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_in, h_in, B, cudaMemcpyHostToDevice));

        float r1 = 0.0f, r2 = 0.0f, r3 = 0.0f;

        CUDA_CHECK(cudaMemset(d_out, 0, sizeof(float)));
        treeReduction<<<FIXED_BLOCKS, BLOCK_SIZE>>>(d_in, d_out, n);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(&r1, d_out, sizeof(float), cudaMemcpyDeviceToHost));

        CUDA_CHECK(cudaMemset(d_out, 0, sizeof(float)));
        warpShuffleReduction<<<FIXED_BLOCKS, BLOCK_SIZE>>>(d_in, d_out, n);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(&r2, d_out, sizeof(float), cudaMemcpyDeviceToHost));

        CUDA_CHECK(cudaMemset(d_out, 0, sizeof(float)));
        pureWarpReduction<<<FIXED_BLOCKS, BLOCK_SIZE>>>(d_in, d_out, n);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(&r3, d_out, sizeof(float), cudaMemcpyDeviceToHost));

        printf("[Correctness N=%-6d]  ref=%.0f\n", n, ref);
        printf("  treeReduction         : %.0f  %s\n", r1,
               fabsf(r1 - ref) < 0.5f ? "PASS" : "FAIL");
        printf("  warpShuffleReduction  : %.0f  %s\n", r2,
               fabsf(r2 - ref) < 0.5f ? "PASS" : "FAIL");
        printf("  pureWarpReduction     : %.0f  %s\n", r3,
               fabsf(r3 - ref) < 0.5f ? "PASS" : "FAIL");

        free(h_in); cudaFree(d_in); cudaFree(d_out);
    }

    // ── Correctness: N=999983, non-power-of-2 ─────────────────────────────
    {
        const int    n  = 999983;
        const size_t B  = n * sizeof(float);
        float *h_in     = (float *)malloc(B);
        for (int i = 0; i < n; i++) h_in[i] = 1.0f;
        float ref = cpuSum(h_in, n);

        float *d_in, *d_out;
        CUDA_CHECK(cudaMalloc(&d_in,  B));
        CUDA_CHECK(cudaMalloc(&d_out, sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_in, h_in, B, cudaMemcpyHostToDevice));

        float r1 = 0.0f, r2 = 0.0f, r3 = 0.0f;

        CUDA_CHECK(cudaMemset(d_out, 0, sizeof(float)));
        treeReduction<<<FIXED_BLOCKS, BLOCK_SIZE>>>(d_in, d_out, n);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(&r1, d_out, sizeof(float), cudaMemcpyDeviceToHost));

        CUDA_CHECK(cudaMemset(d_out, 0, sizeof(float)));
        warpShuffleReduction<<<FIXED_BLOCKS, BLOCK_SIZE>>>(d_in, d_out, n);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(&r2, d_out, sizeof(float), cudaMemcpyDeviceToHost));

        CUDA_CHECK(cudaMemset(d_out, 0, sizeof(float)));
        pureWarpReduction<<<FIXED_BLOCKS, BLOCK_SIZE>>>(d_in, d_out, n);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(&r3, d_out, sizeof(float), cudaMemcpyDeviceToHost));

        printf("\n[Correctness N=%-6d]  ref=%.0f  (non-power-of-2)\n", n, ref);
        printf("  treeReduction         : %.0f  %s\n", r1,
               fabsf(r1 - ref) < 0.5f ? "PASS" : "FAIL");
        printf("  warpShuffleReduction  : %.0f  %s\n", r2,
               fabsf(r2 - ref) < 0.5f ? "PASS" : "FAIL");
        printf("  pureWarpReduction     : %.0f  %s\n\n", r3,
               fabsf(r3 - ref) < 0.5f ? "PASS" : "FAIL");

        free(h_in); cudaFree(d_in); cudaFree(d_out);
    }

    // ── Benchmark: N=16M (~64 MB) ──────────────────────────────────────────
    {
        const int    n  = 1 << 24;
        const size_t B  = (size_t)n * sizeof(float);
        float *h_in     = (float *)malloc(B);
        for (int i = 0; i < n; i++) h_in[i] = 1.0f;

        float *d_in, *d_out;
        CUDA_CHECK(cudaMalloc(&d_in,  B));
        CUDA_CHECK(cudaMalloc(&d_out, sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_in, h_in, B, cudaMemcpyHostToDevice));

        float ms1 = timeKernel(launchTree,     d_in, d_out, n);
        float ms2 = timeKernel(launchShuffle,  d_in, d_out, n);
        float ms3 = timeKernel(launchPureWarp, d_in, d_out, n);

        float bw1 = (float)B / (ms1 * 1e6f);
        float bw2 = (float)B / (ms2 * 1e6f);
        float bw3 = (float)B / (ms3 * 1e6f);

        printf("[Benchmark N=%d (~64 MB)]\n", n);
        printf("  treeReduction         : %6.3f ms  BW=%5.1f GB/s\n",          ms1, bw1);
        printf("  warpShuffleReduction  : %6.3f ms  BW=%5.1f GB/s  (%.2fx)\n", ms2, bw2, ms1/ms2);
        printf("  pureWarpReduction     : %6.3f ms  BW=%5.1f GB/s  (%.2fx)\n", ms3, bw3, ms1/ms3);
        printf("  (theoretical peak ~192 GB/s on GTX 1660Ti)\n\n");
        printf("  Why all times are similar at N=16M:\n");
        printf("    ALL three are dominated by Phase 1 — reading 64 MB from DRAM.\n");
        printf("    The entire reduction tree (Stages 2-4) accounts for < 1%% of time.\n");
        printf("    pureWarpReduction's advantage appears when Phase 1 is cheaper:\n");
        printf("      - Small N (< 1M): reduction is a larger fraction of total time\n");
        printf("      - GPU with fast DRAM (H100 at 3.35 TB/s): phase 1 finishes faster\n");
        printf("      - Kernels that call reduction inside a loop (repeated use)\n\n");

        free(h_in); cudaFree(d_in); cudaFree(d_out);
    }

    // ── Shuffle instruction trace: N=32, show thread 0's accumulation ─────
    printf("[Trace] Warp shuffle on 32 elements = 1,2,3,...,32 (expected sum=528)\n");
    {
        const int    n  = 32;
        const size_t B  = n * sizeof(float);
        float *h_in     = (float *)malloc(B);
        for (int i = 0; i < n; i++) h_in[i] = (float)(i + 1);
        float ref = cpuSum(h_in, n);

        float *d_in, *d_out;
        CUDA_CHECK(cudaMalloc(&d_in,  B));
        CUDA_CHECK(cudaMalloc(&d_out, sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_in, h_in, B, cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaMemset(d_out, 0, sizeof(float)));
        warpShuffleReduction<<<1, BLOCK_SIZE>>>(d_in, d_out, n);
        CUDA_CHECK(cudaDeviceSynchronize());
        float r = 0.0f;
        CUDA_CHECK(cudaMemcpy(&r, d_out, sizeof(float), cudaMemcpyDeviceToHost));

        printf("  CPU sum=%.0f  GPU sum=%.0f  %s\n\n", ref, r,
               fabsf(r - ref) < 0.5f ? "PASS" : "FAIL");

        free(h_in); cudaFree(d_in); cudaFree(d_out);
    }

    // ── Profiling guidance ─────────────────────────────────────────────────
    printf("--- HOW TO PROFILE ---\n\n");
    printf("  .\\compile.bat 05-reduction-patterns\\kernel3.cu profile\n");
    printf("  Open kernel3-analysis.ncu-rep → Summary tab → 3 rows:\n");
    printf("    ID 0: treeReduction\n");
    printf("    ID 1: warpShuffleReduction\n");
    printf("    ID 2: pureWarpReduction\n\n");
    printf("  WHAT TO COMPARE:\n\n");
    printf("  [Section 3 — Shared Memory  ← clearest difference]\n");
    printf("    Shared Mem Transactions (loads + stores):\n");
    printf("      treeReduction:        ~1022 per block × 96 blocks ≈ 98,112 total\n");
    printf("      warpShuffleReduction: ~992  per block × 96 blocks ≈ 95,232 total\n");
    printf("      pureWarpReduction:    16    per block × 96 blocks =  1,536 total\n");
    printf("      → 64x fewer shared memory ops. Look at the raw transaction count.\n");
    printf("    Bank Conflicts: 0 for all three\n\n");
    printf("  [Section 4 — Warp State]\n");
    printf("    treeReduction:\n");
    printf("      Active Threads Per Warp < 32 (divergence at stride 16,8,4,2,1)\n");
    printf("      Barrier stalls: 8 __syncthreads calls per block\n");
    printf("    warpShuffleReduction:\n");
    printf("      Active Threads Per Warp closer to 32 (no guard in shuffle loop)\n");
    printf("      Barrier stalls: 3 __syncthreads calls\n");
    printf("    pureWarpReduction:\n");
    printf("      Active Threads Per Warp = 32 in all shuffle stages\n");
    printf("      Barrier stalls: 1 __syncthreads call (only after 8 warp-sum stores)\n");
    printf("      smsp__thread_inst_executed_per_inst_executed → closest to 1.0\n\n");
    printf("  [Section 2 — Memory Workload]\n");
    printf("    All three: Sectors/Request = 4 (coalesced Phase 1 global loads)\n");
    printf("    All three: DRAM Throughput ≈ same (reading 64 MB is unavoidable)\n");
    printf("    pureWarpReduction: lower L1 Shared Throughput (16 vs ~1000 ops)\n\n");
    printf("  [The key takeaway]\n");
    printf("    pureWarpReduction minimizes shared memory — only 8 floats ever touch it.\n");
    printf("    At N=16M this doesn't change the duration much (DRAM-bound).\n");
    printf("    Run the divergence check to see Active Threads/Warp clearly:\n");
    printf("      .\\compile.bat 05-reduction-patterns\\kernel3.cu divergence\n");

    return 0;
}
