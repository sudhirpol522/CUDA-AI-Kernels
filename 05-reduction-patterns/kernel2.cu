// ============================================================================
// REDUCTION PATTERNS — kernel2.cu
// Topic: Shared Memory Reduction + Padding + Multi-Stage (teaching file)
//
// THREE approaches in this file — all benchmarked together:
//
//   Kernel 1: sharedMemReduction        — baseline (atomicAdd at end)
//   Kernel 2: sharedMemReductionPadded  — padded shared memory
//   Kernel 3: multiStageReduction       — two-pass: no atomics in pass 1
//             combinePartialSums        — pass 2: reduces the 96 partial sums
//
// Compile:   .\compile.bat 05-reduction-patterns\kernel2.cu
// Profile:   .\compile.bat 05-reduction-patterns\kernel2.cu profile
//            Open kernel2-analysis.ncu-rep → Summary tab → four rows.
// Banks:     .\compile.bat 05-reduction-patterns\kernel2.cu banks
//
// ─────────────────────────────────────────────────────────────────────────────
// WHAT IS A BANK CONFLICT?
//
//   GPU shared memory is physically divided into 32 banks, each 4 bytes wide.
//   Element index → bank number = index % 32
//
//   So:
//     sdata[0]  → bank 0    sdata[32] → bank 0    sdata[64] → bank 0
//     sdata[1]  → bank 1    sdata[33] → bank 1    sdata[65] → bank 1
//     sdata[31] → bank 31   sdata[63] → bank 31
//
//   A bank conflict happens when TWO THREADS IN THE SAME WARP read/write
//   DIFFERENT addresses that map to the SAME BANK in the SAME instruction.
//   The GPU serializes those accesses — a 2-way conflict = 2× slower.
//
//   Exception: if ALL threads read the SAME address → "broadcast" → no conflict.
//
// ─────────────────────────────────────────────────────────────────────────────
// DOES SEQUENTIAL REDUCTION HAVE BANK CONFLICTS?
//
//   The key instruction is:  sdata[tid] += sdata[tid + stride]
//
//   At stride=128, warp 0 (threads 0-31) executes the LOAD sdata[tid+128]:
//     Thread 0  → sdata[128] → bank 0
//     Thread 1  → sdata[129] → bank 1
//     Thread 2  → sdata[130] → bank 2
//     ...
//     Thread 31 → sdata[159] → bank 31
//     → All 32 threads hit 32 different banks → ZERO conflicts ✓
//
//   This holds for ALL stride values ≥ 32: consecutive threads always read
//   consecutive addresses, which always map to banks 0,1,...,31 in order.
//
//   For stride < 32 (last 5 rounds): only threads 0..(stride-1) are active.
//   Still no conflict — they access consecutive sdata entries.
//
//   CONCLUSION: Sequential 1D reduction has 0 bank conflicts WITHOUT padding.
//   Padding adds complexity for zero benefit here.
//   Nsight will confirm: Bank Conflicts = 0 for BOTH kernels.
//
// ─────────────────────────────────────────────────────────────────────────────
// WHERE PADDING ACTUALLY HELPS (the 2D matrix case):
//
//   __shared__ float tile[32][32];   // 32*32 = 1024 floats, row-major
//
//   Column access:  tile[row][threadIdx.x]  — consecutive columns, banks 0-31 ✓
//   Row access:     tile[threadIdx.x][col]  — stride 32 between rows
//     → tile[0][col] → bank col%32
//     → tile[1][col] → bank (32+col)%32 = col%32   ← SAME BANK as tile[0][col]!
//     → tile[2][col] → same bank again...
//     → 32-way conflict! Every thread hits the same bank.
//
//   FIX: __shared__ float tile[32][33];  // +1 column padding
//     → tile[0][col] → bank col%32
//     → tile[1][col] → bank (33+col)%32 = (col+1)%32  ← shifted by 1 ✓
//     → tile[2][col] → bank (66+col)%32 = (col+2)%32  ← shifted by 2 ✓
//     → All threads now hit different banks.
//
//   Our 1D reduction never accesses with stride 32, so padding does nothing.
//   The padding kernel below is a teaching exercise — profile it to confirm.
//
// ─────────────────────────────────────────────────────────────────────────────
// WHAT NSIGHT WILL SHOW (for BOTH kernels):
//
//   Shared Memory → Bank Conflicts Load  = 0
//   Shared Memory → Bank Conflicts Store = 0
//   Warp State    → Long Scoreboard (Phase 1, waiting for DRAM)  ← expected
//   Warp State    → Active Threads Per Warp drops for stride < 32 ← next fix
//   Timing        → both kernels nearly identical (padding adds nothing)
//
// ============================================================================

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE   256       // threads per block — power of 2 required for tree reduction
#define FIXED_BLOCKS  96       // 24 SMs × 4 blocks — keeps all SMs busy with no tail waste

// Padding: insert 1 dummy slot every 32 elements to shift bank assignments.
// 256 elements ÷ 32 = 8 groups → 8 padding slots → 264 total physical slots.
// Formula: BLOCK_SIZE + BLOCK_SIZE/32
#define PADDED_SIZE  (BLOCK_SIZE + BLOCK_SIZE / 32)   // 256 + 8 = 264

// ============================================================================
// KERNEL: sharedMemReduction
//
// Parameters:
//   input  — array of N floats on the GPU
//   output — single float on the GPU where the total sum accumulates
//   n      — number of elements in input
// ============================================================================
__global__ void sharedMemReduction(float *input, float *output, int n) {

    // ── Shared memory declaration ─────────────────────────────────────────
    // __shared__ puts this array in the SM's fast scratchpad (~5 cycle access).
    // Size = BLOCK_SIZE floats = 256 × 4 = 1 KB per block.
    // Every block gets its OWN copy — they do NOT share this memory with each other.
    __shared__ float sharedData[BLOCK_SIZE];

    int tid        = threadIdx.x;                       // 0..255 inside this block
    int idx        = blockIdx.x * blockDim.x + tid;    // global thread index
    int gridStride = blockDim.x * gridDim.x;           // total threads in the grid
                                                        // = FIXED_BLOCKS * BLOCK_SIZE

    // ── Phase 1: Grid-stride load into a register ─────────────────────────
    //
    // Instead of one thread handling one element, each thread walks through
    // the ENTIRE array in steps of gridStride, summing into a local register.
    //
    // Example with N=1,000,000, FIXED_BLOCKS=96, BLOCK_SIZE=256:
    //   gridStride = 96 * 256 = 24,576
    //   Thread 0 reads: input[0], input[24576], input[49152], ...  (~40 loads)
    //   Thread 1 reads: input[1], input[24577], input[49153], ...
    //   Thread 255 reads: input[255], input[24831], ...
    //
    // Coalescing check (important for Nsight):
    //   In every iteration, threadIdx.x drives the column offset.
    //   → Consecutive threads read consecutive addresses → 1 cache line per warp.
    //   → Sectors/Request = 4  ← the optimal value you want to see in Nsight.
    //
    // "sum" lives in a REGISTER — no memory writes at all during this phase.
    float sum = 0.0f;
    for (int i = idx; i < n; i += gridStride) {
        sum += input[i];     // coalesced global load, result stored in register
    }

    // ── Store register sum into shared memory ─────────────────────────────
    // Each thread writes its partial sum into its own slot.
    // After this, sharedData[0..255] holds the partial sums for this block.
    sharedData[tid] = sum;

    // Barrier: wait until ALL 256 threads in this block have written.
    // Without this, some threads might read stale/uninitialized values below.
    __syncthreads();

    // ── Phase 2: Tree reduction in shared memory ──────────────────────────
    //
    // We halve the active threads each round.
    // stride=128 → threads 0..127 each add from the upper half:
    //   sharedData[0]   += sharedData[128]
    //   sharedData[1]   += sharedData[129]
    //   ...
    //   sharedData[127] += sharedData[255]
    // stride=64  → threads 0..63 each add: sharedData[k] += sharedData[k+64]
    // stride=32  → threads 0..31  (one full warp)
    // stride=16  → threads 0..15  (half a warp — divergent here!)
    // stride=8   → threads 0..7
    // stride=4   → threads 0..3
    // stride=2   → threads 0,1
    // stride=1   → thread 0 only → sdata[0] = total for this block
    //
    // Why sequential addressing (tid < stride) is better than
    // interleaved addressing (tid % (2*stride) == 0):
    //   Sequential: whole warps go idle at once → no intra-warp divergence
    //               for strides >= 32.
    //   Interleaved: at stride=1, threads 0,2,4,... active, 1,3,5,... idle
    //                → EVERY warp is half-active → 2× slower.
    //
    // Still divergent? Yes — for stride < 32 (the last 5 iterations),
    // only half of warp 0 is active at a time. kernel1.cu's unrolledReduction
    // fixes this using "warp unrolling" (the volatile trick).
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sharedData[tid] += sharedData[tid + stride];  // ~5 cycles (shared mem)
        }
        // Sync before the next stride: wait until all writes in this round finish.
        // Without this, a thread might read sharedData[k+stride] before another
        // thread has updated sharedData[k+stride] from the previous round.
        __syncthreads();
    }

    // ── Write block result to global memory ───────────────────────────────
    // Only thread 0 of each block does this.
    // sdata[0] now holds the sum of ALL elements this block processed.
    // atomicAdd ensures multiple blocks don't corrupt each other's writes.
    // With FIXED_BLOCKS=96 we only have 96 atomic operations total —
    // much less contention than if we launched one block per 256 elements.
    if (tid == 0) {
        atomicAdd(output, sharedData[0]);
    }
}

// ============================================================================
// KERNEL 2: sharedMemReductionPadded
//
// Identical logic to Kernel 1, but uses a padded shared memory array.
//
// PADDING TECHNIQUE:
//   Physical array size = PADDED_SIZE = 264 slots (not 256).
//   Logical thread index tid maps to physical slot: tid + tid/32
//
//   Memory layout (showing groups of 32):
//     tid  0.. 31 → slots  0.. 31   (pad slot = 32,  never touched)
//     tid 32.. 63 → slots 33.. 64   (pad slot = 65,  never touched)
//     tid 64.. 95 → slots 66.. 97   (pad slot = 98,  never touched)
//     tid 96..127 → slots 99..130   (pad slot = 131, never touched)
//     tid128..159 → slots132..163   (pad slot = 164, never touched)
//     tid160..191 → slots165..196   (pad slot = 197, never touched)
//     tid192..223 → slots198..229   (pad slot = 230, never touched)
//     tid224..255 → slots231..262   (slot 263 = unused tail)
//
//   Effect on bank assignment (with padding):
//     tid  0 → slot   0 → bank 0     |  tid  0 partner at stride=128:
//     tid 32 → slot  33 → bank 1     |    slot (128+4)=132 → bank 4
//     tid 64 → slot  66 → bank 2     |
//     tid 96 → slot  99 → bank 3     |  Without padding: partner slot 128 → bank 0
//    tid 128 → slot 132 → bank 4     |  With padding:    partner slot 132 → bank 4
//
//   FOR THIS KERNEL: bank conflicts were already 0 without padding.
//   Padding shifts banks but doesn't remove a problem that didn't exist.
//   Nsight will show Bank Conflicts = 0 for BOTH kernels.
//   Timing will be nearly identical.
//
//   THIS IS THE LESSON: always profile first, then decide if a "fix" is needed.
//   From the profiling guide: "Don't fix bank conflicts that don't exist."
// ============================================================================
__global__ void sharedMemReductionPadded(float *input, float *output, int n) {

    // Padded array: 264 slots instead of 256.
    // The 8 padding slots (indices 32, 65, 98, 131, 164, 197, 230, 263) are
    // never written or read — they exist only to shift bank assignments.
    __shared__ float sharedData[PADDED_SIZE];

    int tid        = threadIdx.x;
    int idx        = blockIdx.x * blockDim.x + tid;
    int gridStride = blockDim.x * gridDim.x;

    // Phase 1: identical grid-stride load (no padding involved yet — registers only)
    float sum = 0.0f;
    for (int i = idx; i < n; i += gridStride)
        sum += input[i];

    // Store with padded index: tid + tid/32
    //   tid=0:   0 + 0/32 = 0 + 0 = 0     (group 0: offset 0)
    //   tid=31:  31 + 31/32 = 31 + 0 = 31
    //   tid=32:  32 + 32/32 = 32 + 1 = 33  (group 1: offset 1 skips slot 32)
    //   tid=63:  63 + 63/32 = 63 + 1 = 64
    //   tid=64:  64 + 64/32 = 64 + 2 = 66  (group 2: offset 2 skips slots 32,65)
    //   tid=255: 255 + 255/32 = 255 + 7 = 262
    int paddedTid = tid + tid / 32;
    sharedData[paddedTid] = sum;
    __syncthreads();

    // Phase 2: tree reduction — BOTH indices must be translated to padded slots.
    //
    //   Without padding: sdata[tid] += sdata[tid + stride]
    //   With padding:    sdata[myIdx] += sdata[partnerIdx]
    //     where myIdx      = tid            + tid/32
    //     and   partnerIdx = (tid + stride) + (tid + stride)/32
    //
    //   The /32 term is the key: it inserts the correct number of padding
    //   gaps for whichever group the index falls in.
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            int myIdx      = tid               + tid / 32;
            int partnerIdx = (tid + stride)    + (tid + stride) / 32;
            sharedData[myIdx] += sharedData[partnerIdx];
        }
        __syncthreads();
    }

    // Thread 0: paddedTid = 0 + 0/32 = 0 → sharedData[0] holds the block sum ✓
    if (tid == 0)
        atomicAdd(output, sharedData[0]);
}

// ============================================================================
// KERNEL 3a: multiStageReduction  (Pass 1 of 2)
//
// WHY MULTI-STAGE?
//   Kernel 1 uses atomicAdd at the very end.  With 96 blocks, 96 threads all
//   race to atomicAdd into ONE address in L2 cache.  The GPU serializes them —
//   those 96 operations happen one at a time.  For 96 blocks that cost is tiny.
//   But if you launched 65,536 blocks (N=16M, one block per 256 elements),
//   you'd have 65,536 serialized atomic operations — a real bottleneck.
//
//   Multi-stage solves this by splitting into two kernel launches:
//     Pass 1: each block reduces its portion and writes to partialSums[blockIdx.x]
//             → DIFFERENT addresses → no serialization at all.
//     Pass 2: a tiny 1-block kernel sums the numBlocks partial sums → one result.
//
// HOW IT WORKS (Pass 1):
//   Identical grid-stride load and shared memory tree reduction as Kernel 1.
//   The ONLY difference: instead of atomicAdd(output, sdata[0]),
//   we do     partialSums[blockIdx.x] = sdata[0]
//   Each block writes to its OWN slot → perfectly parallel, zero contention.
//
//   After Pass 1 completes:
//     partialSums[0]  = sum of elements processed by block 0
//     partialSums[1]  = sum of elements processed by block 1
//     ...
//     partialSums[95] = sum of elements processed by block 95
//
// CONSTRAINT: numBlocks (FIXED_BLOCKS=96) must be ≤ BLOCK_SIZE (256)
//   so that Pass 2 can reduce everything in a single block.
//   96 ≤ 256 ✓
//
// NSIGHT — what Pass 1 shows:
//   Identical profile to sharedMemReduction (Kernel 1) — same access pattern.
//   No atomicAdd in this kernel → no L2 serialization stall at the end.
// ============================================================================
__global__ void multiStageReduction(float *input, float *partialSums, int n) {
    __shared__ float sharedData[BLOCK_SIZE];

    int tid        = threadIdx.x;
    int idx        = blockIdx.x * blockDim.x + tid;
    int gridStride = blockDim.x * gridDim.x;

    // ── Stage 1: Grid-stride accumulation into a register ─────────────────
    // Each thread walks through the entire array collecting its portion.
    // No global writes here — sum lives entirely in a register.
    float sum = 0.0f;
    for (int i = idx; i < n; i += gridStride)
        sum += input[i];

    sharedData[tid] = sum;
    __syncthreads();

    // ── Stage 2: Block-level reduction (sequential addressing) ─────────────
    // Exactly the same tree reduction as Kernel 1.
    // stride halves each round; whole warps go idle → no divergence for stride≥32.
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride)
            sharedData[tid] += sharedData[tid + stride];
        __syncthreads();
    }

    // ── Write this block's result to its OWN slot — no atomic needed ───────
    // partialSums has FIXED_BLOCKS=96 elements, one per block.
    // All 96 writes go to different addresses → hardware can do them in parallel.
    if (tid == 0)
        partialSums[blockIdx.x] = sharedData[0];
}

// ============================================================================
// KERNEL 3b: combinePartialSums  (Pass 2 of 2)
//
// Launched as: combinePartialSums<<<1, BLOCK_SIZE>>>(d_partials, d_out, FIXED_BLOCKS)
//   → 1 block, 256 threads
//
// Input: partialSums[0..95]  (96 values, one per block from Pass 1)
// Output: *output             (the single final sum)
//
// HOW IT LOADS:
//   We have 256 threads but only 96 partial sums.
//   Threads 0..95 each load one partial sum.
//   Threads 96..255 load 0.0f — they contribute nothing to the sum.
//   After __syncthreads(), sharedData[0..95] = block sums, rest = 0.
//
// HOW IT REDUCES:
//   Same sequential tree reduction.
//   At stride=128: threads 0..127 add. Threads 0..95 add real+zero values.
//                  Threads 96..127 add two zeros → no harm.
//   Eventually sharedData[0] = total sum of all 96 partial sums.
//
// HOW LONG DOES THIS TAKE?
//   96 floats is tiny — 96×4 = 384 bytes, fits entirely in L1 cache.
//   log2(256) = 8 rounds of reduction → negligible time (< 0.01 ms).
//   This pass is essentially free compared to the 64 MB read in Pass 1.
//
// NSIGHT — what Pass 2 shows:
//   Very short duration.  No significant DRAM traffic.
//   The reduction tree will be visible but tiny.
// ============================================================================
__global__ void combinePartialSums(float *partialSums, float *output, int numBlocks) {
    __shared__ float sharedData[BLOCK_SIZE];

    int tid = threadIdx.x;

    // Load partial sums — only threads 0..(numBlocks-1) have real data.
    // The rest get 0.0f so they don't pollute the sum.
    // (numBlocks=96 is much less than BLOCK_SIZE=256 — this is intentional.)
    sharedData[tid] = (tid < numBlocks) ? partialSums[tid] : 0.0f;
    __syncthreads();

    // Same sequential tree reduction as always.
    // Works correctly even though most slots in the upper half are 0.0f:
    //   stride=128: threads 0..127 are active.
    //     Threads 0..95   add sharedData[tid] += sharedData[tid+128]
    //       where sharedData[tid+128] = 0.0f for tid>=0 (128+0=128 > 96)
    //       → adds zero: safe, sum unchanged.
    //     Threads 96..127 add sharedData[96..127] += sharedData[224..255]
    //       → all zeros += zeros → still zero.
    //   ... after all rounds, sharedData[0] = sum of partialSums[0..95].
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride)
            sharedData[tid] += sharedData[tid + stride];
        __syncthreads();
    }

    // Thread 0 writes the final answer — a single store, no atomic needed
    // because there is only ONE block in this kernel launch.
    if (tid == 0)
        *output = sharedData[0];
}

// ============================================================================
// CPU reference for correctness check
// Uses double to avoid float accumulation error on large N
// ============================================================================
float cpuSum(const float *arr, int n) {
    double acc = 0.0;
    for (int i = 0; i < n; i++) acc += (double)arr[i];
    return (float)acc;
}

// ============================================================================
// Error-checking macro
// Wraps any CUDA call and prints the error + line number if it fails
// ============================================================================
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error at %s:%d — %s\n",                     \
                    __FILE__, __LINE__, cudaGetErrorString(err));                \
            exit(1);                                                            \
        }                                                                       \
    } while (0)

// ============================================================================
// Timing helper — warms up, then times one kernel launch, returns ms
// ============================================================================
static float timeKernel(void (*launchFn)(float*, float*, int),
                        float *d_in, float *d_out, int n) {
    // Warmup (GPU pipeline cold on first launch)
    CUDA_CHECK(cudaMemset(d_out, 0, sizeof(float)));
    launchFn(d_in, d_out, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Timed run
    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));
    CUDA_CHECK(cudaMemset(d_out, 0, sizeof(float)));
    CUDA_CHECK(cudaEventRecord(t0));
    launchFn(d_in, d_out, n);
    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
    CUDA_CHECK(cudaEventDestroy(t0));
    CUDA_CHECK(cudaEventDestroy(t1));
    return ms;
}

static void launchBaseline(float *d_in, float *d_out, int n) {
    sharedMemReduction<<<FIXED_BLOCKS, BLOCK_SIZE>>>(d_in, d_out, n);
}
static void launchPadded(float *d_in, float *d_out, int n) {
    sharedMemReductionPadded<<<FIXED_BLOCKS, BLOCK_SIZE>>>(d_in, d_out, n);
}

// ============================================================================
// main
// ============================================================================
int main() {
    printf("=== SHARED MEMORY REDUCTION: Baseline vs Padded vs Multi-Stage (kernel2.cu) ===\n\n");
    printf("Shared memory sizes:\n");
    printf("  sharedMemReduction        : %d floats = %d bytes (1 KB/block)\n",
           BLOCK_SIZE, BLOCK_SIZE * (int)sizeof(float));
    printf("  sharedMemReductionPadded  : %d floats = %d bytes (1.03 KB/block)\n",
           PADDED_SIZE, PADDED_SIZE * (int)sizeof(float));
    printf("  multiStageReduction       : %d floats = %d bytes (same as baseline)\n",
           BLOCK_SIZE, BLOCK_SIZE * (int)sizeof(float));
    printf("  Padding formula: tid + tid/32  (inserts 1 gap every 32 elements)\n\n");

    // ── Correctness: all three approaches on N=1024 (all 1s, expected=1024) ─
    {
        const int    n = 1024;
        const size_t B = n * sizeof(float);
        float *h_in    = (float *)malloc(B);
        for (int i = 0; i < n; i++) h_in[i] = 1.0f;
        float ref = cpuSum(h_in, n);

        float *d_in, *d_out, *d_partials;
        CUDA_CHECK(cudaMalloc(&d_in,       B));
        CUDA_CHECK(cudaMalloc(&d_out,      sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_partials, FIXED_BLOCKS * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_in, h_in, B, cudaMemcpyHostToDevice));

        float r1 = 0.0f, r2 = 0.0f, r3 = 0.0f;

        CUDA_CHECK(cudaMemset(d_out, 0, sizeof(float)));
        sharedMemReduction<<<FIXED_BLOCKS, BLOCK_SIZE>>>(d_in, d_out, n);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(&r1, d_out, sizeof(float), cudaMemcpyDeviceToHost));

        CUDA_CHECK(cudaMemset(d_out, 0, sizeof(float)));
        sharedMemReductionPadded<<<FIXED_BLOCKS, BLOCK_SIZE>>>(d_in, d_out, n);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(&r2, d_out, sizeof(float), cudaMemcpyDeviceToHost));

        // Multi-stage: two kernel launches, no atomicAdd in either
        multiStageReduction<<<FIXED_BLOCKS, BLOCK_SIZE>>>(d_in, d_partials, n);
        combinePartialSums<<<1, BLOCK_SIZE>>>(d_partials, d_out, FIXED_BLOCKS);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(&r3, d_out, sizeof(float), cudaMemcpyDeviceToHost));

        printf("[Correctness N=%-6d]  ref=%.0f\n", n, ref);
        printf("  baseline     : GPU=%.0f  %s\n", r1, fabsf(r1 - ref) < 0.5f ? "PASS" : "FAIL");
        printf("  padded       : GPU=%.0f  %s\n", r2, fabsf(r2 - ref) < 0.5f ? "PASS" : "FAIL");
        printf("  multi-stage  : GPU=%.0f  %s\n", r3, fabsf(r3 - ref) < 0.5f ? "PASS" : "FAIL");

        free(h_in); cudaFree(d_in); cudaFree(d_out); cudaFree(d_partials);
    }

    // ── Correctness: non-power-of-2 N ─────────────────────────────────────
    {
        const int    n = 999983;
        const size_t B = n * sizeof(float);
        float *h_in    = (float *)malloc(B);
        for (int i = 0; i < n; i++) h_in[i] = 1.0f;
        float ref = cpuSum(h_in, n);

        float *d_in, *d_out, *d_partials;
        CUDA_CHECK(cudaMalloc(&d_in,       B));
        CUDA_CHECK(cudaMalloc(&d_out,      sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_partials, FIXED_BLOCKS * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_in, h_in, B, cudaMemcpyHostToDevice));

        float r1 = 0.0f, r2 = 0.0f, r3 = 0.0f;

        CUDA_CHECK(cudaMemset(d_out, 0, sizeof(float)));
        sharedMemReduction<<<FIXED_BLOCKS, BLOCK_SIZE>>>(d_in, d_out, n);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(&r1, d_out, sizeof(float), cudaMemcpyDeviceToHost));

        CUDA_CHECK(cudaMemset(d_out, 0, sizeof(float)));
        sharedMemReductionPadded<<<FIXED_BLOCKS, BLOCK_SIZE>>>(d_in, d_out, n);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(&r2, d_out, sizeof(float), cudaMemcpyDeviceToHost));

        multiStageReduction<<<FIXED_BLOCKS, BLOCK_SIZE>>>(d_in, d_partials, n);
        combinePartialSums<<<1, BLOCK_SIZE>>>(d_partials, d_out, FIXED_BLOCKS);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(&r3, d_out, sizeof(float), cudaMemcpyDeviceToHost));

        printf("\n[Correctness N=%-6d]  ref=%.0f  (non-power-of-2)\n", n, ref);
        printf("  baseline     : GPU=%.0f  %s\n", r1, fabsf(r1 - ref) < 0.5f ? "PASS" : "FAIL");
        printf("  padded       : GPU=%.0f  %s\n", r2, fabsf(r2 - ref) < 0.5f ? "PASS" : "FAIL");
        printf("  multi-stage  : GPU=%.0f  %s\n\n", r3, fabsf(r3 - ref) < 0.5f ? "PASS" : "FAIL");

        free(h_in); cudaFree(d_in); cudaFree(d_out); cudaFree(d_partials);
    }

    // ── Benchmark: N=16M (~64 MB), all three approaches timed ────────────
    {
        const int    n  = 1 << 24;   // 16,777,216 elements = 64 MB
        const size_t B  = (size_t)n * sizeof(float);

        float *h_in = (float *)malloc(B);
        for (int i = 0; i < n; i++) h_in[i] = 1.0f;

        float *d_in, *d_out, *d_partials;
        CUDA_CHECK(cudaMalloc(&d_in,       B));
        CUDA_CHECK(cudaMalloc(&d_out,      sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_partials, FIXED_BLOCKS * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_in, h_in, B, cudaMemcpyHostToDevice));

        float ms1 = timeKernel(launchBaseline, d_in, d_out, n);
        float ms2 = timeKernel(launchPadded,   d_in, d_out, n);

        // Multi-stage timing: wrap BOTH kernel launches in ONE pair of events.
        // This measures the true cost of the two-pass approach end-to-end.
        // Warmup first:
        multiStageReduction<<<FIXED_BLOCKS, BLOCK_SIZE>>>(d_in, d_partials, n);
        combinePartialSums<<<1, BLOCK_SIZE>>>(d_partials, d_out, FIXED_BLOCKS);
        CUDA_CHECK(cudaDeviceSynchronize());
        // Timed run:
        cudaEvent_t t0, t1;
        CUDA_CHECK(cudaEventCreate(&t0));
        CUDA_CHECK(cudaEventCreate(&t1));
        CUDA_CHECK(cudaEventRecord(t0));
        multiStageReduction<<<FIXED_BLOCKS, BLOCK_SIZE>>>(d_in, d_partials, n);
        combinePartialSums<<<1, BLOCK_SIZE>>>(d_partials, d_out, FIXED_BLOCKS);
        CUDA_CHECK(cudaEventRecord(t1));
        CUDA_CHECK(cudaEventSynchronize(t1));
        float ms3 = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms3, t0, t1));
        CUDA_CHECK(cudaEventDestroy(t0));
        CUDA_CHECK(cudaEventDestroy(t1));

        float bw1 = (float)B / (ms1 * 1e6f);
        float bw2 = (float)B / (ms2 * 1e6f);
        float bw3 = (float)B / (ms3 * 1e6f);

        printf("[Benchmark N=%d (~64 MB)]\n", n);
        printf("  baseline     : %6.3f ms  BW=%5.1f GB/s\n",             ms1, bw1);
        printf("  padded       : %6.3f ms  BW=%5.1f GB/s  (%.2fx)\n",    ms2, bw2, ms1/ms2);
        printf("  multi-stage  : %6.3f ms  BW=%5.1f GB/s  (%.2fx)\n",    ms3, bw3, ms1/ms3);
        printf("  (theoretical peak ~192 GB/s on GTX 1660Ti)\n\n");
        printf("  multi-stage includes: pass1 (96 blocks, grid-stride) +\n");
        printf("                        pass2 (1 block, reduces 96 partial sums)\n");
        printf("  pass2 is ~0.001 ms — nearly invisible. All time is in pass1.\n\n");

        free(h_in); cudaFree(d_in); cudaFree(d_out); cudaFree(d_partials);
    }

    // ── Profiling guidance ─────────────────────────────────────────────────
    printf("--- HOW TO PROFILE ---\n\n");
    printf("  .\\compile.bat 05-reduction-patterns\\kernel2.cu profile\n");
    printf("  Open kernel2-analysis.ncu-rep → Summary tab → 4 rows:\n");
    printf("    ID 0: sharedMemReduction       (baseline)\n");
    printf("    ID 1: sharedMemReductionPadded (padded)\n");
    printf("    ID 2: multiStageReduction      (pass 1 — the big kernel)\n");
    printf("    ID 3: combinePartialSums        (pass 2 — tiny, ~96 elements)\n\n");
    printf("  WHAT TO COMPARE:\n\n");
    printf("  [Shared Memory section]\n");
    printf("    IDs 0,1,2: Bank Conflicts Load  = 0  — sequential addressing, no conflict\n");
    printf("    ID 3:      Bank Conflicts Load  = 0  — only 96 elements, trivial\n");
    printf("    → Lesson: profile FIRST. Padding adds nothing for 1D sequential.\n\n");
    printf("  [Warp State section  ← most important]\n");
    printf("    IDs 0,1,2: Long Scoreboard (Phase 1 — waiting for DRAM)\n");
    printf("    IDs 0,1,2: Active Threads/Warp drops below 32 at stride<32\n");
    printf("    ID 3:      Nearly instant — 96 floats fit in L1, no DRAM stall\n\n");
    printf("  [Key multi-stage insight in Nsight]\n");
    printf("    Compare ID 0 vs ID 2 — they should look nearly identical.\n");
    printf("    The difference: ID 0 has a tiny L2 atomic at the end (96 contentions).\n");
    printf("    ID 2 writes to 96 separate addresses — no serialization.\n");
    printf("    At FIXED_BLOCKS=96 the benefit is small. But if you launched\n");
    printf("    65,536 blocks (one per 256-element chunk of 16M), single-kernel\n");
    printf("    would have 65,536 serialized atomics — multi-stage wins clearly.\n\n");
    printf("  WHERE PADDING ACTUALLY HELPS:\n");
    printf("    2D shared arrays with power-of-2 width:\n");
    printf("      __shared__ float tile[32][32];  // column access → 32-way conflict!\n");
    printf("      __shared__ float tile[32][33];  // +1 pad → conflict gone ✓\n");
    printf("    NOT needed for 1D sequential reduction.\n");
    printf("    Rule: profile FIRST, then fix. Never add padding speculatively.\n");

    return 0;
}
