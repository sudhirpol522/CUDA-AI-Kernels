// ============================================================================
// REDUCTION PATTERNS — kernel5.cu
// Topic: Warp Shuffle Broadcast + Fused Reduce-Normalize
//
// THREE kernels:
//   Kernel 1: naiveBroadcast         — all threads read sharedData[0] (baseline)
//   Kernel 2: warpReductionBroadcast — only lane 0 reads shmem, shuffle distributes
//   Kernel 3: fuseReduceNormalize    — reduction + normalization in ONE kernel pass
//
// Compile:   .\compile.bat 05-reduction-patterns\kernel5.cu
// Profile:   .\compile.bat 05-reduction-patterns\kernel5.cu profile
//            Open kernel5-analysis.ncu-rep → Summary tab → 3 rows.
//
// ─────────────────────────────────────────────────────────────────────────────
// TWO DIFFERENT SHUFFLE FUNCTIONS
//
//   __shfl_down_sync(mask, var, delta)    ← used for REDUCTION
//     Thread lane  receives value from thread (lane + delta).
//     Used in the reduction loop: each round merges adjacent sums.
//     After 5 rounds: only thread 0 has the correct total.
//
//   __shfl_sync(mask, var, srcLane)       ← used for BROADCAST
//     ALL threads receive the value of var FROM thread srcLane.
//     Every thread ends up with the same value: srcLane's var.
//     "One-to-all" distribution within the warp.
//
//   They are DIFFERENT instructions (SHFL.DOWN vs SHFL.IDX in SASS).
//
// ─────────────────────────────────────────────────────────────────────────────
// THE BROADCAST OPTIMIZATION
//
//   After reduction, sharedData[0] holds the block total.
//   All threads need this value. Naive approach: everyone reads sharedData[0].
//
//   Naive:                            Optimized:
//   ─────────────────────────         ──────────────────────────────────────
//   float r = sharedData[0];          float r = 0.0f;
//   // All 256 threads read shmem     if (laneId == 0)
//   // = 256 MIO reads per block          r = sharedData[0]; // 8 reads/block
//                                     r = __shfl_sync(0xffffffff, r, 0);
//                                     // lane 0 broadcasts to lanes 1-31
//                                     // = 248 MIO reads SAVED per block
//
//   Saving: 248 shared memory reads per block × 96 blocks = 23,808 reads avoided.
//   Nsight Section 3: lower Shared Mem Transaction count for the optimized kernel.
//
// ─────────────────────────────────────────────────────────────────────────────
// WHY BROADCAST MATTERS — THE AI KERNEL USE CASE
//
//   Softmax (simplified):
//     1. Compute sum = Σ exp(input[i])  ← reduction
//     2. output[i] = exp(input[i]) / sum ← every element needs sum
//
//   Without broadcast:
//     Pass 1: reduction kernel → sum stored to global memory
//     Pass 2: divide kernel   → each thread reads sum from global memory
//     = 2 kernel launches + 1 global memory round-trip for sum
//
//   With fused reduce+broadcast:
//     Pass 1: accumulate in registers (grid-stride)
//     Pass 2: tree reduce → sharedData[0]
//     Pass 3: __shfl_sync broadcast to all threads in block
//     Pass 4: each thread normalizes its own elements in-place
//     = 1 kernel launch, sum never touches global memory
//
//   cuDNN and Flash Attention use exactly this pattern.
//
// ─────────────────────────────────────────────────────────────────────────────
// WHAT NSIGHT WILL SHOW:
//
//   [Section 3 — Shared Memory]
//     naiveBroadcast:         Shared Mem Transactions high (all threads read shmem)
//     warpReductionBroadcast: lower (only 8 threads read shmem per broadcast)
//     fuseReduceNormalize:    similar to warpReductionBroadcast
//
//   [Section 4 — Warp State]
//     All kernels: Active Threads/Warp = 32 in shuffle stages (no divergence)
//     Barrier stalls: fuseReduceNormalize has 1 extra __syncthreads vs others
//
//   [Section 2 — Memory Workload]
//     fuseReduceNormalize: DRAM writes ≈ same as reads (writes output[i])
//     Others: only atomic write to output (negligible writes)
//
// ============================================================================

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE    256
#define WARP_SIZE      32
#define FIXED_BLOCKS   96    // 24 SMs × 4

// ============================================================================
// SHARED REDUCTION HELPER (inline, reused by all three kernels)
//
// Reduces sharedData[0..BLOCK_SIZE-1] down to sharedData[0].
// Uses hybrid: 3 rounds of shared memory tree, then 5 rounds of warp shuffle.
// After this function: sharedData[0] = block total.
//                      __syncthreads() is called internally as needed.
// Must be called with: all threads loaded sharedData[tid], then __syncthreads().
// ============================================================================
__device__ void blockReduce(float *sharedData, int tid) {
    // Tree reduction from BLOCK_SIZE → 32 (3 rounds)
    // Each round halves the active threads; whole warps go idle at once.
    for (int stride = BLOCK_SIZE / 2; stride >= WARP_SIZE; stride >>= 1) {
        if (tid < stride)
            sharedData[tid] += sharedData[tid + stride];
        __syncthreads();
    }

    // Warp shuffle for the final 5 rounds (stride 16 → 1)
    // Only warp 0 (threads 0-31) participates.
    // No __syncthreads() needed — all in lockstep.
    if (tid < WARP_SIZE) {
        float val = sharedData[tid];   // last shared memory read
        val += __shfl_down_sync(0xffffffff, val, 16);
        val += __shfl_down_sync(0xffffffff, val,  8);
        val += __shfl_down_sync(0xffffffff, val,  4);
        val += __shfl_down_sync(0xffffffff, val,  2);
        val += __shfl_down_sync(0xffffffff, val,  1);
        // Thread 0 stores the block total back to shared memory
        // for the broadcast phase that follows.
        if (tid == 0)
            sharedData[0] = val;
    }
    // Note: no __syncthreads() here intentionally.
    // Callers that need cross-warp visibility of sharedData[0]
    // must issue their own __syncthreads() after calling blockReduce().
}

// ============================================================================
// KERNEL 1: naiveBroadcast (baseline — all threads read sharedData[0])
//
// Reduction is identical to warpShuffleReduction (kernel3.cu).
// Broadcast is "naive": every thread reads sharedData[0] from shared mem.
// This is the reference to compare Shared Mem Transaction counts against.
// ============================================================================
__global__ void naiveBroadcast(float *input, float *output, int n) {
    __shared__ float sharedData[BLOCK_SIZE];

    int tid        = threadIdx.x;
    int idx        = blockIdx.x * blockDim.x + tid;
    int gridStride = blockDim.x * gridDim.x;

    // Phase 1: Grid-stride accumulation into a register
    float sum = 0.0f;
    for (int i = idx; i < n; i += gridStride)
        sum += input[i];
    sharedData[tid] = sum;
    __syncthreads();

    // Phase 2: Reduce to sharedData[0]
    blockReduce(sharedData, tid);
    __syncthreads();   // wait for thread 0's write to sharedData[0]

    // Phase 3: Naive broadcast — ALL 256 threads read from sharedData[0]
    // = 256 shared memory reads per block
    float result = sharedData[0];

    // Write result to every output position this thread owns
    for (int i = idx; i < n; i += gridStride)
        output[i] = result;
}

// ============================================================================
// KERNEL 2: warpReductionBroadcast (optimized — only lane 0 reads sharedData[0])
//
// BROADCAST OPTIMIZATION:
//   Instead of all 256 threads reading sharedData[0] (256 MIO reads),
//   only 8 threads (lane 0 of each warp) read from shared memory.
//   Then __shfl_sync distributes the value to the other 24 lanes per warp.
//   = 248 fewer shared memory reads per block.
//
// __shfl_sync(mask, var, srcLane):
//   ALL threads in the warp receive the value of var from thread srcLane.
//   This is a "one-to-all" broadcast within the warp.
//   srcLane=0 → lane 0 sends its value to lanes 1-31.
//   After the call: every thread has the same value (lane 0's var).
//
// ABOUT UNINITIALIZED 'result' FOR NON-LANE-0 THREADS:
//   result = 0.0f for lanes 1-31 before __shfl_sync.
//   __shfl_sync OVERWRITES result with lane 0's value.
//   The initial 0.0f is never used — it's overwritten immediately.
//   This is safe: the final value of result for every thread is sharedData[0].
// ============================================================================
__global__ void warpReductionBroadcast(float *input, float *output, int n) {
    __shared__ float sharedData[BLOCK_SIZE];

    int tid        = threadIdx.x;
    int idx        = blockIdx.x * blockDim.x + tid;
    int laneId     = tid % WARP_SIZE;       // 0..31 within this warp
    int gridStride = blockDim.x * gridDim.x;

    // Phase 1: Grid-stride accumulation (identical to naiveBroadcast)
    float sum = 0.0f;
    for (int i = idx; i < n; i += gridStride)
        sum += input[i];
    sharedData[tid] = sum;
    __syncthreads();

    // Phase 2: Reduce to sharedData[0]  (same as naiveBroadcast)
    blockReduce(sharedData, tid);
    __syncthreads();   // ensure sharedData[0] is visible to all threads

    // ── Phase 3: OPTIMIZED BROADCAST via __shfl_sync ──────────────────────
    //
    // Step A: Only lane 0 of each warp reads from shared memory.
    //         Lanes 1-31 start with 0.0f (value doesn't matter — it's overwritten).
    //         Total shared mem reads: 8 (one per warp) instead of 256.
    float result = 0.0f;
    if (laneId == 0)
        result = sharedData[0];

    // Step B: __shfl_sync broadcasts lane 0's result to lanes 1-31.
    //         mask = 0xffffffff → all 32 lanes of THIS warp participate.
    //         srcLane = 0 → lane 0's value is sent to everyone.
    //
    //         What happens per warp:
    //           lane 0: result = sharedData[0]  (set in Step A, confirmed by shuffle)
    //           lane 1: result = 0.0f → OVERWRITTEN with lane 0's sharedData[0]  ✓
    //           lane 2: result = 0.0f → OVERWRITTEN                               ✓
    //           ...
    //           lane 31: result = 0.0f → OVERWRITTEN                              ✓
    //         After: ALL 32 lanes have result = sharedData[0].
    //
    //         This runs for ALL 8 warps simultaneously.
    //         Each warp independently broadcasts from its own lane 0.
    //         All 8 lane-0s have the same value (they all read sharedData[0]).
    //         So all 256 threads end up with the same value. ✓
    result = __shfl_sync(0xffffffff, result, 0);

    // ── Phase 4: Every thread uses result in a register ────────────────────
    // No more shared memory access needed — result lives in a register.
    // Write to every output position owned by this thread.
    for (int i = idx; i < n; i += gridStride)
        output[i] = result;
}

// ============================================================================
// KERNEL 3: fuseReduceNormalize (reduction + normalization in one pass)
//
// USE CASE: Softmax denominator, Layer Norm, Batch Norm
//   Traditional two-pass:
//     Pass 1: reduction kernel → writes blockSum to global memory
//     Pass 2: divide kernel   → reads blockSum from global memory
//     Cost: 2 kernel launches + global memory round-trip
//
//   Fused one-pass (this kernel):
//     Phase 1: load each element to register AND accumulate for reduction
//     Phase 2: reduce to blockSum in sharedData[0]
//     Phase 3: broadcast blockSum to ALL threads via __shfl_sync
//     Phase 4: each thread normalizes its OWN element (input[i] / blockSum)
//     Cost: 1 kernel launch, blockSum never touches global memory
//
// HOW WE KEEP THE ORIGINAL VALUES:
//   In Phase 1, we load input[i] into a LOCAL ARRAY (vals[]).
//   Then in Phase 4, we divide vals[k] / blockSum for each k.
//   This avoids reading input[] a second time.
//
// CONSTRAINT: number of elements per thread = ELEMS_PER_THREAD (fixed here at 8).
//   Each thread processes 8 consecutive elements.
//   Total elements = FIXED_BLOCKS * BLOCK_SIZE * ELEMS_PER_THREAD.
//   For simplicity, we use a simple strided load, not grid-stride, so the
//   number of elements must divide evenly.
// ============================================================================
#define ELEMS_PER_THREAD 8   // each thread handles 8 consecutive elements

__global__ void fuseReduceNormalize(float *input, float *output, int n) {
    __shared__ float sharedData[BLOCK_SIZE];

    int tid         = threadIdx.x;
    int laneId      = tid % WARP_SIZE;
    int blockStart  = blockIdx.x * blockDim.x * ELEMS_PER_THREAD;

    // ── Phase 1: Load ELEMS_PER_THREAD elements per thread into registers ──
    //
    // Element i for thread tid: index = blockStart + tid + i * BLOCK_SIZE
    // This interleaved access is coalesced: threads 0-255 read 256 consecutive
    // floats at a time (one stride of BLOCK_SIZE per iteration).
    //
    // vals[] is a register array — stays in registers for the entire kernel.
    // NO global writes during Phase 1.
    float vals[ELEMS_PER_THREAD];
    float sum = 0.0f;
    for (int k = 0; k < ELEMS_PER_THREAD; k++) {
        int idx = blockStart + k * BLOCK_SIZE + tid;
        vals[k] = (idx < n) ? input[idx] : 0.0f;
        sum += vals[k];
    }

    // ── Phase 2: Store partial sum to shared mem, reduce to block total ────
    sharedData[tid] = sum;
    __syncthreads();

    blockReduce(sharedData, tid);
    __syncthreads();   // wait for thread 0 to write sharedData[0]

    // ── Phase 3: Broadcast block total using __shfl_sync ──────────────────
    // Same pattern as warpReductionBroadcast.
    float blockSum = 0.0f;
    if (laneId == 0)
        blockSum = sharedData[0];
    blockSum = __shfl_sync(0xffffffff, blockSum, 0);
    // Now blockSum = total block sum, live in every thread's register.

    // ── Phase 4: Normalize — each thread divides its stored values ─────────
    // vals[] was saved in Phase 1 and has never left the registers.
    // This division happens entirely in registers — no global reads for input.
    // Avoid divide-by-zero for edge case where all inputs are 0.
    float inv = (blockSum != 0.0f) ? (1.0f / blockSum) : 0.0f;
    for (int k = 0; k < ELEMS_PER_THREAD; k++) {
        int idx = blockStart + k * BLOCK_SIZE + tid;
        if (idx < n)
            output[idx] = vals[k] * inv;   // multiply by reciprocal = one FMA
    }
}

// ============================================================================
// CPU helpers
// ============================================================================
float cpuSum(const float *arr, int n) {
    double acc = 0.0;
    for (int i = 0; i < n; i++) acc += (double)arr[i];
    return (float)acc;
}

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d — %s\n",                    \
                    __FILE__, __LINE__, cudaGetErrorString(err));               \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

static float timeKernel(void (*fn)(float*, float*, int),
                        float *d_in, float *d_out, int n) {
    fn(d_in, d_out, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));
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

static void launchNaive(float *d_in, float *d_out, int n) {
    naiveBroadcast<<<FIXED_BLOCKS, BLOCK_SIZE>>>(d_in, d_out, n);
}
static void launchOptimized(float *d_in, float *d_out, int n) {
    warpReductionBroadcast<<<FIXED_BLOCKS, BLOCK_SIZE>>>(d_in, d_out, n);
}
static void launchFused(float *d_in, float *d_out, int n) {
    int blocks = (n + BLOCK_SIZE * ELEMS_PER_THREAD - 1) / (BLOCK_SIZE * ELEMS_PER_THREAD);
    fuseReduceNormalize<<<blocks, BLOCK_SIZE>>>(d_in, d_out, n);
}

// ============================================================================
// main
// ============================================================================
int main() {
    printf("=== WARP SHUFFLE BROADCAST + FUSED NORMALIZE (kernel5.cu) ===\n\n");
    printf("Broadcast:  only lane 0 of each warp reads shmem,\n");
    printf("            __shfl_sync distributes to all 32 lanes.\n");
    printf("Saved shmem reads per block: (256-8) = 248  (vs naive broadcast)\n\n");
    printf("Fuse pattern: accumulate → reduce → broadcast → normalize\n");
    printf("              all in ONE kernel launch, blockSum never hits DRAM\n\n");

    // ── Correctness: broadcast kernels ────────────────────────────────────
    // Use n = FIXED_BLOCKS * BLOCK_SIZE so each thread handles exactly 1 element.
    // All inputs = 1.0f → block sum = BLOCK_SIZE = 256.
    // Expected: ALL output values = 256.
    {
        const int    n  = FIXED_BLOCKS * BLOCK_SIZE;   // 24576, one per thread
        const size_t B  = n * sizeof(float);
        float *h_in     = (float *)malloc(B);
        float *h_out    = (float *)malloc(B);
        for (int i = 0; i < n; i++) h_in[i] = 1.0f;

        float *d_in, *d_out;
        CUDA_CHECK(cudaMalloc(&d_in,  B));
        CUDA_CHECK(cudaMalloc(&d_out, B));
        CUDA_CHECK(cudaMemcpy(d_in, h_in, B, cudaMemcpyHostToDevice));

        printf("[Correctness] n=%d, all 1s → expected block sum = %d\n", n, BLOCK_SIZE);

        // naiveBroadcast
        CUDA_CHECK(cudaMemset(d_out, 0, B));
        naiveBroadcast<<<FIXED_BLOCKS, BLOCK_SIZE>>>(d_in, d_out, n);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(h_out, d_out, B, cudaMemcpyDeviceToHost));
        bool ok1 = true;
        for (int i = 0; i < n; i++)
            if (fabsf(h_out[i] - BLOCK_SIZE) > 0.5f) { ok1 = false; break; }
        printf("  naiveBroadcast         : output[0]=%.0f  %s\n",
               h_out[0], ok1 ? "PASS (all values equal block sum)" : "FAIL");

        // warpReductionBroadcast
        CUDA_CHECK(cudaMemset(d_out, 0, B));
        warpReductionBroadcast<<<FIXED_BLOCKS, BLOCK_SIZE>>>(d_in, d_out, n);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(h_out, d_out, B, cudaMemcpyDeviceToHost));
        bool ok2 = true;
        for (int i = 0; i < n; i++)
            if (fabsf(h_out[i] - BLOCK_SIZE) > 0.5f) { ok2 = false; break; }
        printf("  warpReductionBroadcast : output[0]=%.0f  %s\n\n",
               h_out[0], ok2 ? "PASS (all values equal block sum)" : "FAIL");

        free(h_in); free(h_out); cudaFree(d_in); cudaFree(d_out);
    }

    // ── Correctness: fuseReduceNormalize ──────────────────────────────────
    // Each block of 256 threads processes 256×8 = 2048 elements.
    // All inputs = 1.0f → block sum = 2048 → normalized value = 1.0/2048.
    {
        const int blocks = 4;
        const int n      = blocks * BLOCK_SIZE * ELEMS_PER_THREAD;  // 4*256*8 = 8192
        const size_t B   = n * sizeof(float);
        float *h_in      = (float *)malloc(B);
        float *h_out     = (float *)malloc(B);
        for (int i = 0; i < n; i++) h_in[i] = 1.0f;
        float expectedNorm = 1.0f / (BLOCK_SIZE * ELEMS_PER_THREAD);   // 1/2048

        float *d_in, *d_out;
        CUDA_CHECK(cudaMalloc(&d_in,  B));
        CUDA_CHECK(cudaMalloc(&d_out, B));
        CUDA_CHECK(cudaMemcpy(d_in, h_in, B, cudaMemcpyHostToDevice));

        fuseReduceNormalize<<<blocks, BLOCK_SIZE>>>(d_in, d_out, n);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(h_out, d_out, B, cudaMemcpyDeviceToHost));

        bool ok = true;
        for (int i = 0; i < n; i++)
            if (fabsf(h_out[i] - expectedNorm) > 1e-6f) { ok = false; break; }

        printf("[Correctness] fuseReduceNormalize n=%d, all 1s\n", n);
        printf("  block sum = %d, expected normalized = 1/%d = %.6f\n",
               BLOCK_SIZE * ELEMS_PER_THREAD, BLOCK_SIZE * ELEMS_PER_THREAD, expectedNorm);
        printf("  output[0]=%.6f  %s\n\n",
               h_out[0], ok ? "PASS (all values = 1/blockSum)" : "FAIL");

        free(h_in); free(h_out); cudaFree(d_in); cudaFree(d_out);
    }

    // ── Benchmark: N=16M ──────────────────────────────────────────────────
    {
        const int    n  = 1 << 24;   // 16M elements
        const size_t B  = (size_t)n * sizeof(float);
        float *h_in     = (float *)malloc(B);
        for (int i = 0; i < n; i++) h_in[i] = 1.0f;

        float *d_in, *d_out;
        CUDA_CHECK(cudaMalloc(&d_in,  B));
        CUDA_CHECK(cudaMalloc(&d_out, B));
        CUDA_CHECK(cudaMemcpy(d_in, h_in, B, cudaMemcpyHostToDevice));

        float ms1 = timeKernel(launchNaive,     d_in, d_out, n);
        float ms2 = timeKernel(launchOptimized, d_in, d_out, n);
        float ms3 = timeKernel(launchFused,     d_in, d_out, n);

        // Bandwidth: both read n floats AND write n floats
        float bw1 = 2.0f * (float)B / (ms1 * 1e6f);
        float bw2 = 2.0f * (float)B / (ms2 * 1e6f);
        float bw3 = 2.0f * (float)B / (ms3 * 1e6f);

        printf("[Benchmark N=%d (~64 MB read + 64 MB write)]\n", n);
        printf("  naiveBroadcast         : %6.3f ms  BW=%5.1f GB/s\n", ms1, bw1);
        printf("  warpReductionBroadcast : %6.3f ms  BW=%5.1f GB/s  (%.2fx)\n",
               ms2, bw2, ms1/ms2);
        printf("  fuseReduceNormalize    : %6.3f ms  BW=%5.1f GB/s  (%.2fx)\n",
               ms3, bw3, ms1/ms3);
        printf("  (theoretical peak: ~192 GB/s read + 192 GB/s write on GTX 1660Ti)\n\n");

        printf("  All three read 64 MB (input) AND write 64 MB (output).\n");
        printf("  The reduction is a small fraction of total time.\n");
        printf("  The broadcast optimization (248 fewer shmem reads/block) is\n");
        printf("  visible in Section 3 of Nsight (transaction count), not in ms.\n\n");

        printf("  fuseReduceNormalize has an extra benefit NOT visible in timing:\n");
        printf("    It saves a complete kernel launch + global mem round-trip for sum.\n");
        printf("    In a real softmax/layernorm pipeline, it replaces TWO kernels.\n\n");

        free(h_in); cudaFree(d_in); cudaFree(d_out);
    }

    // ── Profiling guidance ─────────────────────────────────────────────────
    printf("--- HOW TO PROFILE ---\n\n");
    printf("  .\\compile.bat 05-reduction-patterns\\kernel5.cu profile\n");
    printf("  Open kernel5-analysis.ncu-rep → Summary tab → 3 rows:\n");
    printf("    ID 0: naiveBroadcast\n");
    printf("    ID 1: warpReductionBroadcast\n");
    printf("    ID 2: fuseReduceNormalize\n\n");
    printf("  [Section 3 — Shared Memory  ← KEY DIFFERENCE]\n");
    printf("    Look at 'Shared Load Transactions' or 'Shared Mem Requests':\n");
    printf("      naiveBroadcast:         256 reads/block for the broadcast step\n");
    printf("      warpReductionBroadcast: 8 reads/block for the broadcast step\n");
    printf("      Difference: 248 × 96 blocks = 23,808 fewer shared mem reads\n\n");
    printf("  [Section 4 — Warp State]\n");
    printf("    All: Active Threads/Warp = 32 in shuffle stages (no divergence)\n");
    printf("    Barrier stalls: warpReductionBroadcast has 2 syncs\n");
    printf("                    fuseReduceNormalize has 2 syncs (same)\n\n");
    printf("  [Section 2 — Memory Workload]\n");
    printf("    All: Sectors/Request = 4 (coalesced reads AND writes)\n");
    printf("    DRAM Throughput × 2 (reading input + writing output)\n\n");
    printf("  [__shfl_sync vs __shfl_down_sync in PTX]\n");
    printf("    Generate PTX: nvcc -ptx -arch=sm_75 05-reduction-patterns\\kernel5.cu\n");
    printf("    Search for 'shfl.sync.bfly' (broadcast) vs 'shfl.sync.down' (reduce)\n");
    printf("    Different PTX instructions → different SASS → different hardware paths\n");

    return 0;
}
