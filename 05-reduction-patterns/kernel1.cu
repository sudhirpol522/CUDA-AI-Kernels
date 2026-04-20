// ============================================================================
// REDUCTION PATTERNS — kernel1 vs kernel2 vs kernel3 (profile all together)
//
// Run:  .\compile.bat 05-reduction-patterns\kernel1.cu
// Profile:
//   .\compile.bat 05-reduction-patterns\kernel1.cu divergence
//   .\compile.bat 05-reduction-patterns\kernel1.cu profile
//   Open .ncu-rep → Summary tab → three rows, one per kernel launch
//
// ─────────────────────────────────────────────────────────────────────────────
// KERNEL 1 — naiveReduction
//   Global memory in the loop  +  interleaved (divergent) addressing
//   TWO problems at once — the intentional worst case.
//
//   Problem A — Global memory every stride level:
//     log2(256) = 8 round trips to DRAM per block.
//     Nsight: Stall LG Throttle (load/store unit flooded)
//             DRAM Throughput elevated
//
//   Problem B — Interleaved addressing  tid % (2*stride) == 0:
//     stride=1: threads 0,2,4,... active / 1,3,5,... idle  → every warp diverges
//     stride=2: threads 0,4,8,... active / rest idle
//     Nsight: Active Threads Per Warp well below 32
//             smsp__thread_inst_executed_per_inst_executed < 1.0
//
// ─────────────────────────────────────────────────────────────────────────────
// KERNEL 2 — sequentialReduction
//   Shared memory  +  sequential addressing  (tid < stride)
//   Fixes Problem A completely. Fixes Problem B for large strides.
//
//   Fix A — shared memory:
//     Load once global → shared, then all reduction touches ~5-cycle sdata[].
//     Nsight: LG Throttle drops to near zero → MIO Throttle becomes dominant
//
//   Fix B — sequential addressing:
//     stride=128: warps 0-3 fully active, warps 4-7 fully idle   ← no divergence
//     stride=64:  warps 0-1 fully active                         ← no divergence
//     stride=32:  warp 0  fully active                           ← no divergence
//     stride=16:  threads 0-15 of warp 0 active, 16-31 idle      ← still divergent
//     stride 8,4,2,1: same — last 5 iterations still diverge
//     Nsight: Active Threads Per Warp improves but stays < 32
//
// ─────────────────────────────────────────────────────────────────────────────
// KERNEL 4 — gridStrideReduction
//   Two-phase approach: accumulate into registers first, then block-reduce.
//
//   Phase 1 — grid-stride loop (register accumulation):
//     Each thread steps through the ENTIRE array with a stride equal to the
//     total number of threads launched (blockDim.x * gridDim.x).
//     Thread k processes elements: input[k], input[k+stride], input[k+2*stride], ...
//     The running sum lives in a register — no memory writes during this phase.
//
//   Phase 2 — block reduction (same warp-unrolled pattern as kernel3):
//     Store register sum to sdata[tid], then reduce to sdata[0].
//
//   Why this matters:
//     Previous kernels launch  n/BLOCK_SIZE  blocks — for N=16M that is 65536 blocks.
//     Grid-stride lets you launch a FIXED small grid (e.g. 96 blocks = 24 SMs × 4)
//     regardless of N.  Each thread just does more work in Phase 1.
//
//     Fixed grid benefits:
//       • All 24 SMs are fully loaded for the entire kernel — no tail effect
//       • 96 atomicAdds instead of 65536 — far less contention on output
//       • The global loads in the loop are coalesced (consecutive threads
//         read consecutive addresses every iteration → Sectors/Request = 4)
//
//   What Nsight shows for the grid-stride loop (Phase 1):
//     Long Scoreboard stall: waiting for global loads to return (~600 cycles DRAM)
//     This is EXPECTED — we are bandwidth-bound, reading N floats once.
//     Sectors/Request should be 4 → confirms coalesced access ✓
//     DRAM Throughput: high — this is the memory bandwidth limit of the GPU
//     Fix for this stall = nothing, we have to read all N elements.
//
// ─────────────────────────────────────────────────────────────────────────────
// KERNEL 3 — unrolledReduction
//   Shared memory  +  sequential addressing  +  warp unrolling
//   Fixes ALL divergence. Removes 5 unnecessary __syncthreads() calls.
//
//   Key insight: once stride < 32 (warp size), all remaining active threads
//   are inside warp 0. Threads in the same warp execute in lockstep (SIMT),
//   so __syncthreads() between those iterations is wasted work.
//   We can also drop the  if (tid < stride)  guard for each of those steps.
//
//   The loop runs only for stride = 128 and 64  (both > 32).
//   The last 6 iterations (stride 32,16,8,4,2,1) are hand-unrolled inside
//   an  if (tid < 32)  block — one guard replaces six guarded iterations.
//
//   volatile is required: without it the compiler may cache sdata[tid] in a
//   register. With volatile, every line re-reads from shared memory, so
//   thread 0 always sees the value thread 16 wrote on the previous line.
//
//   Nsight should show:
//     Active Threads Per Warp → 32  (no divergence anywhere)
//     Barrier stall          → reduced (5 fewer __syncthreads calls per block)
//     MIO Throttle           → similar to kernel2 or slightly lower
// ============================================================================

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256

// ============================================================================
// KERNEL 1 — global memory, interleaved (divergent) addressing
//
// Reduction is done IN-PLACE on input[].  The array is destroyed.
// main() re-copies h_in → d_in before the timed run.
//
// Tree walk (blockDim.x = 8 example):
//   stride=1: input[0]+=input[1]  input[2]+=input[3]  ...
//   stride=2: input[0]+=input[2]  input[4]+=input[6]
//   stride=4: input[0]+=input[4]   → block sum in input[blockIdx.x*blockDim.x]
// ============================================================================
__global__ void naiveReduction(float *input, float *output, int n) {
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        if (tid % (2 * stride) == 0) {          // divergent within every warp
            if (idx + stride < n)
                input[idx] += input[idx + stride]; // reads/writes DRAM each pass
        }
        __syncthreads();
    }

    if (tid == 0)
        atomicAdd(output, input[idx]);
}

// ============================================================================
// KERNEL 2 — shared memory, sequential addressing
//
// input[] is read once into sdata[], never written back.
// No need to restore input between warmup and timed run.
//
// Tree walk (blockDim.x = 8 example):
//   stride=4: sdata[0]+=sdata[4]  sdata[1]+=sdata[5]  sdata[2]+=sdata[6]  sdata[3]+=sdata[7]
//   stride=2: sdata[0]+=sdata[2]  sdata[1]+=sdata[3]
//   stride=1: sdata[0]+=sdata[1]   → block sum in sdata[0]
// ============================================================================
__global__ void sequentialReduction(float *input, float *output, int n) {
    __shared__ float sdata[BLOCK_SIZE];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Load once from global → shared (only global traffic this kernel does)
    sdata[tid] = (idx < n) ? input[idx] : 0.0f;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride)                        // whole warps in/out for stride>=32
            sdata[tid] += sdata[tid + stride];   // ~5-cycle shared memory, not DRAM
        __syncthreads();
    }

    if (tid == 0)
        atomicAdd(output, sdata[0]);
}

// ============================================================================
// KERNEL 3 — shared memory, sequential addressing, warp unrolled
//
// Trace through a blockDim.x=8 example to see why omitting guards is safe:
//
//   After loading sdata = [a0 a1 a2 a3 a4 a5 a6 a7]
//   Loop (stride > 2, so only stride=4 runs here):
//     stride=4: sdata = [a0+a4  a1+a5  a2+a6  a3+a7  a4  a5  a6  a7]
//
//   Unroll (tid < 4 guard, then all 4 threads execute every line):
//     v[tid]+=v[tid+2]:
//       tid=0: sdata[0] = (a0+a4)+(a2+a6)  ← correct partial sum
//       tid=1: sdata[1] = (a1+a5)+(a3+a7)  ← correct partial sum
//       tid=2: sdata[2] = (a2+a6)+ a4      ← CORRUPTED — but never read again
//       tid=3: sdata[3] = (a3+a7)+ a5      ← CORRUPTED — but never read again
//     v[tid]+=v[tid+1]:
//       tid=0: sdata[0] = [(a0+a4)+(a2+a6)] + [(a1+a5)+(a3+a7)] = sum all 8 ✓
//       tid=1,2,3: corrupt values, never used
//
//   Result: sdata[0] is always correct. Corrupted slots are never read by
//   the thread that matters (tid=0 at the final step).
// ============================================================================
__global__ void unrolledReduction(float *input, float *output, int n) {
    __shared__ float sdata[BLOCK_SIZE];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (idx < n) ? input[idx] : 0.0f;
    __syncthreads();

    // Loop for strides that span multiple warps — __syncthreads() is needed
    // because threads from different warps don't execute in lockstep.
    for (int stride = blockDim.x / 2; stride > 32; stride >>= 1) {
        if (tid < stride)
            sdata[tid] += sdata[tid + stride];
        __syncthreads();
    }

    // Warp unroll: handles stride = 32, 16, 8, 4, 2, 1
    //
    // All active threads are in warp 0 → lockstep execution → no __syncthreads.
    // One outer guard (tid < 32) replaces six individual if(tid < stride) checks.
    //
    // volatile: tells the compiler "do not cache sdata[tid] in a register."
    // Without it, thread 0's reads of sdata[16], sdata[8], etc. might see
    // stale register-cached values rather than what other threads just wrote.
    if (tid < 32) {
        volatile float *v = sdata;
        v[tid] += v[tid + 32];   // stride=32: warp 0 fully active — no divergence
        v[tid] += v[tid + 16];   // stride=16: threads 0-31 all run this line
        v[tid] += v[tid +  8];   // stride= 8: same warp, still no divergence
        v[tid] += v[tid +  4];
        v[tid] += v[tid +  2];
        v[tid] += v[tid +  1];
    }

    if (tid == 0)
        atomicAdd(output, sdata[0]);
}

// ============================================================================
// KERNEL 4 — grid-stride accumulation + warp-unrolled block reduction
//
// Coalescing in the grid-stride loop (critical — verify in Nsight):
//   Total threads launched = gridDim.x * blockDim.x = FIXED_BLOCKS * 256
//   Iteration 0: thread 0 → input[0],  thread 1 → input[1],  ..., thread 255 → input[255]
//   Iteration 1: thread 0 → input[stride], thread 1 → input[stride+1], ...
//   Every iteration: 256 consecutive threads read 256 consecutive floats = 1 cache line ✓
//   Sectors/Request = 4 in Nsight means this is optimal.
// ============================================================================
#define FIXED_BLOCKS 96   // 24 SMs × 4 — all SMs busy, no tail waste

__global__ void gridStrideReduction(float *input, float *output, int n) {
    __shared__ float sdata[BLOCK_SIZE];

    int tid        = threadIdx.x;
    int idx        = blockIdx.x * blockDim.x + threadIdx.x;
    int gridStride = blockDim.x * gridDim.x;   // total threads in grid

    // ── Phase 1: accumulate into a register (zero global writes here) ─────
    // This loop runs  ceil(n / gridStride)  times per thread.
    // For N=16M, FIXED_BLOCKS=96, BLOCK_SIZE=256:
    //   gridStride = 96*256 = 24576 threads
    //   iterations = 16M / 24576 ≈ 683 loads per thread
    // All 683 loads are perfectly coalesced — Sectors/Request = 4 every time.
    float threadSum = 0.0f;
    for (int i = idx; i < n; i += gridStride)
        threadSum += input[i];

    // ── Phase 2: block-level warp-unrolled reduction (identical to kernel3) ─
    sdata[tid] = threadSum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 32; stride >>= 1) {
        if (tid < stride)
            sdata[tid] += sdata[tid + stride];
        __syncthreads();
    }

    if (tid < 32) {
        volatile float *v = sdata;
        v[tid] += v[tid + 32];
        v[tid] += v[tid + 16];
        v[tid] += v[tid +  8];
        v[tid] += v[tid +  4];
        v[tid] += v[tid +  2];
        v[tid] += v[tid +  1];
    }

    // Only 96 atomicAdds total vs 65536 in kernels 1-3 for N=16M
    if (tid == 0)
        atomicAdd(output, sdata[0]);
}

// ============================================================================
// Helpers
// ============================================================================
float cpuSum(const float *arr, int n) {
    double acc = 0.0;
    for (int i = 0; i < n; i++) acc += arr[i];
    return (float)acc;
}

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error %s:%d — %s\n",                       \
                    __FILE__, __LINE__, cudaGetErrorString(err));               \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

// time one kernel launch, return ms
static float timeKernel(void (*launchFn)(float*, float*, int),
                        float *d_in, float *h_in, size_t B,
                        float *d_out, int n, bool restoreInput) {
    // warmup
    if (restoreInput) CUDA_CHECK(cudaMemcpy(d_in, h_in, B, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_out, 0, sizeof(float)));
    launchFn(d_in, d_out, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    // timed run
    if (restoreInput) CUDA_CHECK(cudaMemcpy(d_in, h_in, B, cudaMemcpyHostToDevice));
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

static void launchNaive(float *d_in, float *d_out, int n) {
    naiveReduction<<<(n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_in, d_out, n);
}
static void launchSeq(float *d_in, float *d_out, int n) {
    sequentialReduction<<<(n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_in, d_out, n);
}
static void launchUnrolled(float *d_in, float *d_out, int n) {
    unrolledReduction<<<(n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_in, d_out, n);
}
static void launchGridStride(float *d_in, float *d_out, int n) {
    // Fixed 96 blocks regardless of N — each thread accumulates ~n/96/BLOCK_SIZE elements
    gridStrideReduction<<<FIXED_BLOCKS, BLOCK_SIZE>>>(d_in, d_out, n);
}

// ============================================================================
// main
// ============================================================================
int main() {
    printf("=== REDUCTION PATTERNS: kernel1 vs kernel2 vs kernel3 vs kernel4 ===\n\n");

    // ─────────────────────────────────────────────────────────────────────
    // Correctness  (N=1024, all 1s → expected = 1024)
    // ─────────────────────────────────────────────────────────────────────
    {
        const int n    = 1024;
        const size_t B = n * sizeof(float);

        float *h_in = (float *)malloc(B);
        for (int i = 0; i < n; i++) h_in[i] = 1.0f;
        float ref = cpuSum(h_in, n);

        float *d_in, *d_out;
        CUDA_CHECK(cudaMalloc(&d_in,  B));
        CUDA_CHECK(cudaMalloc(&d_out, sizeof(float)));

        // kernel 1
        CUDA_CHECK(cudaMemcpy(d_in, h_in, B, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_out, 0, sizeof(float)));
        launchNaive(d_in, d_out, n);
        CUDA_CHECK(cudaDeviceSynchronize());
        float r1 = 0.0f;
        CUDA_CHECK(cudaMemcpy(&r1, d_out, sizeof(float), cudaMemcpyDeviceToHost));
        printf("[Correctness] kernel1  CPU=%.0f  GPU=%.0f  %s\n",
               ref, r1, fabsf(r1 - ref) < 0.5f ? "PASS" : "FAIL");

        // kernel 2
        CUDA_CHECK(cudaMemcpy(d_in, h_in, B, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_out, 0, sizeof(float)));
        launchSeq(d_in, d_out, n);
        CUDA_CHECK(cudaDeviceSynchronize());
        float r2 = 0.0f;
        CUDA_CHECK(cudaMemcpy(&r2, d_out, sizeof(float), cudaMemcpyDeviceToHost));
        printf("[Correctness] kernel2  CPU=%.0f  GPU=%.0f  %s\n",
               ref, r2, fabsf(r2 - ref) < 0.5f ? "PASS" : "FAIL");

        // kernel 3
        CUDA_CHECK(cudaMemcpy(d_in, h_in, B, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_out, 0, sizeof(float)));
        launchUnrolled(d_in, d_out, n);
        CUDA_CHECK(cudaDeviceSynchronize());
        float r3 = 0.0f;
        CUDA_CHECK(cudaMemcpy(&r3, d_out, sizeof(float), cudaMemcpyDeviceToHost));
        printf("[Correctness] kernel3  CPU=%.0f  GPU=%.0f  %s\n",
               ref, r3, fabsf(r3 - ref) < 0.5f ? "PASS" : "FAIL");

        // kernel 4 — use a non-power-of-2 N to stress bounds handling
        const int n4   = 10007;
        const size_t B4 = n4 * sizeof(float);
        float *h_in4 = (float *)malloc(B4);
        for (int i = 0; i < n4; i++) h_in4[i] = 1.0f;
        float ref4 = cpuSum(h_in4, n4);

        float *d_in4, *d_out4;
        CUDA_CHECK(cudaMalloc(&d_in4,  B4));
        CUDA_CHECK(cudaMalloc(&d_out4, sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_in4, h_in4, B4, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_out4, 0, sizeof(float)));
        launchGridStride(d_in4, d_out4, n4);
        CUDA_CHECK(cudaDeviceSynchronize());
        float r4 = 0.0f;
        CUDA_CHECK(cudaMemcpy(&r4, d_out4, sizeof(float), cudaMemcpyDeviceToHost));
        printf("[Correctness] kernel4  CPU=%.0f  GPU=%.0f  %s  (n=%d, non-power-of-2)\n\n",
               ref4, r4, fabsf(r4 - ref4) < 0.5f ? "PASS" : "FAIL", n4);

        free(h_in4); cudaFree(d_in4); cudaFree(d_out4);
        free(h_in);
        cudaFree(d_in);
        cudaFree(d_out);
    }

    // ─────────────────────────────────────────────────────────────────────
    // Benchmark  (N = 16M floats, ~64 MB)
    // ─────────────────────────────────────────────────────────────────────
    {
        const int n    = 1 << 24;
        const size_t B = (size_t)n * sizeof(float);

        float *h_in = (float *)malloc(B);
        for (int i = 0; i < n; i++) h_in[i] = 1.0f;

        float *d_in, *d_out;
        CUDA_CHECK(cudaMalloc(&d_in,  B));
        CUDA_CHECK(cudaMalloc(&d_out, sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_in, h_in, B, cudaMemcpyHostToDevice));

        float ms1 = timeKernel(launchNaive,     d_in, h_in, B, d_out, n, true);
        float ms2 = timeKernel(launchSeq,       d_in, h_in, B, d_out, n, false);
        float ms3 = timeKernel(launchUnrolled,  d_in, h_in, B, d_out, n, false);
        float ms4 = timeKernel(launchGridStride,d_in, h_in, B, d_out, n, false);

        float bw1 = (float)B / (ms1 * 1e6f);
        float bw2 = (float)B / (ms2 * 1e6f);
        float bw3 = (float)B / (ms3 * 1e6f);
        float bw4 = (float)B / (ms4 * 1e6f);

        int blocks1 = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
        printf("[Benchmark]   kernel1 (global + divergent)    Time=%6.3f ms  BW=%5.1f GB/s  (%d blocks)\n",     ms1, bw1, blocks1);
        printf("[Benchmark]   kernel2 (shared + sequential)   Time=%6.3f ms  BW=%5.1f GB/s  %.2fx vs k1\n",     ms2, bw2, ms1/ms2);
        printf("[Benchmark]   kernel3 (shared + warp unroll)  Time=%6.3f ms  BW=%5.1f GB/s  %.2fx vs k1\n",     ms3, bw3, ms1/ms3);
        printf("[Benchmark]   kernel4 (grid-stride, 96 blks)  Time=%6.3f ms  BW=%5.1f GB/s  %.2fx vs k1\n",     ms4, bw4, ms1/ms4);
        printf("              (theoretical peak ~192 GB/s on GTX 1660Ti)\n");
        printf("              kernel3 launched %d blocks; kernel4 launched %d blocks\n",
               blocks1, FIXED_BLOCKS);

        free(h_in);
        cudaFree(d_in);
        cudaFree(d_out);
    }

    // ─────────────────────────────────────────────────────────────────────
    // Nsight guidance
    // ─────────────────────────────────────────────────────────────────────
    printf("\n--- PROFILING GUIDE ---\n");
    printf("Profile:   .\\compile.bat 05-reduction-patterns\\kernel1.cu profile\n");
    printf("Divergence:.\\compile.bat 05-reduction-patterns\\kernel1.cu divergence\n\n");
    printf("In Nsight Summary tab you will see FOUR kernel launches:\n");
    printf("  ID 0 = naiveReduction      (kernel1)\n");
    printf("  ID 1 = sequentialReduction (kernel2)\n");
    printf("  ID 2 = unrolledReduction   (kernel3)\n");
    printf("  ID 3 = gridStrideReduction (kernel4)\n\n");
    printf("What to compare across the three:\n\n");
    printf("  LG Throttle stall:\n");
    printf("    kernel1: HIGH  (every loop iter hits DRAM)\n");
    printf("    kernel2: ~zero (data lives in shared memory)\n");
    printf("    kernel3: ~zero (same fix as kernel2)\n\n");
    printf("  MIO Throttle stall:\n");
    printf("    kernel1: low   (shared memory not used)\n");
    printf("    kernel2: dominant new stall\n");
    printf("    kernel3: same or slightly lower (fewer shared mem ops from removed syncs)\n\n");
    printf("  Active Threads Per Warp:\n");
    printf("    kernel1: well below 32 (divergent every stride from stride=1)\n");
    printf("    kernel2: improved — large strides OK, last 5 still diverge\n");
    printf("    kernel3: should reach 32  (no conditional inside the unrolled block)\n\n");
    printf("  Barrier stall (waiting at __syncthreads):\n");
    printf("    kernel2: 8 syncthreads per block  (one per stride level)\n");
    printf("    kernel3: 3 syncthreads per block  (only for stride 128, 64, load)\n");
    printf("             Barrier stall cycles should drop noticeably\n\n");
    printf("  smsp__thread_inst_executed_per_inst_executed (divergence check):\n");
    printf("    kernel1: << 1.0\n");
    printf("    kernel2: closer to 1.0\n");
    printf("    kernel3: should be 1.0\n");
    printf("    kernel4: should be 1.0  (loop body has no branch — all 256 threads active)\n\n");
    printf("  kernel4-specific — grid-stride loop phase (Phase 1):\n");
    printf("    Dominant stall: Long Scoreboard  (waiting for global loads)\n");
    printf("    This is EXPECTED — bandwidth-limited, not a bug to fix\n");
    printf("    Sectors/Request: should be 4  (coalesced — consecutive threads,\n");
    printf("                                   consecutive addresses every iteration)\n");
    printf("    DRAM Throughput: high — we are near the memory bandwidth ceiling\n");
    printf("    Compare k3 vs k4 DRAM Throughput: k4 should be same or higher\n");
    printf("    because fewer blocks = less scheduler overhead = SM stays busier\n");

    return 0;
}
