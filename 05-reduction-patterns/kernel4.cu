// ============================================================================
// REDUCTION PATTERNS — kernel4.cu
// Topic: Template-Based Warp Reduction (compile-time BLOCK_SIZE)
//
// TWO kernels compared:
//   Kernel 1: nonTemplateReduction   — same algorithm, runtime blockDim checks
//   Kernel 2: templateWarpReduction  — compile-time BLOCK_SIZE → dead code gone
//
// Plus: one kernel instantiated at BLOCK_SIZE=128 and 64 to show flexibility.
//
// Compile:   .\compile.bat 05-reduction-patterns\kernel4.cu
// Profile:   .\compile.bat 05-reduction-patterns\kernel4.cu profile
//            Open kernel4-analysis.ncu-rep → Summary tab → 2 rows.
//
// ─────────────────────────────────────────────────────────────────────────────
// WHAT COMPILE-TIME BLOCK_SIZE BUYS YOU
//
//   Problem with runtime checks:
//     if (blockDim.x >= 256) { ... }
//     → Even if blockDim.x IS always 256, this is a RUNTIME branch.
//     → The PTX/SASS contains a compare + conditional jump instruction.
//     → The CPU compiler can eliminate this via constant propagation, but
//       nvcc does not always do so reliably for device code.
//
//   With a template:
//     if (BLOCK_SIZE >= 256) { ... }          // BLOCK_SIZE is a compile-time int
//     → The compiler sees a constant boolean TRUE or FALSE at compile time.
//     → The entire if-block is either ALWAYS compiled in or ALWAYS removed.
//     → No branch instruction in the generated PTX at all.
//     → The removed block's __syncthreads() is also removed — zero overhead.
//
//   Example: templateWarpReduction<128><<<grid, 128>>>(...)
//     if (BLOCK_SIZE >= 256) { ... }   → FALSE → entire block deleted ✓
//     if (BLOCK_SIZE >= 128) { ... }   → TRUE  → block kept ✓
//     if (BLOCK_SIZE >=  64) { ... }   → TRUE  → block kept ✓
//     Generated SASS: no code for the 256-stride step at all.
//
//   Same kernel handles BLOCK_SIZE = 32, 64, 128, 256, 512, 1024 — each
//   instantiation is as efficient as a hand-written kernel for that size.
//
// ─────────────────────────────────────────────────────────────────────────────
// ABOUT __syncthreads() IN DEAD TEMPLATE BRANCHES
//
//   Rule: __syncthreads() must be reached by ALL threads in a block,
//         or undefined behavior results.
//
//   Safe:    if (BLOCK_SIZE >= 256) { __syncthreads(); }
//     → BLOCK_SIZE is the same for every thread (it's a compile-time constant).
//     → Either ALL threads enter this branch, or NONE do.
//     → No undefined behavior.
//
//   UNSAFE:  if (blockDim.x == 256 && tid < 128) { __syncthreads(); }
//     → Only some threads enter → UB.
//     (This bug is unrelated to templates — just don't gate syncs on tid.)
//
// ─────────────────────────────────────────────────────────────────────────────
// ABOUT THE MISSING __syncthreads() BEFORE THE WARP SHUFFLE
//
//   After the last shared-memory reduction step:
//     if (BLOCK_SIZE >= 64) {
//         if (tid < 32) sharedData[tid] += sharedData[tid + 32];
//     }
//     // ← No __syncthreads() here. Is this safe?
//
//   YES. Here is why:
//     - Only threads 0-31 write to sharedData[0..31] in that step.
//     - Only threads 0-31 read from sharedData[0..31] in the shuffle.
//     - ALL of threads 0-31 are in WARP 0, which executes in lockstep (SIMT).
//     - Within a single warp, a write by thread K to shared memory is
//       immediately visible to thread K and ALL other threads in the SAME warp
//       on the very next instruction — NO barrier needed.
//     - Threads 32-255 don't participate in either step.
//     If this code had threads from DIFFERENT warps writing and reading,
//     a sync would be mandatory. That's why strides >= 64 still have syncs.
//
// ─────────────────────────────────────────────────────────────────────────────
// EXPLICIT UNROLLING vs #pragma unroll
//
//   Template version (explicit):
//     val += __shfl_down_sync(0xffffffff, val, 16);
//     val += __shfl_down_sync(0xffffffff, val,  8);
//     ...5 lines total...
//     Compiler sees 5 independent shuffle instructions — can schedule freely.
//
//   Loop version (with #pragma unroll):
//     #pragma unroll
//     for (int offset = 16; offset > 0; offset >>= 1)
//         val += __shfl_down_sync(0xffffffff, val, offset);
//     Same result IF the compiler unrolls successfully. Not always guaranteed.
//
//   From the profiling guide:
//     "#pragma unroll fixes MIO Throttle" ← MISCONCEPTION
//     Before unroll: MIO = 24.70 cycles. After: 24.78 cycles. Unchanged.
//     Unrolling removes loop bookkeeping (branch + counter update), not MIO ops.
//
// ─────────────────────────────────────────────────────────────────────────────
// WHAT NSIGHT WILL SHOW:
//
//   [Divergence check]
//     Both kernels: Active Threads Per Warp ≈ 32 in all shuffle stages
//
//   [Source tab]
//     nonTemplateReduction: shows branch instructions for blockDim.x checks
//     templateWarpReduction<256>: those lines produce NO assembly (dead code)
//     Compare PTX with:  nvcc -ptx -arch=sm_75 05-reduction-patterns\kernel4.cu
//     Look for "setp" (set predicate = compare) instructions — template has fewer.
//
//   [Shared Memory]
//     Both: Bank Conflicts = 0 (sequential addressing)
//     Transactions roughly equal (same algorithm, same block size)
//
// ============================================================================

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define DEFAULT_BLOCK  256   // the block size we benchmark at
#define FIXED_BLOCKS    96   // 24 SMs × 4 — saturates all SMs, no tail waste

// ============================================================================
// KERNEL 1: nonTemplateReduction  (baseline — runtime blockDim.x checks)
//
// This uses RUNTIME comparisons: "if (blockDim.x >= 256)".
// At compile time, nvcc does NOT know whether blockDim.x is 256 or not.
// Each branch produces a real PTX setp + bra (compare + conditional jump).
//
// For the SAME launch config (blockDim.x = 256 always), the branches are
// always taken, but the hardware still pays for the compare instruction.
// ============================================================================
__global__ void nonTemplateReduction(float *input, float *output, int n) {
    __shared__ float sharedData[DEFAULT_BLOCK];

    int tid        = threadIdx.x;
    int idx        = blockIdx.x * blockDim.x + tid;
    int gridStride = blockDim.x * gridDim.x;

    float sum = 0.0f;
    for (int i = idx; i < n; i += gridStride)
        sum += input[i];

    sharedData[tid] = sum;
    __syncthreads();

    // Runtime branches — compare against blockDim.x at runtime
    if (blockDim.x >= 256) {
        if (tid < 128) sharedData[tid] += sharedData[tid + 128];
        __syncthreads();
    }
    if (blockDim.x >= 128) {
        if (tid < 64) sharedData[tid] += sharedData[tid + 64];
        __syncthreads();
    }
    if (blockDim.x >= 64) {
        if (tid < 32) sharedData[tid] += sharedData[tid + 32];
        // No __syncthreads() — threads 0-31 are warp 0, write+read in lockstep
    }

    if (tid < 32) {
        // Load from shared mem into register — last shared mem access in this kernel
        float val = sharedData[tid];
        // Explicit shuffle steps (no loop, no loop counter overhead)
        val += __shfl_down_sync(0xffffffff, val, 16);
        val += __shfl_down_sync(0xffffffff, val,  8);
        val += __shfl_down_sync(0xffffffff, val,  4);
        val += __shfl_down_sync(0xffffffff, val,  2);
        val += __shfl_down_sync(0xffffffff, val,  1);
        // Thread 0 has the block total — write directly from register, no shared mem
        if (tid == 0)
            atomicAdd(output, val);
    }
}

// ============================================================================
// KERNEL 2: templateWarpReduction<BLOCK_SIZE>  (compile-time BLOCK_SIZE)
//
// TEMPLATE PARAMETER: BLOCK_SIZE must match the actual blockDim.x at launch.
//   templateWarpReduction<256><<<grid, 256>>>(...)  ← correct
//   templateWarpReduction<128><<<grid, 128>>>(...)  ← correct (different instantiation)
//   templateWarpReduction<256><<<grid, 128>>>(...)  ← BUG: mismatch!
//
// Each different BLOCK_SIZE instantiation produces DIFFERENT compiled code.
// The compiler generates the optimal SASS for each case independently.
//
// BLOCK_SIZE must be a power of 2 between 32 and 1024.
// ============================================================================
template <int BLOCK_SIZE>
__global__ void templateWarpReduction(float *input, float *output, int n) {

    // Shared memory size is a compile-time constant.
    // The compiler knows exactly how much to allocate — no dynamic sizing.
    __shared__ float sharedData[BLOCK_SIZE];

    int tid        = threadIdx.x;
    int idx        = blockIdx.x * blockDim.x + tid;
    int gridStride = blockDim.x * gridDim.x;

    // ── Grid-stride accumulation (same as all previous kernels) ───────────
    float sum = 0.0f;
    for (int i = idx; i < n; i += gridStride)
        sum += input[i];

    sharedData[tid] = sum;
    __syncthreads();

    // ── Compile-time unrolled tree reduction: BLOCK_SIZE → 32 ─────────────
    //
    // Each if-block is either FULLY INCLUDED or FULLY REMOVED at compile time.
    //
    // For templateWarpReduction<256>:
    //   BLOCK_SIZE >= 512 → FALSE → block deleted (no code generated)
    //   BLOCK_SIZE >= 256 → TRUE  → 128-stride step included ✓
    //   BLOCK_SIZE >= 128 → TRUE  → 64-stride step included  ✓
    //   BLOCK_SIZE >=  64 → TRUE  → 32-stride step included  ✓
    //
    // For templateWarpReduction<128>:
    //   BLOCK_SIZE >= 512 → FALSE → deleted
    //   BLOCK_SIZE >= 256 → FALSE → deleted (128-stride step gone entirely!)
    //   BLOCK_SIZE >= 128 → TRUE  → 64-stride step included  ✓
    //   BLOCK_SIZE >=  64 → TRUE  → 32-stride step included  ✓
    //
    // For templateWarpReduction<64>:
    //   Only the 32-stride step runs. 3 blocks deleted.

    if (BLOCK_SIZE >= 512) {
        if (tid < 256) sharedData[tid] += sharedData[tid + 256];
        __syncthreads();
    }
    if (BLOCK_SIZE >= 256) {
        if (tid < 128) sharedData[tid] += sharedData[tid + 128];
        __syncthreads();
    }
    if (BLOCK_SIZE >= 128) {
        if (tid < 64) sharedData[tid] += sharedData[tid + 64];
        __syncthreads();
    }
    if (BLOCK_SIZE >= 64) {
        // This is the last multi-warp step.
        // No __syncthreads() after because the next step is warp 0 only:
        //   writers (tid < 32) and readers (tid < 32) are the SAME warp.
        //   Warp 0 executes in lockstep — write is visible on next instruction.
        if (tid < 32) sharedData[tid] += sharedData[tid + 32];
    }

    // ── Warp-level reduction: 32 values → 1, entirely in registers ─────────
    //
    // Only warp 0 (threads 0-31) participates.
    // Load the 32 values from shared memory into registers — last shared mem touch.
    // Then 5 explicitly unrolled shuffle steps: no loop, no loop counter,
    // no branch instructions, compiler can schedule these instructions freely.
    //
    // After the 5 steps: val in thread 0 = sum of all 32 values.
    //   threads 1-31: partially correct or garbage — never used.
    //
    // CORRECTION from the template's original sharedData[0] = val approach:
    //   The template wrote sharedData[0] = val INSIDE if (tid < 32) without
    //   a guard → all 32 threads write to sharedData[0] → race condition.
    //   Fix: use the register val directly in thread 0. No shared mem write needed.
    if (tid < 32) {
        float val = sharedData[tid];
        //
        // Round 1: offset=16
        //   thread 0 += thread 16's val  → val[0] = sdata[0] + sdata[16]
        //   thread 1 += thread 17's val  → val[1] = sdata[1] + sdata[17]
        //   ...
        //   thread 15 += thread 31's val → val[15] = sdata[15] + sdata[31]
        //   thread 16: source 32 >= 32   → keeps own val[16] (unused)
        val += __shfl_down_sync(0xffffffff, val, 16);
        //
        // Round 2: offset=8
        //   thread 0 += thread 8's val   → val[0] = (sdata[0]+sdata[16]) + (sdata[8]+sdata[24])
        //   thread 8's val is (sdata[8]+sdata[24]) — correct because thread 8 DID participate
        //   in round 1 and its source (thread 24) was in range.
        val += __shfl_down_sync(0xffffffff, val,  8);
        //
        // Round 3: offset=4
        //   thread 0 accumulates 8 original values
        val += __shfl_down_sync(0xffffffff, val,  4);
        //
        // Round 4: offset=2  → thread 0 accumulates 16 values
        val += __shfl_down_sync(0xffffffff, val,  2);
        //
        // Round 5: offset=1  → thread 0 has sum of all 32 values ✓
        val += __shfl_down_sync(0xffffffff, val,  1);

        // Thread 0 writes block total directly from register — no shared mem write,
        // no extra sync needed. This is the cleanest possible final write.
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
// Timing helper
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

static void launchNonTemplate(float *d_in, float *d_out, int n) {
    nonTemplateReduction<<<FIXED_BLOCKS, DEFAULT_BLOCK>>>(d_in, d_out, n);
}
static void launchTemplate256(float *d_in, float *d_out, int n) {
    templateWarpReduction<256><<<FIXED_BLOCKS, 256>>>(d_in, d_out, n);
}
static void launchTemplate128(float *d_in, float *d_out, int n) {
    // 128 threads/block → need 2× as many blocks to cover same data
    templateWarpReduction<128><<<FIXED_BLOCKS * 2, 128>>>(d_in, d_out, n);
}
static void launchTemplate64(float *d_in, float *d_out, int n) {
    // 64 threads/block → need 4× as many blocks
    templateWarpReduction<64><<<FIXED_BLOCKS * 4, 64>>>(d_in, d_out, n);
}

// ============================================================================
// main
// ============================================================================
int main() {
    printf("=== TEMPLATE WARP REDUCTION (kernel4.cu) ===\n\n");
    printf("Compile-time BLOCK_SIZE eliminates dead code branches.\n");
    printf("Each template instantiation generates optimal SASS for that block size.\n\n");
    printf("Instantiations in this benchmark:\n");
    printf("  nonTemplateReduction        : blockDim.x=256, runtime checks\n");
    printf("  templateWarpReduction<256>  : blockDim.x=256, compile-time eliminated\n");
    printf("  templateWarpReduction<128>  : blockDim.x=128, 128-stride step removed\n");
    printf("  templateWarpReduction<64>   : blockDim.x=64,  two top steps removed\n\n");

    // ── Correctness: N=1024 ────────────────────────────────────────────────
    {
        const int    n = 1024;
        const size_t B = n * sizeof(float);
        float *h_in    = (float *)malloc(B);
        for (int i = 0; i < n; i++) h_in[i] = 1.0f;
        float ref = cpuSum(h_in, n);

        float *d_in, *d_out;
        CUDA_CHECK(cudaMalloc(&d_in,  B));
        CUDA_CHECK(cudaMalloc(&d_out, sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_in, h_in, B, cudaMemcpyHostToDevice));

        float r1 = 0.0f, r2 = 0.0f, r3 = 0.0f, r4 = 0.0f;

        CUDA_CHECK(cudaMemset(d_out, 0, sizeof(float)));
        nonTemplateReduction<<<FIXED_BLOCKS, DEFAULT_BLOCK>>>(d_in, d_out, n);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(&r1, d_out, sizeof(float), cudaMemcpyDeviceToHost));

        CUDA_CHECK(cudaMemset(d_out, 0, sizeof(float)));
        templateWarpReduction<256><<<FIXED_BLOCKS, 256>>>(d_in, d_out, n);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(&r2, d_out, sizeof(float), cudaMemcpyDeviceToHost));

        CUDA_CHECK(cudaMemset(d_out, 0, sizeof(float)));
        templateWarpReduction<128><<<FIXED_BLOCKS * 2, 128>>>(d_in, d_out, n);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(&r3, d_out, sizeof(float), cudaMemcpyDeviceToHost));

        CUDA_CHECK(cudaMemset(d_out, 0, sizeof(float)));
        templateWarpReduction<64><<<FIXED_BLOCKS * 4, 64>>>(d_in, d_out, n);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(&r4, d_out, sizeof(float), cudaMemcpyDeviceToHost));

        printf("[Correctness N=%-6d]  ref=%.0f\n", n, ref);
        printf("  nonTemplateReduction       : %.0f  %s\n", r1, fabsf(r1-ref)<0.5f?"PASS":"FAIL");
        printf("  templateWarpReduction<256> : %.0f  %s\n", r2, fabsf(r2-ref)<0.5f?"PASS":"FAIL");
        printf("  templateWarpReduction<128> : %.0f  %s\n", r3, fabsf(r3-ref)<0.5f?"PASS":"FAIL");
        printf("  templateWarpReduction<64>  : %.0f  %s\n", r4, fabsf(r4-ref)<0.5f?"PASS":"FAIL");

        free(h_in); cudaFree(d_in); cudaFree(d_out);
    }

    // ── Correctness: N=999983, non-power-of-2 ─────────────────────────────
    {
        const int    n = 999983;
        const size_t B = n * sizeof(float);
        float *h_in    = (float *)malloc(B);
        for (int i = 0; i < n; i++) h_in[i] = 1.0f;
        float ref = cpuSum(h_in, n);

        float *d_in, *d_out;
        CUDA_CHECK(cudaMalloc(&d_in,  B));
        CUDA_CHECK(cudaMalloc(&d_out, sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_in, h_in, B, cudaMemcpyHostToDevice));

        float r1 = 0.0f, r2 = 0.0f, r3 = 0.0f, r4 = 0.0f;

        CUDA_CHECK(cudaMemset(d_out, 0, sizeof(float)));
        nonTemplateReduction<<<FIXED_BLOCKS, DEFAULT_BLOCK>>>(d_in, d_out, n);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(&r1, d_out, sizeof(float), cudaMemcpyDeviceToHost));

        CUDA_CHECK(cudaMemset(d_out, 0, sizeof(float)));
        templateWarpReduction<256><<<FIXED_BLOCKS, 256>>>(d_in, d_out, n);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(&r2, d_out, sizeof(float), cudaMemcpyDeviceToHost));

        CUDA_CHECK(cudaMemset(d_out, 0, sizeof(float)));
        templateWarpReduction<128><<<FIXED_BLOCKS * 2, 128>>>(d_in, d_out, n);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(&r3, d_out, sizeof(float), cudaMemcpyDeviceToHost));

        CUDA_CHECK(cudaMemset(d_out, 0, sizeof(float)));
        templateWarpReduction<64><<<FIXED_BLOCKS * 4, 64>>>(d_in, d_out, n);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(&r4, d_out, sizeof(float), cudaMemcpyDeviceToHost));

        printf("\n[Correctness N=%-6d]  ref=%.0f  (non-power-of-2)\n", n, ref);
        printf("  nonTemplateReduction       : %.0f  %s\n", r1, fabsf(r1-ref)<0.5f?"PASS":"FAIL");
        printf("  templateWarpReduction<256> : %.0f  %s\n", r2, fabsf(r2-ref)<0.5f?"PASS":"FAIL");
        printf("  templateWarpReduction<128> : %.0f  %s\n", r3, fabsf(r3-ref)<0.5f?"PASS":"FAIL");
        printf("  templateWarpReduction<64>  : %.0f  %s\n\n", r4, fabsf(r4-ref)<0.5f?"PASS":"FAIL");

        free(h_in); cudaFree(d_in); cudaFree(d_out);
    }

    // ── Benchmark: N=16M ──────────────────────────────────────────────────
    {
        const int    n  = 1 << 24;
        const size_t B  = (size_t)n * sizeof(float);
        float *h_in     = (float *)malloc(B);
        for (int i = 0; i < n; i++) h_in[i] = 1.0f;

        float *d_in, *d_out;
        CUDA_CHECK(cudaMalloc(&d_in,  B));
        CUDA_CHECK(cudaMalloc(&d_out, sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_in, h_in, B, cudaMemcpyHostToDevice));

        float ms1 = timeKernel(launchNonTemplate,  d_in, d_out, n);
        float ms2 = timeKernel(launchTemplate256,  d_in, d_out, n);
        float ms3 = timeKernel(launchTemplate128,  d_in, d_out, n);
        float ms4 = timeKernel(launchTemplate64,   d_in, d_out, n);

        float bw1 = (float)B / (ms1 * 1e6f);
        float bw2 = (float)B / (ms2 * 1e6f);
        float bw3 = (float)B / (ms3 * 1e6f);
        float bw4 = (float)B / (ms4 * 1e6f);

        printf("[Benchmark N=%d (~64 MB)]\n", n);
        printf("  nonTemplateReduction       (blk=256): %6.3f ms  BW=%5.1f GB/s\n", ms1, bw1);
        printf("  templateWarpReduction<256> (blk=256): %6.3f ms  BW=%5.1f GB/s  (%.2fx)\n", ms2, bw2, ms1/ms2);
        printf("  templateWarpReduction<128> (blk=128): %6.3f ms  BW=%5.1f GB/s  (%.2fx)\n", ms3, bw3, ms1/ms3);
        printf("  templateWarpReduction<64>  (blk= 64): %6.3f ms  BW=%5.1f GB/s  (%.2fx)\n", ms4, bw4, ms1/ms4);
        printf("  (theoretical peak ~192 GB/s on GTX 1660Ti)\n\n");

        printf("  Why <256> and nonTemplate time nearly the same:\n");
        printf("    Both dominated by Phase 1 (64 MB DRAM read).\n");
        printf("    Template's advantage (fewer branch instructions) is microscopic\n");
        printf("    vs the hundreds of millions of global loads.\n\n");
        printf("  Why <128> and <64> may differ:\n");
        printf("    Smaller blocks → more blocks launched → better tail coverage.\n");
        printf("    But also: each block does fewer threads → more iterations per thread.\n");
        printf("    The occupancy and scheduling interplay determines the winner.\n\n");
        printf("  The template's REAL value:\n");
        printf("    ONE source file handles block sizes 32, 64, 128, 256, 512, 1024.\n");
        printf("    Each instantiation is optimal — no runtime overhead for other sizes.\n");
        printf("    Libraries like CUB/Thrust use this pattern extensively.\n\n");

        free(h_in); cudaFree(d_in); cudaFree(d_out);
    }

    // ── PTX inspection guidance ────────────────────────────────────────────
    printf("--- HOW TO SEE THE DEAD CODE ELIMINATION ---\n\n");
    printf("  Generate PTX (human-readable assembly):\n");
    printf("    nvcc -ptx -arch=sm_75 05-reduction-patterns\\kernel4.cu\n");
    printf("    (opens kernel4.ptx in the same folder)\n\n");
    printf("  Search for 'templateWarpReduction' in the PTX file.\n");
    printf("  You will see TWO separate function bodies:\n");
    printf("    _Z20templateWarpReductionILi256E... (the <256> instantiation)\n");
    printf("    _Z20templateWarpReductionILi128E... (the <128> instantiation)\n");
    printf("    _Z20templateWarpReductionILi64E...  (the <64>  instantiation)\n\n");
    printf("  The <128> body is SHORTER than <256>:\n");
    printf("    Missing: the setp/bra instructions for the 128-stride step.\n");
    printf("    Missing: the __syncthreads() call that went with it.\n\n");
    printf("  In Nsight Compute Source tab:\n");
    printf("    Profile with: .\\compile.bat 05-reduction-patterns\\kernel4.cu profile\n");
    printf("    Select each kernel in the dropdown.\n");
    printf("    For templateWarpReduction<128>: the BLOCK_SIZE>=256 lines show no SASS.\n");
    printf("    For nonTemplateReduction: every if-blockDim.x line has branch SASS.\n\n");
    printf("  [Warp State]\n");
    printf("    All template kernels: Active Threads/Warp = 32 in shuffle stages.\n");
    printf("    Barrier stalls: 2-3 syncs for <256>, 1-2 for <128>, 1 for <64>.\n");

    return 0;
}
