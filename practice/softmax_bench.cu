// Softmax Benchmark — GTX 1660 Ti, Peak BW: 288 GB/s
// Matrix: 4096x4096 floats (67.1 MB), averaged over 200 runs
//
//   kernel                 passes   time(ms)   BW(GB/s)   % of peak   speedup
//   ─────────────────────  ──────   ────────   ────────   ─────────   ───────
//   naive (3-pass)           4       17.21      15.6        5.4%        1.0x
//   online (2-pass)          3       14.19      14.2        4.9%        1.2x
//   coalesced warp (2-pass)  3        1.51     133.5       46.3%       11.4x
//
// KEY INSIGHT: naive vs online is only 1.2x — barely matters.
//              naive vs coalesced is 11.4x — that's the real fix.
//
// Why naive and online are both stuck at ~5%:
//   One thread per row. A warp of 32 threads reads 32 different rows,
//   addresses N floats apart → 32 cache-line transactions per step (non-coalesced).
//   expf() adds compute cost (~20 cycles each) on top of that.
//   Reducing passes from 3→2 saves time proportionally (17.21/14.19 ≈ 4/3),
//   but the bandwidth % stays the same because both reads AND expf calls scale.
//
// Why coalesced warp jumps to 46%:
//   32 threads per row, each reading adjacent columns → 1 cache line per step.
//   Same coalescing fix that took GEMV from 6% → 43%.
//   Now expf() is the main bottleneck (not memory), which is why we stop at 46%
//   instead of the ~73–92% the pure-memory GEMV kernels achieved.

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

#ifndef CEIL_DIV
#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))
#endif

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// negative infinity as a float bit pattern — avoids the INFINITY double-to-float warning
#define NEG_INF_F (-__int_as_float(0x7f800000))

// =============================================================================
// kernel 0: naive 3-pass — 17.18 ms, 15.6 GB/s (5.4% of peak)
// =============================================================================
// One thread per row. Three separate loops over N columns:
//   pass 1: find max              → N reads
//   pass 2: sum of exp(x - max)  → N reads + N expf calls
//   pass 3: write exp(x - max)/L → N reads + N expf calls + N writes
// Total: 4*N floats touched per row.
//
// Non-coalesced: adjacent threads read adjacent rows (stride N apart),
// so each warp issues 32 cache-line transactions per column step.
// Combined with the cost of expf, this kernel is both memory- and compute-limited.
// =============================================================================
__global__ void softmax_kernel_0(float* __restrict__ matd, float* __restrict__ resd, int M, int N) {
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    if (row < M) {
        float m = NEG_INF_F;
        float L = 0.0f;
        for (int col = 0; col < N; col++) { int i = row*N+col; m = max(m, matd[i]); }
        for (int col = 0; col < N; col++) { int i = row*N+col; L += expf(matd[i] - m); }
        for (int col = 0; col < N; col++) { int i = row*N+col; resd[i] = expf(matd[i] - m) / L; }
    }
}
void run_kernel_0(float* matd, float* resd, int M, int N) {
    softmax_kernel_0<<<CEIL_DIV(M, 1024), 1024>>>(matd, resd, M, N);
}

// =============================================================================
// kernel 1: online 2-pass — 13.26 ms, 15.2 GB/s (5.3% of peak)
// =============================================================================
// Same one-thread-per-row structure, but merges pass 1 and pass 2 into one:
//   pass 1: track running max m and running sum L simultaneously
//           when a new max is found: L = L * exp(m_old - m_new)  (rescale)
//   pass 2: write exp(x - m) / L  → N reads + N expf calls + N writes
// Total: 3*N floats touched per row (vs 4*N for naive).
//
// Speedup over naive: 17.18 / 13.26 = 1.30x ≈ 4/3 pass ratio.
// The improvement is purely from removing one full pass — the memory access
// pattern and expf cost are otherwise identical.
//
// Numerical accuracy: max_err ~3e-9 vs naive ~3e-10.
// The online version rescales L incrementally which accumulates a small
// extra floating-point error, but both are well within float precision.
// =============================================================================
__global__ void softmax_kernel_1(float* __restrict__ matd, float* __restrict__ resd, int M, int N) {
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    if (row < M) {
        float m = NEG_INF_F;
        float L = 0.0f;
        for (int col = 0; col < N; col++) {
            float curr = matd[row*N+col];
            if (curr > m) { L = L * expf(m - curr); m = curr; }
            L += expf(curr - m);
        }
        for (int col = 0; col < N; col++)
            resd[row*N+col] = expf(matd[row*N+col] - m) / L;
    }
}
void run_kernel_1(float* matd, float* resd, int M, int N) {
    softmax_kernel_1<<<CEIL_DIV(M, 1024), 1024>>>(matd, resd, M, N);
}

// =============================================================================
// kernel 2: coalesced warp — one warp (32 threads) per row
// =============================================================================
// Same online algorithm, but now 32 threads share the work for one row.
//
// Memory access — coalesced:
//   tid 0  reads col 0, 32, 64, ...
//   tid 1  reads col 1, 33, 65, ...    ← 32 consecutive addresses per step
//   ...                                    = 1 cache line = 1 transaction
//
// After the column loop each thread holds a partial (m_local, L_local).
// We then need a warp-wide reduce to get the true global max and sum.
//
// Reduction order matters: find global max first, then rescale each thread's
// partial L, then sum all the rescaled L values.
// =============================================================================
namespace {
__device__ __forceinline__ float warpReduceMax(float val) {
    for (int offset = 16; offset > 0; offset /= 2)
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, offset));
    return val;
}
__device__ __forceinline__ float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_xor_sync(0xffffffff, val, offset);
    return val;
}
// shuffle-based block reductions (used by vectorized kernel)
template<typename T>
__device__ __forceinline__ void blockReduceMax(T val, T* smem, T identity) {
    int tid = threadIdx.x;
    val = warpReduceMax(val);
    if (tid % 32 == 0) smem[tid / 32] = val;
    __syncthreads();
    if (tid < CEIL_DIV((int)blockDim.x, 32)) val = smem[tid]; else val = identity;
    if (tid / 32 == 0) val = warpReduceMax(val);
    if (tid == 0) smem[0] = val;
    __syncthreads();
}
template<typename T>
__device__ __forceinline__ void blockReduceSum(T val, T* smem) {
    int tid = threadIdx.x;
    val = warpReduceSum(val);
    if (tid % 32 == 0) smem[tid / 32] = val;
    __syncthreads();
    if (tid < CEIL_DIV((int)blockDim.x, 32)) val = smem[tid]; else val = T(0);
    if (tid / 32 == 0) val = warpReduceSum(val);
    if (tid == 0) smem[0] = val;
    __syncthreads();
}
}

__global__ void softmax_kernel_2(float* __restrict__ matd, float* __restrict__ resd, int M, int N) {
    int row = blockIdx.x;
    int tid = threadIdx.x;   // 0..31
    if (row >= M) return;

    // pass 1: each thread runs online softmax on its own columns (stride 32)
    float m_local = -__int_as_float(0x7f800000);
    float L_local = 0.0f;
    for (int col = tid; col < N; col += 32) {
        float curr = matd[row * N + col];
        if (curr > m_local) { L_local *= expf(m_local - curr); m_local = curr; }
        L_local += expf(curr - m_local);
    }

    // warp reduce: find the global max across all 32 lanes
    float m_global = warpReduceMax(m_local);

    // rescale each lane's partial sum to the global max, then sum
    L_local *= expf(m_local - m_global);
    float L_global = warpReduceSum(L_local);

    // pass 2: normalize — all 32 threads write their columns
    for (int col = tid; col < N; col += 32)
        resd[row * N + col] = expf(matd[row * N + col] - m_global) / L_global;
}
void run_kernel_2(float* matd, float* resd, int M, int N) {
    softmax_kernel_2<<<M, 32>>>(matd, resd, M, N);
}

// =============================================================================
// kernel 3: shared memory block reduction — 1024 threads per row
// =============================================================================
// Same online algorithm within each thread, but now 1024 threads (32 warps)
// share one row. Block-level max and sum are found via a classic smem tree
// reduction instead of warp shuffles.
//
// Pass 1: each thread runs online softmax on its slice of columns (stride 1024).
//         Result: every thread holds (local_max, local_norm).
//
// smem tree reduce — max:
//   All 1024 threads write local_max to smem[tid], then halve active threads
//   each step: stride=512 → 256 → ... → 1. O(log N) steps, each needs __syncthreads.
//
// smem tree reduce — sum:
//   Each thread rescales its local_norm by exp(local_max - row_max) first,
//   then the same halving tree sums them. row_norm = smem[0].
//
// Pass 2: each thread normalizes its columns.
// =============================================================================
__global__ void softmax_kernel_3(float* __restrict__ xd, float* __restrict__ resd, int M, int N) {
    __shared__ float smem[1024];

    int row = blockIdx.x;
    int tid = threadIdx.x;
    if (row >= M) return;

    float* input_row  = xd   + row * N;
    float* output_row = resd + row * N;

    // pass 1: online softmax over this thread's columns
    float local_max  = -__int_as_float(0x7f800000);
    float local_norm = 0.0f;
    for (int i = tid; i < N; i += blockDim.x) {
        float x = input_row[i];
        if (x > local_max) { local_norm *= expf(local_max - x); local_max = x; }
        local_norm += expf(x - local_max);
    }

    // smem tree: find row max
    smem[tid] = local_max;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) smem[tid] = fmaxf(smem[tid], smem[tid + stride]);
        __syncthreads();
    }
    float row_max = smem[0];
    __syncthreads();

    // smem tree: find row sum (rescale each thread's local_norm first)
    smem[tid] = local_norm * expf(local_max - row_max);
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) smem[tid] += smem[tid + stride];
        __syncthreads();
    }
    float row_norm = smem[0];
    __syncthreads();

    // pass 2: normalize
    for (int i = tid; i < N; i += blockDim.x)
        output_row[i] = expf(input_row[i] - row_max) / row_norm;
}
void run_kernel_3(float* matd, float* resd, int M, int N) {
    softmax_kernel_3<<<M, 1024>>>(matd, resd, M, N);
}

// =============================================================================
// kernel 4: vectorized float4 + shuffle block reduction — 1024 threads per row
// =============================================================================
// Combines two optimizations on top of kernel 3:
//
// 1. float4 loads (16 bytes per instruction instead of 4):
//    Each thread loads 4 floats at once, reducing load instruction count 4x.
//    Keeps the memory bus more saturated per instruction issued.
//
// 2. Warp-shuffle blockReduceMax/Sum instead of smem tree:
//    Tree reduction: log2(1024) = 10 rounds of smem read/write + __syncthreads
//    Shuffle reduction: 5 rounds of register shuffles (no smem, no __syncthreads
//    inside the warp) + 1 smem round for the 32 warp sums
//    → fewer __syncthreads calls → less stall time
//
// Tail handling: if N is not divisible by 4, the last (N%4) elements are
// processed as scalars by the first (N%4) threads.
// =============================================================================
__global__ void softmax_kernel_4(float* __restrict__ xd, float* __restrict__ resd, int M, int N) {
    extern __shared__ float smem[];
    int row = blockIdx.x;
    int tid = threadIdx.x;
    if (row >= M) return;

    float* input_row  = xd   + row * N;
    float* output_row = resd + row * N;

    int    n_float4s = N / 4;
    int    tail      = N % 4;
    float4* in_vec   = reinterpret_cast<float4*>(input_row);
    float4* out_vec  = reinterpret_cast<float4*>(output_row);

    // pass 1: online softmax with float4 loads
    float local_max  = -__int_as_float(0x7f800000);
    float local_norm = 0.0f;
    for (int i = tid; i < n_float4s; i += blockDim.x) {
        float4 e = in_vec[i];
        float  chunk_max = fmaxf(fmaxf(e.x, e.y), fmaxf(e.z, e.w));
        if (chunk_max > local_max) { local_norm *= __expf(local_max - chunk_max); local_max = chunk_max; }
        local_norm += __expf(e.x - local_max) + __expf(e.y - local_max)
                    + __expf(e.z - local_max) + __expf(e.w - local_max);
    }
    // scalar tail
    if (tail && tid < tail) {
        float v = input_row[n_float4s * 4 + tid];
        if (v > local_max) { local_norm *= __expf(local_max - v); local_max = v; }
        local_norm += __expf(v - local_max);
    }

    // shuffle block reduce: max
    blockReduceMax<float>(local_max, smem, -__int_as_float(0x7f800000));
    float row_max = smem[0];

    // rescale then shuffle block reduce: sum
    blockReduceSum<float>(local_norm * __expf(local_max - row_max), smem);
    float row_norm = smem[0];

    // pass 2: normalize with float4 stores
    for (int i = tid; i < n_float4s; i += blockDim.x) {
        float4 e = in_vec[i];
        e.x = __expf(e.x - row_max) / row_norm;
        e.y = __expf(e.y - row_max) / row_norm;
        e.z = __expf(e.z - row_max) / row_norm;
        e.w = __expf(e.w - row_max) / row_norm;
        out_vec[i] = e;
    }
    if (tail && tid < tail)
        output_row[n_float4s * 4 + tid] = __expf(input_row[n_float4s * 4 + tid] - row_max) / row_norm;
}
void run_kernel_4(float* matd, float* resd, int M, int N) {
    size_t smem = CEIL_DIV(1024, 32) * sizeof(float);
    softmax_kernel_4<<<M, 1024, smem>>>(matd, resd, M, N);
}

// ─── timing helper ────────────────────────────────────────────────────────────
float time_kernel(void (*fn)(float*, float*, int, int),
                  float* dm, float* dr, int M, int N,
                  int warmup = 10, int runs = 200) {
    for (int i = 0; i < warmup; i++) fn(dm, dr, M, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0)); CUDA_CHECK(cudaEventCreate(&t1));
    CUDA_CHECK(cudaEventRecord(t0));
    for (int i = 0; i < runs; i++) fn(dm, dr, M, N);
    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));
    float ms; CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
    CUDA_CHECK(cudaEventDestroy(t0)); CUDA_CHECK(cudaEventDestroy(t1));
    return ms / runs;
}

int main() {
    const int M = 8192, N = 8192;
    const size_t bytes = (size_t)M * N * sizeof(float);
    const double peak_bw = 288.0;

    float* h_mat = new float[M * N];
    float* h_ref = new float[M * N]();
    float* h_res = new float[M * N]();

    srand(42);
    for (int i = 0; i < M * N; i++) h_mat[i] = ((float)rand() / RAND_MAX) * 4.f - 2.f;

    for (int r = 0; r < M; r++) {
        float m = -INFINITY, L = 0.f;
        for (int c = 0; c < N; c++) m = fmaxf(m, h_mat[r*N+c]);
        for (int c = 0; c < N; c++) L += expf(h_mat[r*N+c] - m);
        for (int c = 0; c < N; c++) h_ref[r*N+c] = expf(h_mat[r*N+c] - m) / L;
    }

    float *dm, *dr;
    CUDA_CHECK(cudaMalloc(&dm, bytes));
    CUDA_CHECK(cudaMalloc(&dr, bytes));
    CUDA_CHECK(cudaMemcpy(dm, h_mat, bytes, cudaMemcpyHostToDevice));

    struct { const char* name; void (*fn)(float*, float*, int, int); int passes; } kernels[] = {
        {"naive (3-pass)",        run_kernel_0, 4},
        {"online (2-pass)",       run_kernel_1, 3},
        {"coalesced warp 2-pass", run_kernel_2, 3},
        {"smem block 1024t",      run_kernel_3, 3},
        {"vectorized float4",     run_kernel_4, 3},
    };

    printf("GPU: GTX 1660 Ti | Peak BW: %.0f GB/s | Matrix: %dx%d (%.1f MB)\n\n",
           peak_bw, M, N, bytes/1e6);
    printf("%-18s | passes | time(ms) | BW(GB/s) | %%peak | max_err\n", "kernel");
    printf("──────────────────+────────+──────────+──────────+───────+────────\n");

    for (auto& k : kernels) {
        float ms = time_kernel(k.fn, dm, dr, M, N);
        CUDA_CHECK(cudaMemcpy(h_res, dr, bytes, cudaMemcpyDeviceToHost));
        float max_err = 0.f;
        for (int i = 0; i < M*N; i++) max_err = fmaxf(max_err, fabsf(h_res[i]-h_ref[i]));
        double bw  = ((double)k.passes * bytes) / (ms * 1e-3) / 1e9;
        double pct = bw / peak_bw * 100.0;
        printf("%-18s |   %d    | %8.4f | %8.2f | %5.1f%% | %.2e\n",
               k.name, k.passes, ms, bw, pct, max_err);
    }

    delete[] h_mat; delete[] h_ref; delete[] h_res;
    cudaFree(dm); cudaFree(dr);
    return 0;
}
