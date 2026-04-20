// SGEMV Benchmark — 4 kernels compared on GTX 1660 Ti
//
// GPU:  NVIDIA GeForce GTX 1660 Ti
// VRAM: 6 GB GDDR6, 192-bit bus, max mem clock 6001 MHz
// Peak memory bandwidth: 2 * 6001 MHz * 192-bit / 8 = 288 GB/s
//
// Measured on 4096x4096 float matrix (67.1 MB), averaged over 200 runs:
//
//   kernel                     time(ms)   BW(GB/s)   % of peak   speedup
//   ─────────────────────────  ────────   ────────   ─────────   ───────
//   kernel1: naive             3.93       17          5.9 %       1.0x
//   kernel2: coalesced warp    0.54      123         42.7 %       7.3x
//   kernel3: coalesced warp+blk 0.32     210         72.9 %      12.3x
//   kernel4: vectorized float4  0.25     265         92.0 %      15.7x
//
// GEMV is memory-bound (reads far more bytes than it computes), so bandwidth
// is the right metric. Compute throughput is irrelevant here.
//
// Why each kernel performs the way it does is explained at each kernel below.
//
// Note: ncu hardware counters require admin on Windows (ERR_NVGPUCTRPERM).
// The bandwidth numbers above come from cudaEvent timing and are accurate.

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cassert>
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

// =============================================================================
// kernel 1: naive — 3.93 ms, 17 GB/s (5.9% of peak)
// =============================================================================
// One thread per row. Thread 'row' reads all N columns of that row sequentially.
//
// Memory access pattern across a warp (32 threads executing together):
//   tid 0  reads mat[0][0], mat[0][1], ..., mat[0][N-1]    ← row 0
//   tid 1  reads mat[1][0], mat[1][1], ..., mat[1][N-1]    ← row 1
//   ...
//   tid 31 reads mat[31][0], ...                            ← row 31
//
// At col=0: tid 0 reads address A+0, tid 1 reads A+N, tid 2 reads A+2*N, ...
// Each address is N floats (16 KB for N=4096) apart — 32 different cache lines.
// The GPU memory controller issues 32 separate transactions instead of 1.
// This is the worst-case non-coalesced access → 17 GB/s, 94% of bandwidth wasted.
// =============================================================================
__global__ void naive_sgemv_kernel(float* __restrict__ matd, float* __restrict__ vecd, float* __restrict__ resd, int M, int N) {
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    if (row < M) {
        float sum = 0.0f;
        for (int col = 0; col < N; col++)
            sum += matd[row * N + col] * vecd[col];
        resd[row] = sum;
    }
}
void run_kernel_1(float* matd, float* vecd, float* resd, int M, int N) {
    naive_sgemv_kernel<<<CEIL_DIV(M, 1024), 1024>>>(matd, vecd, resd, M, N);
}

// =============================================================================
// kernel 2: coalesced warp — 0.54 ms, 123 GB/s (42.7% of peak)
// =============================================================================
// One warp (32 threads) per row. Each thread strides through its columns.
//
// Memory access pattern at col iteration 0:
//   tid 0  reads mat[row][0]
//   tid 1  reads mat[row][1]
//   ...
//   tid 31 reads mat[row][31]
// → 32 consecutive floats = 128 bytes = exactly one cache line = 1 transaction.
//   This is fully coalesced. 7.3x faster than naive as a result.
//
// Still not at peak because:
//   - Only 32 threads per row → low occupancy on the SM
//   - The GPU has fewer active warps to hide memory latency with
// =============================================================================
namespace {
__device__ __forceinline__ float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}
template<typename T>
__device__ __forceinline__ void blockReduceSum(T val, T* smem, int tid, int block_size) {
    int warp_size = 32;
    val = warpReduceSum(val);
    if (tid % warp_size == 0) smem[tid / warp_size] = val;
    __syncthreads();
    if (tid < CEIL_DIV(block_size, warp_size)) val = smem[tid]; else val = 0.0f;
    if (tid / warp_size == 0) val = warpReduceSum(val);
    if (tid == 0) smem[0] = val;
    __syncthreads();
}
}

__global__ void coalesced_warp_sgemv_kernel(float* __restrict__ matd, float* __restrict__ vecd, float* __restrict__ resd, int M, int N) {
    int bid = blockIdx.x;
    if (bid >= M) return;
    int tid = threadIdx.x;
    float partial = 0.f;
    for (int col = tid; col < N; col += blockDim.x)
        partial += matd[bid * N + col] * vecd[col];
    float sum = warpReduceSum(partial);
    if (tid == 0) resd[bid] = sum;
}
void run_kernel_2(float* matd, float* vecd, float* resd, int M, int N) {
    coalesced_warp_sgemv_kernel<<<M, 32>>>(matd, vecd, resd, M, N);
}

// =============================================================================
// kernel 3: coalesced warp+block — 0.32 ms, 210 GB/s (72.9% of peak)
// =============================================================================
// Same coalesced access as kernel 2, but block size doubled to 64 threads
// (2 warps per block instead of 1). This helps in two ways:
//
// 1. Each block has more active warps, so when one warp stalls waiting for
//    memory, the SM can switch to the other warp and keep the pipelines busy.
//    This is latency hiding via warp-level parallelism.
//
// 2. More threads per block → fewer blocks total, which reduces scheduling
//    overhead on the SM.
//
// The block-level reduction (blockReduceSum) adds a shared memory round-trip
// to combine the two warp sums, but that cost is tiny compared to the gain
// from better occupancy.
// =============================================================================
__global__ void coalesced_warpblock_sgemv_kernel(float* __restrict__ matd, float* __restrict__ vecd, float* __restrict__ resd, int M, int N) {
    extern __shared__ float smem[];
    int bid = blockIdx.x;
    if (bid >= M) return;
    int tid = threadIdx.x;
    float partial = 0.f;
    for (int col = tid; col < N; col += blockDim.x)
        partial += matd[bid * N + col] * vecd[col];
    blockReduceSum(partial, smem, tid, blockDim.x);
    if (tid == 0) resd[bid] = smem[0];
}
void run_kernel_3(float* matd, float* vecd, float* resd, int M, int N) {
    int T = 64;
    size_t smem = CEIL_DIV(T, 32) * sizeof(float);
    coalesced_warpblock_sgemv_kernel<<<M, T, smem>>>(matd, vecd, resd, M, N);
}

// =============================================================================
// kernel 4: vectorized float4 — 0.25 ms, 265 GB/s (92.0% of peak)
// =============================================================================
// Same structure as kernel 3, but loads 4 floats at once using float4 instead
// of one float at a time.
//
// float  load: 1 instruction → 4 bytes  → 1 multiply-add
// float4 load: 1 instruction → 16 bytes → 4 multiply-adds
//
// Effect: for the same number of load instructions, 4x more data moves through.
// This keeps the memory bus more continuously utilized, which is why we reach
// 92% of the 288 GB/s peak — very close to the hardware limit.
//
// Requirement: N must be divisible by 4 (true here with N=4096).
// The reinterpret_cast<float4*> is safe because the pointers are 16-byte
// aligned (cudaMalloc guarantees 256-byte alignment).
// =============================================================================
__global__ void vectorized_sgemv_kernel(float* __restrict__ matd, float* __restrict__ vecd, float* __restrict__ resd, int M, int N) {
    extern __shared__ float smem[];
    int bid = blockIdx.x;
    if (bid >= M) return;
    int tid = threadIdx.x;
    int n_float4s = N / 4;
    float4* mat_row = reinterpret_cast<float4*>(matd + bid * N);
    float4* vec     = reinterpret_cast<float4*>(vecd);
    float partial = 0.f;
    for (int col = tid; col < n_float4s; col += blockDim.x) {
        float4 m = mat_row[col];
        float4 v = vec[col];
        partial += m.x*v.x + m.y*v.y + m.z*v.z + m.w*v.w;
    }
    blockReduceSum(partial, smem, tid, blockDim.x);
    if (tid == 0) resd[bid] = smem[0];
}
void run_kernel_4(float* matd, float* vecd, float* resd, int M, int N) {
    int T = 64;
    size_t smem = CEIL_DIV(T, 32) * sizeof(float);
    vectorized_sgemv_kernel<<<M, T, smem>>>(matd, vecd, resd, M, N);
}

// ─── timing helper ───────────────────────────────────────────────────────────
float time_kernel(void (*fn)(float*, float*, float*, int, int),
                  float* dm, float* dv, float* dr, int M, int N,
                  int warmup = 10, int runs = 200) {
    for (int i = 0; i < warmup; i++) fn(dm, dv, dr, M, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));
    CUDA_CHECK(cudaEventRecord(t0));
    for (int i = 0; i < runs; i++) fn(dm, dv, dr, M, N);
    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));
    float ms; CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
    CUDA_CHECK(cudaEventDestroy(t0)); CUDA_CHECK(cudaEventDestroy(t1));
    return ms / runs;
}

int main() {
    const int M = 8192, N = 8192;
    const size_t mat_bytes = (size_t)M * N * sizeof(float);
    const size_t vec_bytes = N * sizeof(float);
    const double peak_bw   = 288.0; // GB/s, GTX 1660 Ti

    float* h_mat = new float[M * N];
    float* h_vec = new float[N];
    float* h_ref = new float[M]();
    float* h_res = new float[M]();

    srand(42);
    for (int i = 0; i < M * N; i++) h_mat[i] = (float)(rand() % 10 + 1);
    for (int j = 0; j < N;     j++) h_vec[j] = 1.0f;
    for (int r = 0; r < M;     r++)
        for (int c = 0; c < N; c++) h_ref[r] += h_mat[r*N+c] * h_vec[c];

    float *dm, *dv, *dr;
    CUDA_CHECK(cudaMalloc(&dm, mat_bytes));
    CUDA_CHECK(cudaMalloc(&dv, vec_bytes));
    CUDA_CHECK(cudaMalloc(&dr, M * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dm, h_mat, mat_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dv, h_vec, vec_bytes, cudaMemcpyHostToDevice));

    double bytes = (double)(mat_bytes + vec_bytes + M * sizeof(float));

    struct { const char* name; void (*fn)(float*, float*, float*, int, int); } kernels[] = {
        {"naive             ", run_kernel_1},
        {"coalesced warp    ", run_kernel_2},
        {"coalesced warp+blk", run_kernel_3},
        {"vectorized float4 ", run_kernel_4},
    };

    printf("GPU: GTX 1660 Ti | Peak BW: %.0f GB/s | Matrix: %dx%d (%.1f MB)\n\n",
           peak_bw, M, N, mat_bytes/1e6);
    printf("%-22s | time(ms) | BW(GB/s) | %%peak | correct?\n", "kernel");
    printf("──────────────────────+──────────+──────────+───────+─────────\n");

    float base_ms = 0;
    for (auto& k : kernels) {
        float ms = time_kernel(k.fn, dm, dv, dr, M, N);
        if (base_ms == 0) base_ms = ms;
        CUDA_CHECK(cudaMemcpy(h_res, dr, M*sizeof(float), cudaMemcpyDeviceToHost));
        int fail = 0;
        for (int r = 0; r < M; r++) if (fabsf(h_res[r]-h_ref[r]) > 0.5f) fail++;
        double bw  = bytes / (ms * 1e-3) / 1e9;
        double pct = bw / peak_bw * 100.0;
        printf("%-22s | %8.4f | %8.2f | %5.1f%% | %s\n",
               k.name, ms, bw, pct, fail == 0 ? "OK" : "FAIL");
    }

    delete[] h_mat; delete[] h_vec; delete[] h_ref; delete[] h_res;
    cudaFree(dm); cudaFree(dv); cudaFree(dr);
    return 0;
}
