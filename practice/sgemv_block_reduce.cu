// GEMV Coalesced Warp+Block Implementation
// Based in part on Maharshi Pandya's CUDA optimization blog (Apache-2.0 license)
// https://github.com/Maharshi-Pandya/cuda-mode-resource-stream

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <cmath>

// Ceiling division macro
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

namespace {

__device__ __forceinline__ float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

template<typename T>
__device__ __forceinline__ void blockReduceSum(T val, T* smem, int tid, int block_size) {
    int warp_size = 32;

    // Phase 1: every thread calls warpReduceSum at the same time.
    // warp 0 (tid 0-31) reduces among itself -> tid 0  holds sum of warp 0
    // warp 1 (tid 32-63) reduces among itself -> tid 32 holds sum of warp 1
    // both warps run IN PARALLEL, not one after the other
    val = warpReduceSum(val);
    if (tid % warp_size == 0)
        printf("[tid %2d | warp %d] after warpReduceSum = %.1f\n",
               tid, tid / warp_size, val);

    // Phase 2: lane 0 of each warp writes its result to shared memory
    if (tid % warp_size == 0) smem[tid / warp_size] = val;
    __syncthreads();
    if (tid % warp_size == 0)
        printf("[tid %2d] wrote %.1f to smem[%d]\n", tid, val, tid / warp_size);

    // Phase 3: load warp sums back — only first (num_warps) threads get real values
    if (tid < CEIL_DIV(block_size, warp_size)) {
        val = smem[tid];
        printf("[tid %2d] loaded smem[%d] = %.1f\n", tid, tid, val);
    } else {
        val = 0.0f;
    }

    // Phase 4: warp 0 does one final warpReduceSum over the warp sums
    // tid 0 gets smem[0]+smem[1] = total block sum
    if (tid / warp_size == 0) {
        val = warpReduceSum(val);
    }

    // Phase 5: tid 0 stores final result
    if (tid == 0) {
        printf("[tid  0] FINAL block sum = %.1f\n", val);
        smem[0] = val;
    }
    __syncthreads();
}

} // namespace

__global__ void coalesced_warpblock_sgmev_kernel(float* __restrict__ matd, float* __restrict__ vecd, float* __restrict__ resd, int M, int N) {
    extern __shared__ float smem[];

    int bid = blockIdx.x;
    if (bid >= M) return;

    int tid = threadIdx.x;
    
    float partial_sum = 0.f;
    for (int col = tid; col < N; col += blockDim.x) {
        partial_sum += matd[bid * N + col] * vecd[col];
    }
    printf("[row %d | tid %2d | warp %d] partial_sum = %.1f\n",
           bid, tid, tid / 32, partial_sum);
    __syncthreads();

    blockReduceSum(partial_sum, smem, tid, blockDim.x);
    
    if (tid == 0) {
        resd[bid] = smem[0];
    }
}

void run_kernel_3(float* __restrict__ matd, float* __restrict__ vecd, float* __restrict__ resd, int M, int N) {
    int NUM_THREADS = 64;
    int warp_size = 32;

    dim3 block_size(NUM_THREADS);
    dim3 grid_size(M);
    
    size_t shared_mem_size = CEIL_DIV(block_size.x, warp_size) * sizeof(float);

    coalesced_warpblock_sgmev_kernel<<<grid_size, block_size, shared_mem_size>>>(matd, vecd, resd, M, N);
}

int main() {
    // 1 row, 64 columns, 64 threads (= 2 warps) — small enough to read every line
    // mat[0][col] = col+1, vec = all 1s
    // expected: 1+2+...+64 = 2080
    const int M = 1, N = 64;

    float h_mat[M * N], h_vec[N], h_res[M] = {};
    for (int i = 0; i < M * N; i++) h_mat[i] = (float)(i + 1);
    for (int j = 0; j < N;     j++) h_vec[j] = 1.f;

    float *d_mat, *d_vec, *d_res;
    CUDA_CHECK(cudaMalloc(&d_mat, M * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_vec, N     * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_res, M     * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_mat, h_mat, M * N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vec, h_vec, N     * sizeof(float), cudaMemcpyHostToDevice));

    run_kernel_3(d_mat, d_vec, d_res, M, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_res, d_res, M * sizeof(float), cudaMemcpyDeviceToHost));

    float ref = 0.f;
    for (int c = 0; c < N; c++) ref += h_mat[c] * h_vec[c];
    printf("\nGPU = %.1f  CPU = %.1f  %s\n", h_res[0], ref,
           fabsf(h_res[0] - ref) < 0.1f ? "OK" : "FAIL");

    cudaFree(d_mat); cudaFree(d_vec); cudaFree(d_res);
    return 0;
}
