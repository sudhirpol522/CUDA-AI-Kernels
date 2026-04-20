// GEMV Coalesced Warp Implementation
// Based in part on Maharshi Pandya's CUDA optimization blog (Apache-2.0 license)
// https://github.com/Maharshi-Pandya/cuda-mode-resource-stream

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cassert>

/**
 * CUDA error checking macro
 * Checks CUDA function calls for errors and exits on failure
 */
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
/**
 * Warp-level sum reduction using shuffle operations
 * Efficiently reduces values across all threads in a warp (32 threads)
 * Uses CUDA shuffle intrinsics for low-latency communication
 * 
 * @param val Input value to reduce
 * @return Sum of all values in the warp
 */
__device__ __forceinline__ float warpReduceSum(float val, int bid, int tid) {
    // Perform reduction in log2(32) = 5 steps
    // Each step halves the offset distance
    printf("warpReduceSum: val = %f, bid = %d, tid = %d\n", val, bid, tid);
    for (int offset = 16; offset > 0; offset /= 2) {
        // Shuffle down: get value from thread (current + offset)
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}
} 

/**
 * Coalesced Warp SGEMM kernel for matrix-vector multiplication
 * Optimized version using warp-level parallelism and coalesced memory access
 * 
 * Key optimizations:
 * - Each block processes one row of the matrix
 * - Each block calculates one output element
 * - Columns are accessed in coalesced manner by threads (stride-pattern)
 * - Performs warp-level sum reduction only (no block-level reduction needed)
 * - Block size must equal warp size (32 threads)
 * 
 * Memory access pattern:
 * - Threads access consecutive columns, enabling coalesced memory reads
 * - Each thread processes multiple columns with stride = blockDim.x
 * 
 * @param matd Input matrix (M×N, device memory)
 * @param vecd Input vector (N, device memory)
 * @param resd Output vector (M, device memory)
 * @param M Number of rows in matrix
 * @param N Number of columns in matrix
 */
__global__ void coalesced_warp_sgmev_kernel(float* __restrict__ matd, float* __restrict__ vecd, float* __restrict__ resd, int M, int N) {
    // Ensure block size equals warp size for optimal performance
    assert(blockDim.x == warpSize);

    // Get block index (one block per row)
    int bid = blockIdx.x;
    if (bid >= M) return;

    // Get thread index within block
    int tid = threadIdx.x;
    
    // Compute partial sum using coalesced memory access pattern
    // Each thread processes columns at stride = blockDim.x
    float partial_sum = 0.f;
    for (int col = tid; col < N; col += blockDim.x) {
        partial_sum += matd[bid * N + col] * vecd[col];
    }

    // Perform warp-level reduction to get final sum
    // Thread 0 will have the final result
    float sum = warpReduceSum(partial_sum, bid, tid);
    if (tid == 0) {
        resd[bid] = sum;
    }
}

/**
 * Launcher function for coalesced warp SGEMM kernel
 * Configures and launches the kernel with warp-sized blocks
 * 
 * @param matd Input matrix (M×N, device memory)
 * @param vecd Input vector (N, device memory)
 * @param resd Output vector (M, device memory)
 * @param M Number of rows in matrix
 * @param N Number of columns in matrix
 */
void run_kernel_2(float* __restrict__ matd, float* __restrict__ vecd, float* __restrict__ resd, int M, int N) {
    // Use warp size (32 threads) for optimal performance
    int NUM_THREADS = 32;

    // Configure kernel launch parameters
    // One block per row, each block has warp size threads
    dim3 block_size(NUM_THREADS);
    dim3 grid_size(M);

    // Launch CUDA kernel
    coalesced_warp_sgmev_kernel<<<grid_size, block_size>>>(matd, vecd, resd, M, N);
}

int main() {
    const int M = 4, N = 8;

    // host data — matrix filled 1..M*N, vector all 1s
    float h_mat[M * N], h_vec[N], h_res[M] = {};
    for (int i = 0; i < M * N; i++) h_mat[i] = (float)(i + 1);
    for (int j = 0; j < N;     j++) h_vec[j] = 1.0f;

    // allocate + copy to device
    float *d_mat, *d_vec, *d_res;
    CUDA_CHECK(cudaMalloc(&d_mat, M * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_vec, N     * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_res, M     * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_mat, h_mat, M * N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vec, h_vec, N     * sizeof(float), cudaMemcpyHostToDevice));

    run_kernel_2(d_mat, d_vec, d_res, M, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_res, d_res, M * sizeof(float), cudaMemcpyDeviceToHost));

    // print input
    printf("Matrix [%d x %d]:\n", M, N);
    for (int r = 0; r < M; r++) {
        printf("  row%d: [", r);
        for (int c = 0; c < N; c++) printf(" %5.1f", h_mat[r * N + c]);
        printf(" ]\n");
    }
    printf("Vector: [");
    for (int j = 0; j < N; j++) printf(" %.1f", h_vec[j]);
    printf(" ]\n\n");

    // print result vs cpu reference
    printf("Row | GPU result | CPU ref | match?\n");
    printf("----+------------+---------+-------\n");
    for (int r = 0; r < M; r++) {
        float ref = 0.f;
        for (int c = 0; c < N; c++) ref += h_mat[r * N + c] * h_vec[c];
        printf("  %d |   %7.1f  |  %7.1f | %s\n",
               r, h_res[r], ref, fabsf(h_res[r] - ref) < 0.01f ? "OK" : "FAIL");
    }

    cudaFree(d_mat); cudaFree(d_vec); cudaFree(d_res);
    return 0;
}