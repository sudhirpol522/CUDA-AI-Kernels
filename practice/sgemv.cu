// GEMV Naive Implementation
// Based in part on Maharshi Pandya's CUDA optimization blog (Apache-2.0 license)
// https://github.com/Maharshi-Pandya/cuda-mode-resource-stream

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

// Ceiling division macro
#ifndef CEIL_DIV
#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))
#endif

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

/**
 * Naive SGEMM kernel for matrix-vector multiplication
 * Computes: result = matrix * vector
 * 
 * Performance characteristics:
 * - Each thread calculates one element of the output vector
 * - The row index is calculated using block index and thread index
 * - Uses linearized indexing
 * - Memory accesses are not coalesced (poor performance)
 * 
 * @param matd Input matrix (M×N, device memory)
 * @param vecd Input vector (N, device memory)
 * @param resd Output vector (M, device memory)
 * @param M Number of rows in matrix and size of output vector
 * @param N Number of columns in matrix and size of input vector
 */
__global__ void naive_sgemv_kernel(float* __restrict__ matd, float* __restrict__ vecd, float* __restrict__ resd, int M, int N) {
    // Calculate global thread index across all blocks
    int row = blockDim.x * blockIdx.x + threadIdx.x;

    // Bounds check to ensure we don't access out-of-range elements
    if (row < M) {
        float sum = 0.0f;
        // Compute dot product of matrix row and input vector
        for (int col = 0; col < N; col++) {
            printf("thread %d: matd[%d][%d] = %f, vecd[%d] = %f\n", threadIdx.x, row, col, matd[row * N + col], col, vecd[col]);
            break;
            sum += matd[row * N + col] * vecd[col];
        }
        // Store result in output vector
        resd[row] = sum;
    }
}

/**
 * CUDA wrapper function for naive SGEMM kernel
 * Launches the kernel with appropriate grid and block dimensions
 * 
 * @param matd Input matrix (M×N, device memory)
 * @param vecd Input vector (N, device memory)
 * @param resd Output vector (M, device memory)
 * @param M Number of rows in matrix and size of output vector
 * @param N Number of columns in matrix and size of input vector
 */
void run_kernel_1(float* __restrict__ matd, float* __restrict__ vecd, float* __restrict__ resd, int M, int N) {
    // Configure kernel launch parameters
    dim3 block_size(1024);  // Maximum threads per block
    dim3 grid_size(CEIL_DIV(M, block_size.x));  // Number of blocks needed

    // Launch CUDA kernel
    naive_sgemv_kernel<<<grid_size, block_size>>>(matd, vecd, resd, M, N);
}

// ---------------------------------------------------------------------------
// main: small 4×4 test so the per-thread printf stays readable
//
// NOTE: the kernel currently has a printf + break on line 53-54.
// That means each thread prints only the FIRST element of its row and
// then exits the loop, so resd[row] = 0 for all rows.
// The debug print shows you exactly which matrix cell and vector element
// each thread would access on its first step.
// ---------------------------------------------------------------------------
int main() {
    const int M = 4, N = 4;

    // Host data
    // Matrix (row-major):
    //  [ 1  2  3  4 ]
    //  [ 5  6  7  8 ]
    //  [ 9 10 11 12 ]
    //  [13 14 15 16 ]
    float h_mat[M * N], h_vec[N], h_res[M] = {};
    for (int i = 0; i < M * N; i++) h_mat[i] = (float)(i + 1);
    for (int j = 0; j < N;     j++) h_vec[j] = 1.0f;  // vector of all 1s

    printf("=== Input ===\n");
    printf("Matrix [%d×%d]:\n", M, N);
    for (int r = 0; r < M; r++) {
        printf("  row%d: [", r);
        for (int c = 0; c < N; c++) printf(" %5.1f", h_mat[r*N+c]);
        printf(" ]\n");
    }
    printf("Vector [%d]: [", N);
    for (int j = 0; j < N; j++) printf(" %.1f", h_vec[j]);
    printf(" ]\n\n");

    // Allocate + copy to device
    float *d_mat, *d_vec, *d_res;
    CUDA_CHECK(cudaMalloc(&d_mat, M * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_vec, N     * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_res, M     * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_mat, h_mat, M * N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vec, h_vec, N     * sizeof(float), cudaMemcpyHostToDevice));

    printf("=== Kernel printf output (one line per thread, only col=0 due to break) ===\n");
    run_kernel_1(d_mat, d_vec, d_res, M, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result back
    CUDA_CHECK(cudaMemcpy(h_res, d_res, M * sizeof(float), cudaMemcpyDeviceToHost));

    printf("\n=== Result vector (all 0 because break skips the accumulation) ===\n");
    for (int r = 0; r < M; r++)
        printf("  resd[%d] = %.1f\n", r, h_res[r]);

    // CPU reference (what it SHOULD produce)
    printf("\n=== CPU reference (dot product of each row with [1,1,1,1]) ===\n");
    for (int r = 0; r < M; r++) {
        float ref = 0.f;
        for (int c = 0; c < N; c++) ref += h_mat[r*N+c] * h_vec[c];
        printf("  ref[%d] = %.1f   (row sum: %d+%d+%d+%d)\n",
               r, ref,
               (int)h_mat[r*N+0], (int)h_mat[r*N+1],
               (int)h_mat[r*N+2], (int)h_mat[r*N+3]);
    }

    cudaFree(d_mat); cudaFree(d_vec); cudaFree(d_res);
    return 0;
}