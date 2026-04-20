#include <cuda_runtime.h>
#include <cstdio>   // was "cstudio" — typo
#include <cmath>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

__global__ void naivematrixmultiplication(float* mat, float* vec, float* output, int M, int N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M) {
        float ans = 0.0f;
        for (int col = 0; col < N; col++) {
            ans += mat[row * N + col] * vec[col];  // was "ans+ ="
        }
        output[row] = ans;
    }
}

__global__ void coalsedmatrixmultiplication(float* mat, float* vec, float* output, int M, int N) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    if (row < M) {
        float ans = 0.0f;
        for (int col = tid; col < N; col += blockDim.x) {
            ans += mat[row * N + col] * vec[col];  // was "ans+ ="
        }
        // bug fix: stride must start at warpSize/2 (=16), not warpSize (=32)
        // offset=32 is out of bounds for a 32-thread warp
        for (int stride = warpSize / 2; stride > 0; stride /= 2) {
            ans += __shfl_down_sync(0xffffffff, ans, stride);
        }
        if (tid == 0) {
            output[row] = ans;
        }
    }
}

int main() {
    const int M = 4, N = 8;

    float h_mat[M * N], h_vec[N], h_naive[M] = {}, h_coalesced[M] = {};
    for (int i = 0; i < M * N; i++) h_mat[i] = (float)(i + 1);
    for (int j = 0; j < N;     j++) h_vec[j] = 1.0f;

    float *d_mat, *d_vec, *d_out;
    CUDA_CHECK(cudaMalloc(&d_mat, M * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_vec, N     * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, M     * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_mat, h_mat, M * N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vec, h_vec, N     * sizeof(float), cudaMemcpyHostToDevice));

    // naive: one thread per row, block of 32
    naivematrixmultiplication<<<1, 32>>>(d_mat, d_vec, d_out, M, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_naive, d_out, M * sizeof(float), cudaMemcpyDeviceToHost));

    // coalesced: one block per row, 32 threads per block
    coalsedmatrixmultiplication<<<M, 32>>>(d_mat, d_vec, d_out, M, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_coalesced, d_out, M * sizeof(float), cudaMemcpyDeviceToHost));

    printf("Row |  CPU ref  |  naive  | coalesced\n");
    printf("----+-----------+---------+----------\n");
    for (int r = 0; r < M; r++) {
        float ref = 0.f;
        for (int c = 0; c < N; c++) ref += h_mat[r * N + c] * h_vec[c];
        printf("  %d |   %6.1f  |  %6.1f | %6.1f  %s\n",
               r, ref, h_naive[r], h_coalesced[r],
               (fabsf(h_naive[r]-ref)<0.01f && fabsf(h_coalesced[r]-ref)<0.01f) ? "OK" : "FAIL");
    }

    cudaFree(d_mat); cudaFree(d_vec); cudaFree(d_out);
    return 0;
}
