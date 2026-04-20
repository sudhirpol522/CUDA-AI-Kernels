#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cfloat>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// your kernel from coasled_access.cu (bugs fixed)
__global__ void warpReduceMax(float* input, float* output, int M, int N) {
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    if (bid >= M) return;
    float max_val = -__int_as_float(0x7f800000);
    for (int col = tid; col < N; col += blockDim.x) {
        max_val = fmax(max_val, input[bid * N + col]);
    }
    for (int offset = 16; offset > 0; offset /= 2)
        max_val = fmax(max_val, __shfl_down_sync(0xffffffff, max_val, offset));
    if (tid == 0)
        output[bid] = max_val;
}

int main() {
    const int M = 1024, N = 1024;

    float* h_mat = new float[M * N];
    float* h_out = new float[M]();

    srand(42);
    for (int i = 0; i < M * N; i++)
        h_mat[i] = ((float)rand() / RAND_MAX) * 100.f - 50.f;

    // plant a known max in row 500 so we can verify
    h_mat[500 * N + 777] = 9999.f;

    float *d_mat, *d_out;
    CUDA_CHECK(cudaMalloc(&d_mat, M * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, M     * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_mat, h_mat, M * N * sizeof(float), cudaMemcpyHostToDevice));

    warpReduceMax<<<M, 32>>>(d_mat, d_out, M, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_out, d_out, M * sizeof(float), cudaMemcpyDeviceToHost));

    printf("Row |  GPU max  |  CPU max  | match?\n");
    printf("----+-----------+-----------+-------\n");
    for (int r = 0; r < 10; r++) {
        float ref = -FLT_MAX;
        for (int c = 0; c < N; c++) ref = fmax(ref, h_mat[r * N + c]);
        printf(" %3d | %9.4f | %9.4f | %s\n",
               r, h_out[r], ref, fabsf(h_out[r] - ref) < 1e-4f ? "OK" : "FAIL");
    }

    printf("\nplanted 9999 in row 500 col 777 -> GPU got: %.1f\n", h_out[500]);

    delete[] h_mat;
    delete[] h_out;
    cudaFree(d_mat);
    cudaFree(d_out);
    return 0;
}
