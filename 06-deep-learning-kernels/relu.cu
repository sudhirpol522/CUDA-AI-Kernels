// ReLU activation kernel
//
// Elementwise y[i] = max(x[i], 0). Memory-bound: 1 load + 1 store per element,
// zero arithmetic intensity, so performance is gated by DRAM bandwidth.
//
// This version uses a grid-stride loop so the kernel works correctly for any N
// even when the launch grid is clamped below ceil(N/256) (e.g. when the caller
// wants to cap the number of blocks to match the GPU's SM count).

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

__global__ void relu_kernel(const float* input, float* output, int N) {
    int ind = blockDim.x * blockIdx.x + threadIdx.x;
    for (int i = ind; i < N; i += blockDim.x * gridDim.x) {
        output[i] = input[i] > 0.0f ? input[i] : 0.0f;
    }
}

extern "C" void solve(const float* input, float* output, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    relu_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N);
    cudaDeviceSynchronize();
}

int main() {
    const int N = 1 << 22;

    float* h_in  = new float[N];
    float* h_out = new float[N];
    srand(42);
    for (int i = 0; i < N; i++) {
        h_in[i] = (float)rand() / RAND_MAX * 4.0f - 2.0f;
    }

    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in,  N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice));

    solve(d_in, d_out, N);

    CUDA_CHECK(cudaMemcpy(h_out, d_out, N * sizeof(float), cudaMemcpyDeviceToHost));

    int errs = 0;
    for (int i = 0; i < N; i++) {
        float expected = h_in[i] > 0.0f ? h_in[i] : 0.0f;
        if (h_out[i] != expected) { errs++; if (errs < 5) printf("  mismatch at %d: %f vs %f\n", i, h_out[i], expected); }
    }
    printf("N = %d, errors = %d -> %s\n", N, errs, errs == 0 ? "OK" : "FAIL");

    delete[] h_in;
    delete[] h_out;
    cudaFree(d_in);
    cudaFree(d_out);
    return 0;
}
