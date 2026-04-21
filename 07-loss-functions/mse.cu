// Mean Squared Error (MSE) loss kernel
//
// MSE = (1/N) * sum_i (preds[i] - targets[i])^2
//
// Implementation:
//   - Each thread computes (preds[i] - targets[i])^2 for a chunk of elements
//     (grid-stride loop, so N can be arbitrarily large).
//   - Warp reduce with __shfl_down_sync (5 shuffle rounds).
//   - One atomicAdd per block to accumulate into a single float in global memory.
//   - Host divides by N for the final mean.

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
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

__global__ void mse_kernel(const float* preds, const float* targets,
                           float* mse_out, int N) {
    __shared__ float warp_sums[32];

    int tid  = threadIdx.x;
    int gid  = blockIdx.x * blockDim.x + tid;
    int lane = tid & 31;
    int warp = tid >> 5;

    // grid-stride loop: each thread accumulates its own partial sum of squared errors
    float local = 0.0f;
    for (int i = gid; i < N; i += blockDim.x * gridDim.x) {
        float d = preds[i] - targets[i];
        local += d * d;
    }

    // warp reduction
    for (int s = 16; s > 0; s >>= 1)
        local += __shfl_down_sync(0xffffffff, local, s);

    if (lane == 0) warp_sums[warp] = local;
    __syncthreads();

    // first warp reduces per-warp sums and adds the block total into global memory
    if (warp == 0) {
        local = (tid < (blockDim.x + 31) / 32) ? warp_sums[lane] : 0.0f;
        for (int s = 16; s > 0; s >>= 1)
            local += __shfl_down_sync(0xffffffff, local, s);
        if (lane == 0) atomicAdd(mse_out, local);
    }
}

extern "C" void solve(const float* preds, const float* targets,
                      float* mse, int N) {
    CUDA_CHECK(cudaMemset(mse, 0, sizeof(float)));
    int threads = 256;
    int blocks  = min(1024, (N + threads - 1) / threads);
    mse_kernel<<<blocks, threads>>>(preds, targets, mse, N);

    float host_sum;
    CUDA_CHECK(cudaMemcpy(&host_sum, mse, sizeof(float), cudaMemcpyDeviceToHost));
    host_sum /= N;
    CUDA_CHECK(cudaMemcpy(mse, &host_sum, sizeof(float), cudaMemcpyHostToDevice));
}

int main() {
    const int N = 1 << 20; // 1,048,576 elements

    float* h_preds   = new float[N];
    float* h_targets = new float[N];

    srand(42);
    for (int i = 0; i < N; i++) {
        h_targets[i] = (float)rand() / RAND_MAX * 10.0f;
        h_preds[i]   = h_targets[i] + ((float)rand() / RAND_MAX * 2.0f - 1.0f);
    }

    // CPU reference
    double ref = 0.0;
    for (int i = 0; i < N; i++) {
        double d = (double)h_preds[i] - (double)h_targets[i];
        ref += d * d;
    }
    ref /= N;

    float *d_preds, *d_targets, *d_mse;
    CUDA_CHECK(cudaMalloc(&d_preds,   N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_targets, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_mse,         sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_preds,   h_preds,   N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_targets, h_targets, N * sizeof(float), cudaMemcpyHostToDevice));

    solve(d_preds, d_targets, d_mse, N);

    float h_mse;
    CUDA_CHECK(cudaMemcpy(&h_mse, d_mse, sizeof(float), cudaMemcpyDeviceToHost));

    printf("N        = %d\n", N);
    printf("GPU MSE  = %.6f\n", h_mse);
    printf("CPU MSE  = %.6f\n", ref);
    printf("abs err  = %.2e\n", fabs((double)h_mse - ref));
    printf("status   = %s\n", fabs((double)h_mse - ref) < 1e-3 ? "OK" : "FAIL");

    delete[] h_preds;
    delete[] h_targets;
    cudaFree(d_preds);
    cudaFree(d_targets);
    cudaFree(d_mse);
    return 0;
}
