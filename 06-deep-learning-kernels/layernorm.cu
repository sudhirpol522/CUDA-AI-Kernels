// Layer Normalization kernel
//
// y[i] = gamma[i] * (x[i] - mean) / sqrt(var + eps) + beta[i]
//
// where mean and var are computed over the N elements of each row.
//
// Strategy (same reduction pattern as the mean kernel, extended to two passes):
//   Pass 1: each thread accumulates partial sum -> warp reduce -> block reduce -> row_mean
//   Pass 2: each thread accumulates partial variance -> same reduction -> row_var
//   Pass 3: normalize and apply affine transform (gamma, beta)
//
// One block per row. blockDim.x should be a multiple of 32 and <= 1024.

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

// Reduces val across the entire block and returns the result to every thread.
__device__ __forceinline__ float blockReduceSum(float val, float* smem) {
    int tid = threadIdx.x;

    // Phase 1: reduce within each warp
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);

    // Each warp's lane-0 writes its partial sum to shared memory
    if (tid % 32 == 0)
        smem[tid / 32] = val;
    __syncthreads();

    // Phase 2: first warp reduces the per-warp partial sums
    if (tid < warpSize) {
        val = (tid < (blockDim.x + 31) / 32) ? smem[tid] : 0.0f;
        for (int offset = warpSize / 2; offset > 0; offset /= 2)
            val += __shfl_down_sync(0xffffffff, val, offset);
        if (tid == 0) smem[0] = val;   // broadcast result through shared memory
    }
    __syncthreads();

    return smem[0];  // same value returned to all threads
}

__global__ void layernorm_kernel(const float* __restrict__ input,
                                 float* __restrict__ output,
                                 const float* __restrict__ gamma,
                                 const float* __restrict__ beta,
                                 int M, int N, float eps) {
    __shared__ float smem[32];   // one slot per warp (max 32 warps for 1024-thread block)

    int row = blockIdx.x;
    int tid = threadIdx.x;
    if (row >= M) return;

    const float* x = input  + row * N;
    float*       y = output + row * N;

    // Pass 1: mean
    float sum = 0.0f;
    for (int i = tid; i < N; i += blockDim.x)
        sum += x[i];
    float row_mean = blockReduceSum(sum, smem) / N;

    // Pass 2: variance  E[(x - mean)^2]
    float var = 0.0f;
    for (int i = tid; i < N; i += blockDim.x) {
        float d = x[i] - row_mean;
        var += d * d;
    }
    float row_var = blockReduceSum(var, smem) / N;
    float inv_std = rsqrtf(row_var + eps);

    // Pass 3: normalize + affine transform
    for (int i = tid; i < N; i += blockDim.x) {
        float norm = (x[i] - row_mean) * inv_std;
        y[i] = gamma[i] * norm + beta[i];
    }
}

void run_layernorm(const float* input, float* output,
                   const float* gamma, const float* beta,
                   int M, int N, float eps) {
    dim3 block(256);
    dim3 grid(M);
    layernorm_kernel<<<grid, block>>>(input, output, gamma, beta, M, N, eps);
}

// ---------- correctness test ----------
int main() {
    const int M = 512, N = 1024;
    const float eps = 1e-5f;

    size_t mat_bytes = (size_t)M * N * sizeof(float);
    size_t vec_bytes = N * sizeof(float);

    float* h_in    = new float[M * N];
    float* h_out   = new float[M * N];
    float* h_gamma = new float[N];
    float* h_beta  = new float[N];
    float* h_ref   = new float[M * N];

    srand(42);
    for (int i = 0; i < M * N; i++)
        h_in[i] = (float)rand() / RAND_MAX * 4.0f - 2.0f;
    for (int i = 0; i < N; i++) {
        h_gamma[i] = 1.0f + (float)rand() / RAND_MAX * 0.1f;
        h_beta[i]  = (float)rand() / RAND_MAX * 0.1f;
    }

    // CPU reference
    for (int r = 0; r < M; r++) {
        double mean = 0.0, var = 0.0;
        for (int c = 0; c < N; c++) mean += h_in[r * N + c];
        mean /= N;
        for (int c = 0; c < N; c++) {
            double d = h_in[r * N + c] - mean;
            var += d * d;
        }
        var /= N;
        float inv_std = 1.0f / sqrtf((float)var + eps);
        for (int c = 0; c < N; c++) {
            float norm = (h_in[r * N + c] - (float)mean) * inv_std;
            h_ref[r * N + c] = h_gamma[c] * norm + h_beta[c];
        }
    }

    float *d_in, *d_out, *d_gamma, *d_beta;
    CUDA_CHECK(cudaMalloc(&d_in,    mat_bytes));
    CUDA_CHECK(cudaMalloc(&d_out,   mat_bytes));
    CUDA_CHECK(cudaMalloc(&d_gamma, vec_bytes));
    CUDA_CHECK(cudaMalloc(&d_beta,  vec_bytes));
    CUDA_CHECK(cudaMemcpy(d_in,    h_in,    mat_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_gamma, h_gamma, vec_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_beta,  h_beta,  vec_bytes, cudaMemcpyHostToDevice));

    run_layernorm(d_in, d_out, d_gamma, d_beta, M, N, eps);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_out, d_out, mat_bytes, cudaMemcpyDeviceToHost));

    float max_err = 0.0f;
    for (int i = 0; i < M * N; i++) {
        float err = fabsf(h_out[i] - h_ref[i]);
        if (err > max_err) max_err = err;
    }
    printf("M=%d N=%d  max_abs_err=%.2e  -> %s\n",
           M, N, max_err, max_err < 1e-4f ? "OK" : "FAIL");

    delete[] h_in; delete[] h_out; delete[] h_gamma;
    delete[] h_beta; delete[] h_ref;
    cudaFree(d_in); cudaFree(d_out); cudaFree(d_gamma); cudaFree(d_beta);
    return 0;
}
