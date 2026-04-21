// Batch Normalization kernel
//
// Normalizes over the batch (N) dimension for each channel independently:
//   y[n,c] = gamma[c] * (x[n,c] - mean[c]) / sqrt(var[c] + eps) + beta[c]
//
// Contrast with Layer Norm which normalizes over C for each sample.
//
// Data layout: row-major, x[n,c] = input[n*C + c]
// Because channels are strided (not contiguous), memory access in reduce_sum/
// reduce_var is non-coalesced — threads in a warp read across rows instead of
// across consecutive addresses. This is a known cost of batch norm on row-major
// data and is why batch norm is typically slower than layer norm at inference.
//
// Three-kernel approach:
//   1. reduce_sum  : per-channel mean   (one block per channel, tree reduction)
//   2. reduce_var  : per-channel variance (same pattern, needs mean from step 1)
//   3. apply_batch_norm : element-wise affine transform

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

template<int BLOCKSIZE>
__global__ void reduce_sum(const float* input, float* output, int N, int C) {
    __shared__ float shareMem[BLOCKSIZE];
    int channel_id = blockIdx.x;
    int tidx = threadIdx.x;
    int bdim = blockDim.x;
    if (channel_id >= C) return;

    shareMem[tidx] = 0.0f;
    for (int sample_id = tidx; sample_id < N; sample_id += bdim)
        shareMem[tidx] += input[sample_id * C + channel_id];
    __syncthreads();

    for (int stride = bdim / 2; stride > 0; stride >>= 1) {
        if (tidx < stride) shareMem[tidx] += shareMem[tidx + stride];
        __syncthreads();
    }
    if (tidx == 0) output[channel_id] = shareMem[0] / N;
}

template<int BLOCKSIZE>
__global__ void reduce_var(const float* input, const float* mean, float* output, int N, int C) {
    __shared__ float shareMem[BLOCKSIZE];
    int channel_id = blockIdx.x;
    int tidx = threadIdx.x;
    int bdim = blockDim.x;
    if (channel_id >= C) return;

    float channel_mean = mean[channel_id];
    shareMem[tidx] = 0.0f;
    for (int sample_id = tidx; sample_id < N; sample_id += bdim) {
        float diff = input[sample_id * C + channel_id] - channel_mean;
        shareMem[tidx] += diff * diff;
    }
    __syncthreads();

    for (int stride = bdim / 2; stride > 0; stride >>= 1) {
        if (tidx < stride) shareMem[tidx] += shareMem[tidx + stride];
        __syncthreads();
    }
    if (tidx == 0) output[channel_id] = shareMem[0] / N;
}

__global__ void apply_batch_norm(const float* input, const float* mean, const float* var,
                                 const float* gamma, const float* beta, float* output,
                                 int N, int C, float eps) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * C) return;
    int channel_id = idx % C;
    float x  = input[idx];
    float mu = mean[channel_id];
    float g  = gamma[channel_id];
    float b  = beta[channel_id];
    output[idx] = g * (x - mu) / sqrtf(var[channel_id] + eps) + b;
}

extern "C" void solve(const float* input, const float* gamma, const float* beta,
                      float* output, int N, int C, float eps) {
    const int BLOCKSIZE = 256;
    float *d_mean, *d_var;
    CUDA_CHECK(cudaMalloc(&d_mean, C * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_var,  C * sizeof(float)));

    reduce_sum<BLOCKSIZE><<<C, BLOCKSIZE>>>(input, d_mean, N, C);
    reduce_var<BLOCKSIZE><<<C, BLOCKSIZE>>>(input, d_mean, d_var, N, C);

    int total = N * C;
    apply_batch_norm<<<(total + 255) / 256, 256>>>(
        input, d_mean, d_var, gamma, beta, output, N, C, eps);

    CUDA_CHECK(cudaDeviceSynchronize());
    cudaFree(d_mean);
    cudaFree(d_var);
}

int main() {
    const int N = 64, C = 128;
    const float eps = 1e-5f;
    const int total = N * C;

    float* h_in    = new float[total];
    float* h_g     = new float[C];
    float* h_b     = new float[C];
    float* h_out   = new float[total];
    float* h_ref   = new float[total];

    srand(42);
    for (int i = 0; i < total; i++) h_in[i] = (float)rand() / RAND_MAX * 4.0f - 2.0f;
    for (int c = 0; c < C; c++) {
        h_g[c] = 1.0f + (float)rand() / RAND_MAX * 0.1f;
        h_b[c] = (float)rand() / RAND_MAX * 0.1f;
    }

    // CPU reference: per-channel mean and variance
    for (int c = 0; c < C; c++) {
        double mean = 0.0, var = 0.0;
        for (int n = 0; n < N; n++) mean += h_in[n * C + c];
        mean /= N;
        for (int n = 0; n < N; n++) { double d = h_in[n*C+c] - mean; var += d*d; }
        var /= N;
        float inv_std = 1.0f / sqrtf((float)var + eps);
        for (int n = 0; n < N; n++) {
            int idx = n * C + c;
            h_ref[idx] = h_g[c] * (h_in[idx] - (float)mean) * inv_std + h_b[c];
        }
    }

    float *d_in, *d_g, *d_b, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in,  total * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_g,   C     * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b,   C     * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, total * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_in, h_in, total * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_g,  h_g,  C     * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b,  h_b,  C     * sizeof(float), cudaMemcpyHostToDevice));

    solve(d_in, d_g, d_b, d_out, N, C, eps);

    CUDA_CHECK(cudaMemcpy(h_out, d_out, total * sizeof(float), cudaMemcpyDeviceToHost));

    float max_err = 0.0f;
    for (int i = 0; i < total; i++)
        max_err = fmaxf(max_err, fabsf(h_out[i] - h_ref[i]));

    printf("N=%d C=%d  max_abs_err=%.2e  -> %s\n",
           N, C, max_err, max_err < 1e-4f ? "OK" : "FAIL");

    delete[] h_in; delete[] h_g; delete[] h_b; delete[] h_out; delete[] h_ref;
    cudaFree(d_in); cudaFree(d_g); cudaFree(d_b); cudaFree(d_out);
    return 0;
}
