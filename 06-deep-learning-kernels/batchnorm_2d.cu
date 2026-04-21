// Batch Normalization — 2D tiled kernel
//
// Improvement over batchnorm.cu (3-kernel approach):
//   - Single kernel launch (no inter-kernel synchronization overhead)
//   - One pass over input (reads each element once for both mean and variance)
//   - Coalesced memory access: threads in the same warp read consecutive channels
//
// Layout: row-major, input[n, c] = input[n*C + c]
//
// Thread block: 32 x 32 (1024 threads)
//   threadIdx.x → channel within the 32-channel tile
//   threadIdx.y → sample group (handles samples threadIdx.y, threadIdx.y+32, ...)
//
// Grid: ceil(C/32) blocks in x, one block per 32-channel tile
//
// Each block owns 32 consecutive channels and all N samples:
//   Phase 1: every thread accumulates sum and sum-of-squares for its sample slice
//   Phase 2: tree reduction along threadIdx.y collapses 32 partial sums per channel
//   Phase 3: compute mean and inv_std from shared memory, write normalized output
//
// Variance formula: Var = E[x^2] - E[x]^2  (one-pass, algebraically equivalent
// to the two-pass form but less numerically stable when values are large).
// fmaxf guard prevents negative variance from fp32 cancellation error.

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

const int TILE = 32;

__global__ void batchnorm_2d_kernel(const float* input, const float* gamma,
                                    const float* beta, float* output,
                                    int N, int C, float eps) {
    __shared__ float s_sum[TILE][TILE];
    __shared__ float s_var[TILE][TILE];

    int local_row = threadIdx.y;
    int local_col = threadIdx.x;
    int channel   = blockIdx.x * blockDim.x + local_col;

    // Phase 1: each thread accumulates its sample slice for one channel
    float s = 0.0f, v = 0.0f;
    if (channel < C) {
        for (int i = threadIdx.y; i < N; i += blockDim.y) {
            float val = input[i * C + channel];
            s += val;
            v += val * val;
        }
    }
    s_sum[local_row][local_col] = s;
    s_var[local_row][local_col] = v;
    __syncthreads();

    // Phase 2: tree reduction along the row (Y) dimension
    // After this, s_sum[0][col] holds the total sum for channel col
    for (int j = blockDim.y / 2; j > 0; j >>= 1) {
        if (threadIdx.y < j) {
            s_sum[local_row][local_col] += s_sum[local_row + j][local_col];
            s_var[local_row][local_col] += s_var[local_row + j][local_col];
        }
        __syncthreads();
    }

    // Phase 3: compute per-channel statistics (all threads share the result
    //          via shared memory row 0)
    float mean    = s_sum[0][threadIdx.x] / (float)N;
    float var     = fmaxf(s_var[0][threadIdx.x] / (float)N - mean * mean, 0.0f);
    float inv_std = rsqrtf(var + eps);

    float g = (channel < C) ? gamma[channel] : 0.0f;
    float b = (channel < C) ? beta[channel]  : 0.0f;

    // Write normalized output for this thread's sample slice
    for (int i = local_row; i < N; i += blockDim.y) {
        if (channel < C)
            output[i * C + channel] = g * (input[i * C + channel] - mean) * inv_std + b;
    }
}

extern "C" void solve(const float* input, const float* gamma, const float* beta,
                      float* output, int N, int C, float eps) {
    dim3 block(TILE, TILE);
    dim3 grid((C + TILE - 1) / TILE, 1);
    batchnorm_2d_kernel<<<grid, block>>>(input, gamma, beta, output, N, C, eps);
    CUDA_CHECK(cudaDeviceSynchronize());
}

int main() {
    const int N = 512, C = 256;
    const float eps = 1e-5f;
    const int total = N * C;

    float* h_in  = new float[total];
    float* h_g   = new float[C];
    float* h_b   = new float[C];
    float* h_out = new float[total];
    float* h_ref = new float[total];

    srand(42);
    for (int i = 0; i < total; i++) h_in[i] = (float)rand() / RAND_MAX * 4.0f - 2.0f;
    for (int c = 0; c < C; c++) {
        h_g[c] = 1.0f + (float)rand() / RAND_MAX * 0.1f;
        h_b[c] = (float)rand() / RAND_MAX * 0.1f;
    }

    // CPU reference (two-pass, double precision)
    for (int c = 0; c < C; c++) {
        double mean = 0.0, var = 0.0;
        for (int n = 0; n < N; n++) mean += h_in[n*C+c];
        mean /= N;
        for (int n = 0; n < N; n++) { double d = h_in[n*C+c]-mean; var += d*d; }
        var /= N;
        float inv_std = 1.0f / sqrtf((float)var + eps);
        for (int n = 0; n < N; n++) {
            int idx = n*C+c;
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
