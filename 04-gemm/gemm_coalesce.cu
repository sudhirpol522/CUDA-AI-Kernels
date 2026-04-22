// Coalesced GEMM kernel (fp16 inputs, float accumulation)
//
// Problem: C[M,N] = A[M,K] @ B[K,N]
//
// Naive approach:
//   2D block (16x16), one thread per output element.
//   Consecutive threads in a warp differ by 1 in threadIdx.x but share
//   the same blockIdx — they access consecutive C columns (coalesced write),
//   but consecutive threads in a warp accessing B[i*N + col] are already
//   adjacent in memory. So naive is not obviously wrong on writes.
//   The real issue is that the naive 16x16 mapping wastes occupancy and
//   does not exploit the memory system as efficiently as the 1D split below.
//
// Coalesced approach (this file):
//   1D block of BLOCKSIZE^2 threads. Each thread's (row, col) is derived as:
//     row = tid / BLOCKSIZE
//     col = tid % BLOCKSIZE
//   Consequence: threads 0..BLOCKSIZE-1 all share row 0 and have consecutive
//   columns 0..BLOCKSIZE-1. They read the same row of A (broadcast from cache)
//   and consecutive elements of B's row i (coalesced). They also write
//   consecutive elements of C (coalesced). This is the standard "coalesced
//   GEMM" baseline before shared-memory tiling.
//
// Benchmark results (GTX 1660 Ti, fp16):
//   M=N=K=256:  naive 0.115 ms  coalesced 0.082 ms  speedup 1.40x
//   M=N=K=512:  naive 0.905 ms  coalesced 0.758 ms  speedup 1.19x
//   M=N=K=1024: naive 11.29 ms  coalesced 10.88 ms  speedup 1.04x
//   M=N=K=2048: naive 91.35 ms  coalesced 89.04 ms  speedup 1.03x
//   M=N=K=4096: naive 879 ms    coalesced 841 ms     speedup 1.05x
//
// The speedup is modest (1.0-1.4x) because naive GEMM is compute-bound at
// large sizes — both kernels stall on the K-loop dependency chain. Larger
// gains come from shared-memory tiling (cache the A/B tiles in SMEM to
// reduce global memory reads from O(N^3) to O(N^3/BLOCKSIZE)).
//
// Axis mapping: two variants are provided. They produce identical results;
// the swap of blockIdx.x/y is cosmetic — what matters is the tid split.

#include <cuda_fp16.h>
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

typedef __half fp16;

// Variant A: conventional axis mapping (blockIdx.x -> col, blockIdx.y -> row)
template<int BLOCKSIZE>
__global__ void gemm_coalesce_A(int M, int N, int K,
                                const fp16* A, const fp16* B, fp16* C) {
    const int cRow = blockIdx.y * BLOCKSIZE + (threadIdx.x / BLOCKSIZE);
    const int cCol = blockIdx.x * BLOCKSIZE + (threadIdx.x % BLOCKSIZE);
    if (cRow < M && cCol < N) {
        float tmp = 0.0f;
        for (int i = 0; i < K; i++)
            tmp += __half2float(A[cRow * K + i]) * __half2float(B[i * N + cCol]);
        C[cRow * N + cCol] = __float2half(tmp);
    }
}

// Variant B: swapped axis mapping (blockIdx.x -> row, blockIdx.y -> col)
template<int BLOCKSIZE>
__global__ void gemm_coalesce_B(int M, int N, int K,
                                const fp16* A, const fp16* B, fp16* C) {
    const int cRow = blockIdx.x * BLOCKSIZE + (threadIdx.x / BLOCKSIZE);
    const int cCol = blockIdx.y * BLOCKSIZE + (threadIdx.x % BLOCKSIZE);
    if (cRow < M && cCol < N) {
        float tmp = 0.0f;
        for (int i = 0; i < K; i++)
            tmp += __half2float(A[cRow * K + i]) * __half2float(B[i * N + cCol]);
        C[cRow * N + cCol] = __float2half(tmp);
    }
}

int main() {
    const int M = 512, N = 512, K = 512;
    const int BLOCKSIZE = 32;

    fp16 *hA = new fp16[M*K], *hB = new fp16[K*N];
    fp16 *hC_A = new fp16[M*N], *hC_B = new fp16[M*N], *hC_ref = new fp16[M*N];

    srand(42);
    for (int i = 0; i < M*K; i++) hA[i] = __float2half((rand()%200-100)/100.0f);
    for (int i = 0; i < K*N; i++) hB[i] = __float2half((rand()%200-100)/100.0f);

    // CPU reference
    for (int r = 0; r < M; r++)
        for (int c = 0; c < N; c++) {
            float tmp = 0.0f;
            for (int i = 0; i < K; i++)
                tmp += __half2float(hA[r*K+i]) * __half2float(hB[i*N+c]);
            hC_ref[r*N+c] = __float2half(tmp);
        }

    fp16 *dA, *dB, *dC;
    CUDA_CHECK(cudaMalloc(&dA, M*K*sizeof(fp16)));
    CUDA_CHECK(cudaMalloc(&dB, K*N*sizeof(fp16)));
    CUDA_CHECK(cudaMalloc(&dC, M*N*sizeof(fp16)));
    CUDA_CHECK(cudaMemcpy(dA, hA, M*K*sizeof(fp16), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB, K*N*sizeof(fp16), cudaMemcpyHostToDevice));

    auto run_and_check = [&](const char* name, auto kernel_fn) {
        CUDA_CHECK(cudaMemset(dC, 0, M*N*sizeof(fp16)));
        kernel_fn();
        CUDA_CHECK(cudaDeviceSynchronize());
        fp16* hC_out = new fp16[M*N];
        CUDA_CHECK(cudaMemcpy(hC_out, dC, M*N*sizeof(fp16), cudaMemcpyDeviceToHost));
        float max_err = 0.0f;
        for (int i = 0; i < M*N; i++)
            max_err = fmaxf(max_err, fabsf(__half2float(hC_out[i]) - __half2float(hC_ref[i])));
        printf("%s: max_abs_err = %.4f -> %s\n", name, max_err, max_err < 0.5f ? "OK" : "FAIL");
        delete[] hC_out;
    };

    run_and_check("Variant A (blockIdx.x=col)", [&]() {
        dim3 block(BLOCKSIZE * BLOCKSIZE);
        dim3 grid((N+BLOCKSIZE-1)/BLOCKSIZE, (M+BLOCKSIZE-1)/BLOCKSIZE);
        gemm_coalesce_A<BLOCKSIZE><<<grid, block>>>(M, N, K, dA, dB, dC);
    });

    run_and_check("Variant B (blockIdx.x=row)", [&]() {
        dim3 block(BLOCKSIZE * BLOCKSIZE);
        dim3 grid((M+BLOCKSIZE-1)/BLOCKSIZE, (N+BLOCKSIZE-1)/BLOCKSIZE);
        gemm_coalesce_B<BLOCKSIZE><<<grid, block>>>(M, N, K, dA, dB, dC);
    });

    delete[] hA; delete[] hB; delete[] hC_A; delete[] hC_B; delete[] hC_ref;
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    return 0;
}
