#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

#define MATRIX_SIZE 512
#define TILE_WIDTH 32

// ============================================================================
// KERNEL 1: Tiled Matrix Multiplication (2D block configuration)
// Each thread block cooperatively loads TILE_WIDTH x TILE_WIDTH tiles of A
// and B into shared memory, then computes partial dot products.
// ============================================================================
__global__ void tiledMatMul(float *A, float *B, float *C, int width) {
    __shared__ float tileA[TILE_WIDTH][TILE_WIDTH];
    __shared__ float tileB[TILE_WIDTH][TILE_WIDTH];

    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;

    float accumulator = 0.0f;
    int numTiles = (width + TILE_WIDTH - 1) / TILE_WIDTH;

    for (int t = 0; t < numTiles; t++) {
        // Load tile of A: each thread loads A[row][t*TILE_WIDTH + threadIdx.x]
        int aCol = t * TILE_WIDTH + threadIdx.x;
        tileA[threadIdx.y][threadIdx.x] = (row < width && aCol < width)
                                          ? A[row * width + aCol] : 0.0f;

        // Load tile of B: each thread loads B[t*TILE_WIDTH + threadIdx.y][col]
        int bRow = t * TILE_WIDTH + threadIdx.y;
        tileB[threadIdx.y][threadIdx.x] = (bRow < width && col < width)
                                          ? B[bRow * width + col] : 0.0f;

        __syncthreads();

        // Accumulate partial dot product from this tile
        #pragma unroll
        for (int k = 0; k < TILE_WIDTH; k++) {
            accumulator += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < width && col < width) {
        C[row * width + col] = accumulator;
    }
}

// ============================================================================
// KERNEL 2: Tiled MatMul with 1D block configuration
// Uses a flat 1D block of TILE_WIDTH*TILE_WIDTH threads.
// Each thread maps its 1D index to a 2D tile position.
// Useful for comparing occupancy vs. 2D block layout.
// ============================================================================
__global__ void tiledMatMul1D(float *A, float *B, float *C, int width) {
    __shared__ float tileA[TILE_WIDTH][TILE_WIDTH];
    __shared__ float tileB[TILE_WIDTH][TILE_WIDTH];

    // Map flat thread index to 2D tile coordinates
    int tid  = threadIdx.x;
    int ty   = tid / TILE_WIDTH;   // local row within tile
    int tx   = tid % TILE_WIDTH;   // local col within tile
    int row  = blockIdx.y * TILE_WIDTH + ty;
    int col  = blockIdx.x * TILE_WIDTH + tx;

    float accumulator = 0.0f;
    int numTiles = (width + TILE_WIDTH - 1) / TILE_WIDTH;

    for (int t = 0; t < numTiles; t++) {
        int aCol = t * TILE_WIDTH + tx;
        int bRow = t * TILE_WIDTH + ty;

        tileA[ty][tx] = (row < width && aCol < width) ? A[row * width + aCol] : 0.0f;
        tileB[ty][tx] = (bRow < width && col < width) ? B[bRow * width + col] : 0.0f;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_WIDTH; k++) {
            accumulator += tileA[ty][k] * tileB[k][tx];
        }

        __syncthreads();
    }

    if (row < width && col < width) {
        C[row * width + col] = accumulator;
    }
}

// ============================================================================
// KERNEL 3: Transposed B tile — reduces MIO pressure
// Stores tileB transposed so inner loop reads BOTH tiles as rows:
//   tileA[ty][k]   → broadcast (all threads same ty, same k → 1 MIO op)
//   tileBT[tx][k]  → each thread reads own row → no bank conflicts
// Net result: structured MIO ops instead of scattered column reads
// ============================================================================
__global__ void tiledMatMulOpt(float *A, float *B, float *C, int width) {
    __shared__ float tileA[TILE_WIDTH][TILE_WIDTH];
    __shared__ float tileBT[TILE_WIDTH][TILE_WIDTH + 1]; // +1 padding eliminates bank conflicts

    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;
    int ty  = threadIdx.y;
    int tx  = threadIdx.x;

    float accumulator = 0.0f;
    int numTiles = (width + TILE_WIDTH - 1) / TILE_WIDTH;

    for (int t = 0; t < numTiles; t++) {
        int aCol = t * TILE_WIDTH + tx;
        int bRow = t * TILE_WIDTH + ty;

        // Load A tile normally
        tileA[ty][tx]  = (row < width && aCol < width) ? A[row * width + aCol] : 0.0f;

        // Load B tile TRANSPOSED: B[bRow][col] → tileBT[tx][ty]
        // So tileBT[tx][k] = B[t*TILE_WIDTH + k][col] for any k
        tileBT[tx][ty] = (bRow < width && col < width) ? B[bRow * width + col] : 0.0f;

        __syncthreads();

        // tileA[ty][k]:  all 32 threads share same ty & k → BROADCAST → 1 MIO op
        // tileBT[tx][k]: each thread has unique tx → different banks (+1 pad) → no conflicts
        #pragma unroll
        for (int k = 0; k < TILE_WIDTH; k++) {
            accumulator += tileA[ty][k] * tileBT[tx][k];
        }

        __syncthreads();
    }

    if (row < width && col < width) {
        C[row * width + col] = accumulator;
    }
}

// ============================================================================
// KERNEL 4: Vectorized float4 global loads
//
// Key idea: instead of each thread loading 1 float (4-byte LD), threads
// with tx % 4 == 0 each load a float4 (16-byte LD = 4 floats in 1 instruction).
// This cuts global-load instruction count by 4x per tile row.
//
// Requirements:
//   - width divisible by 4 (guaranteed for MATRIX_SIZE=512)
//   - cudaMalloc guarantees >= 256-byte alignment → float4 always aligned
//
// Trade-off: 3/4 of threads are idle during tile loading (tx%4 != 0).
// The gain is fewer load instructions, not more bytes — bandwidth is unchanged
// since scalar loads were already coalesced.
// ============================================================================
__global__ void tiledMatMulVectorized(float *A, float *B, float *C, int width) {
    __shared__ float tileA[TILE_WIDTH][TILE_WIDTH];
    __shared__ float tileB[TILE_WIDTH][TILE_WIDTH];

    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;
    int ty  = threadIdx.y;
    int tx  = threadIdx.x;

    float accumulator = 0.0f;
    int numTiles = (width + TILE_WIDTH - 1) / TILE_WIDTH;

    for (int t = 0; t < numTiles; t++) {
        int aCol = t * TILE_WIDTH + tx;  // global column for A
        int bRow = t * TILE_WIDTH + ty;  // global row for B

        // ---- Vectorized load of tileA ----
        // Each float4 thread (tx = 0, 4, 8, …, 28) loads 4 consecutive A elements.
        // tx=0  → tileA[ty][0..3],  tx=4 → tileA[ty][4..7], …, tx=28 → tileA[ty][28..31]
        if ((tx & 3) == 0) {
            if (row < width && aCol + 3 < width) {
                // Fast path: fully in-bounds — one 128-bit load
                float4 a4 = *reinterpret_cast<float4*>(&A[row * width + aCol]);
                tileA[ty][tx+0] = a4.x;
                tileA[ty][tx+1] = a4.y;
                tileA[ty][tx+2] = a4.z;
                tileA[ty][tx+3] = a4.w;
            } else {
                // Edge path: near matrix boundary — scalar fallback
                for (int i = 0; i < 4; i++)
                    tileA[ty][tx+i] = (row < width && aCol+i < width)
                                      ? A[row * width + aCol + i] : 0.0f;
            }
        }

        // ---- Vectorized load of tileB ----
        // B[bRow][col..col+3] → tileB[ty][tx..tx+3]
        // col = blockIdx.x*TILE_WIDTH + tx, so col+k covers exactly the tile columns
        if ((tx & 3) == 0) {
            if (bRow < width && col + 3 < width) {
                float4 b4 = *reinterpret_cast<float4*>(&B[bRow * width + col]);
                tileB[ty][tx+0] = b4.x;
                tileB[ty][tx+1] = b4.y;
                tileB[ty][tx+2] = b4.z;
                tileB[ty][tx+3] = b4.w;
            } else {
                for (int i = 0; i < 4; i++)
                    tileB[ty][tx+i] = (bRow < width && col+i < width)
                                      ? B[bRow * width + col + i] : 0.0f;
            }
        }

        __syncthreads();

        // Inner dot product — all threads participate, float4 had no effect here
        #pragma unroll
        for (int k = 0; k < TILE_WIDTH; k++)
            accumulator += tileA[ty][k] * tileB[k][tx];

        __syncthreads();
    }

    if (row < width && col < width)
        C[row * width + col] = accumulator;
}

// ============================================================================
// CPU reference multiply (small matrices only — for correctness check)
// ============================================================================
void cpuMatMul(float *A, float *B, float *C, int width) {
    for (int i = 0; i < width; i++)
        for (int j = 0; j < width; j++) {
            float sum = 0.0f;
            for (int k = 0; k < width; k++)
                sum += A[i * width + k] * B[k * width + j];
            C[i * width + j] = sum;
        }
}

bool verify(float *gpu, float *ref, int width, const char *label) {
    for (int i = 0; i < width * width; i++) {
        if (fabsf(gpu[i] - ref[i]) > 1e-1f) {
            printf("  [%s] FAILED at index %d: got %.4f, expected %.4f\n",
                   label, i, gpu[i], ref[i]);
            return false;
        }
    }
    printf("  [%s] PASSED\n", label);
    return true;
}

void initMatrix(float *M, int width) {
    for (int i = 0; i < width * width; i++)
        M[i] = (float)(i % 13) * 0.1f;   // small values to avoid FP overflow
}

// ============================================================================
// MAIN
// ============================================================================
int main() {
    const int N = MATRIX_SIZE;
    const int VERIFY_N = 64;   // small size for CPU reference check

    // -----------------------------------------------------------------------
    // Correctness verification on small matrix
    // -----------------------------------------------------------------------
    printf("=== Correctness Check (%dx%d) ===\n", VERIFY_N, VERIFY_N);

    size_t vBytes = VERIFY_N * VERIFY_N * sizeof(float);
    float *vA = (float*)malloc(vBytes);
    float *vB = (float*)malloc(vBytes);
    float *vC = (float*)malloc(vBytes);
    float *vRef = (float*)malloc(vBytes);
    float *d_vA, *d_vB, *d_vC;
    cudaMalloc(&d_vA, vBytes);
    cudaMalloc(&d_vB, vBytes);
    cudaMalloc(&d_vC, vBytes);

    initMatrix(vA, VERIFY_N);
    initMatrix(vB, VERIFY_N);
    cpuMatMul(vA, vB, vRef, VERIFY_N);
    cudaMemcpy(d_vA, vA, vBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_vB, vB, vBytes, cudaMemcpyHostToDevice);

    dim3 vBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 vGrid((VERIFY_N + TILE_WIDTH - 1) / TILE_WIDTH,
               (VERIFY_N + TILE_WIDTH - 1) / TILE_WIDTH);

    // Kernel 1 verify
    tiledMatMul<<<vGrid, vBlock>>>(d_vA, d_vB, d_vC, VERIFY_N);
    cudaDeviceSynchronize();
    cudaMemcpy(vC, d_vC, vBytes, cudaMemcpyDeviceToHost);
    verify(vC, vRef, VERIFY_N, "tiledMatMul (2D block)");

    // Kernel 2 verify
    dim3 vBlock1D(TILE_WIDTH * TILE_WIDTH);
    tiledMatMul1D<<<vGrid, vBlock1D>>>(d_vA, d_vB, d_vC, VERIFY_N);
    cudaDeviceSynchronize();
    cudaMemcpy(vC, d_vC, vBytes, cudaMemcpyDeviceToHost);
    verify(vC, vRef, VERIFY_N, "tiledMatMul1D (1D block)");

    // Kernel 3 verify
    tiledMatMulOpt<<<vGrid, vBlock>>>(d_vA, d_vB, d_vC, VERIFY_N);
    cudaDeviceSynchronize();
    cudaMemcpy(vC, d_vC, vBytes, cudaMemcpyDeviceToHost);
    verify(vC, vRef, VERIFY_N, "tiledMatMulOpt (transposed B)");

    // Kernel 4 verify
    tiledMatMulVectorized<<<vGrid, vBlock>>>(d_vA, d_vB, d_vC, VERIFY_N);
    cudaDeviceSynchronize();
    cudaMemcpy(vC, d_vC, vBytes, cudaMemcpyDeviceToHost);
    verify(vC, vRef, VERIFY_N, "tiledMatMulVectorized (float4 loads)");

    cudaFree(d_vA); cudaFree(d_vB); cudaFree(d_vC);
    free(vA); free(vB); free(vC); free(vRef);

    // -----------------------------------------------------------------------
    // Performance comparison on full matrix
    // -----------------------------------------------------------------------
    printf("\n=== Performance Benchmark (%dx%d) ===\n", N, N);

    size_t bytes = (size_t)N * N * sizeof(float);
    float *hA = (float*)malloc(bytes);
    float *hB = (float*)malloc(bytes);
    float *hC = (float*)malloc(bytes);
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    initMatrix(hA, N);
    initMatrix(hB, N);
    cudaMemcpy(d_A, hA, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, hB, bytes, cudaMemcpyHostToDevice);

    dim3 block2D(TILE_WIDTH, TILE_WIDTH);
    dim3 block1D(TILE_WIDTH * TILE_WIDTH);
    dim3 grid((N + TILE_WIDTH - 1) / TILE_WIDTH,
              (N + TILE_WIDTH - 1) / TILE_WIDTH);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float ms = 0.0f;

    // Warm up (not timed — avoids driver init cost skewing results)
    tiledMatMul<<<grid, block2D>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();

    // --- Kernel 1: 2D block ---
    cudaEventRecord(start);
    tiledMatMul<<<grid, block2D>>>(d_A, d_B, d_C, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    float gflops1 = (2.0f * N * N * N) / (ms * 1e6f);
    printf("  tiledMatMul   (2D block %dx%d): %.3f ms  |  %.1f GFLOP/s\n",
           TILE_WIDTH, TILE_WIDTH, ms, gflops1);

    // --- Kernel 2: 1D block ---
    cudaEventRecord(start);
    tiledMatMul1D<<<grid, block1D>>>(d_A, d_B, d_C, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    float gflops2 = (2.0f * N * N * N) / (ms * 1e6f);
    printf("  tiledMatMul1D (1D block %d)  : %.3f ms  |  %.1f GFLOP/s\n",
           TILE_WIDTH * TILE_WIDTH, ms, gflops2);

    // --- Kernel 3: Transposed B ---
    cudaEventRecord(start);
    tiledMatMulOpt<<<grid, block2D>>>(d_A, d_B, d_C, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    float gflops3 = (2.0f * N * N * N) / (ms * 1e6f);
    printf("  tiledMatMulOpt (transposed B): %.3f ms  |  %.1f GFLOP/s\n", ms, gflops3);

    // --- Kernel 4: float4 vectorized loads ---
    cudaEventRecord(start);
    tiledMatMulVectorized<<<grid, block2D>>>(d_A, d_B, d_C, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    float gflops4 = (2.0f * N * N * N) / (ms * 1e6f);
    printf("  tiledMatMulVec  (float4 load): %.3f ms  |  %.1f GFLOP/s\n", ms, gflops4);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(hA); free(hB); free(hC);

    printf("\n=== What to look at in Nsight Compute ===\n");
    printf("  1. SM Throughput       -> how busy the SMs are (target >80%%)\n");
    printf("  2. Memory Throughput   -> L1/L2/DRAM hit rates\n");
    printf("  3. Shared Mem Conflicts-> bank conflicts on tileA/tileB loads\n");
    printf("  4. Warp Efficiency     -> divergent branches reduce this\n");
    printf("  5. Occupancy           -> active warps / max warps per SM\n");
    printf("  6. Compute Roofline    -> are we memory or compute bound?\n");

    return 0;
}
