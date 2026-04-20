#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

#define MATRIX_SIZE 512
#define TILE_WIDTH      16
#define THREAD_TILE      2   // each thread computes 2x2 outputs
#define THREAD_TILE_4    4   // each thread computes 4x4 outputs

// ============================================================================
// KERNEL 0: Naive matmul — no shared memory, reads everything from DRAM
//
// Problem: every inner-loop iteration loads A[row][i] and B[i][col] directly
// from global memory (~600 cycle latency each). No data reuse at all.
//
// Nsight will show:
//   - Long Scoreboard stalls (~40+ cycles) — waiting on DRAM every iteration
//   - DRAM Throughput near 100%
//   - L2 Hit Rate very low (~20%)
//   - ~50-100 GFLOP/s
//
// Fix: load tiles into shared memory so the block reuses data (Kernel 1).
// ============================================================================
__global__ void sgemmNaive(float *A, float *B, float *C, int width) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < width && col < width) {
        float tmp = 0.0f;
        for (int i = 0; i < width; ++i)
            tmp += A[row * width + i] * B[i * width + col];
        C[row * width + col] = tmp;
    }
}

// ============================================================================
// KERNEL 1: Baseline tiled matmul (for comparison)
// ============================================================================
__global__ void baselineMatMul(float *A, float *B, float *C, int width) {
    __shared__ float tileA[TILE_WIDTH][TILE_WIDTH];
    __shared__ float tileB[TILE_WIDTH][TILE_WIDTH];

    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;
    float acc = 0.0f;
    int numTiles = (width + TILE_WIDTH - 1) / TILE_WIDTH;

    for (int t = 0; t < numTiles; t++) {
        tileA[threadIdx.y][threadIdx.x] =
            (row < width && t * TILE_WIDTH + threadIdx.x < width)
            ? A[row * width + t * TILE_WIDTH + threadIdx.x] : 0.0f;
        tileB[threadIdx.y][threadIdx.x] =
            (t * TILE_WIDTH + threadIdx.y < width && col < width)
            ? B[(t * TILE_WIDTH + threadIdx.y) * width + col] : 0.0f;
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < TILE_WIDTH; k++) acc += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        __syncthreads();
    }
    if (row < width && col < width) C[row * width + col] = acc;
}

// ============================================================================
// KERNEL 2: 2x2 Register Tiling
//
// Idea: instead of 1 thread → 1 output, each thread computes a 2x2 block
// of 4 output elements. This doubles arithmetic intensity:
//   - Same shared memory loads
//   - 4x more FMA work per thread
//   → Better FLOP/MIO ratio → less MIO throttle
//
// Layout:
//   Block: dim3(TILE_WIDTH, TILE_WIDTH) = 16x16 = 256 threads
//   Each thread computes 2x2 = 4 outputs
//   Each block covers (16*2) x (16*2) = 32x32 output tile
//
// *** BUG FIX in original code ***
//   col = blockIdx.x * TILE_WIDTH + threadIdx.x           ← WRONG
//   col = blockIdx.x * (TILE_WIDTH*THREAD_TILE)           ← CORRECT
//         + threadIdx.x * THREAD_TILE
//   Without this fix, threads overlap their output columns.
// ============================================================================
__global__ void registerTiledMatMul2x2(float *A, float *B, float *C, int width) {
    __shared__ float tileA[TILE_WIDTH * THREAD_TILE][TILE_WIDTH];       // [32][16]
    __shared__ float tileB[TILE_WIDTH][TILE_WIDTH * THREAD_TILE];       // [16][32]

    // Starting row/col of this thread's 2x2 output block
    int row = blockIdx.y * (TILE_WIDTH * THREAD_TILE) + threadIdx.y * THREAD_TILE;
    int col = blockIdx.x * (TILE_WIDTH * THREAD_TILE) + threadIdx.x * THREAD_TILE; // FIXED

    float accum[THREAD_TILE][THREAD_TILE] = {0.0f};  // 2x2 accumulators — stay in registers
    int numTiles = (width + TILE_WIDTH - 1) / TILE_WIDTH;

    for (int t = 0; t < numTiles; t++) {
        int aCol = t * TILE_WIDTH + threadIdx.x;    // A column this thread loads
        int bRow = t * TILE_WIDTH + threadIdx.y;    // B row this thread loads

        // ---- Load tileA ----
        // Each thread (ty, tx) loads 2 rows of A into tileA[ty*2+i][tx]
        // With ty=0..15 and i=0,1: all 32 rows covered
        // With tx=0..15: all 16 columns covered
        for (int i = 0; i < THREAD_TILE; i++) {
            int loadRow = row + i;
            tileA[threadIdx.y * THREAD_TILE + i][threadIdx.x] =
                (loadRow < width && aCol < width) ? A[loadRow * width + aCol] : 0.0f;
        }

        // ---- Load tileB ----
        // Each thread (ty, tx) loads 1 row, 2 columns of B into tileB[ty][tx*2+j]
        // With ty=0..15: all 16 rows covered (bRow = t*16+ty)
        // With tx=0..15 and j=0,1: all 32 columns covered (col+j = bx*32+tx*2+j)
        for (int j = 0; j < THREAD_TILE; j++) {
            tileB[threadIdx.y][threadIdx.x * THREAD_TILE + j] =
                (bRow < width && col + j < width) ? B[bRow * width + col + j] : 0.0f;
        }

        __syncthreads();

        // ---- Compute 2x2 output block ----
        // tileA[ty*2+i][k]: A sub-matrix rows
        // tileB[k][tx*2+j]: B sub-matrix columns
        // This loop does 4x more FMAs than baseline for the same tile loads
        #pragma unroll
        for (int k = 0; k < TILE_WIDTH; k++) {
            for (int i = 0; i < THREAD_TILE; i++) {
                for (int j = 0; j < THREAD_TILE; j++) {
                    accum[i][j] += tileA[threadIdx.y * THREAD_TILE + i][k] *
                                   tileB[k][threadIdx.x * THREAD_TILE + j];
                }
            }
        }

        __syncthreads();
    }

    // ---- Store 2x2 output block ----
    for (int i = 0; i < THREAD_TILE; i++) {
        for (int j = 0; j < THREAD_TILE; j++) {
            int storeRow = row + i;
            int storeCol = col + j;
            if (storeRow < width && storeCol < width)
                C[storeRow * width + storeCol] = accum[i][j];
        }
    }
}

// ============================================================================
// KERNEL 3: 4x4 Register Tiling
//
// Same idea as 2x2, but each thread now computes a 4x4 = 16 output block.
//
// FMA/MIO ratio analysis:
//   per k step in inner loop:
//     MIO reads: tileA[ty*4+i][k] for i=0..3  → 4 reads
//                tileB[k][tx*4+j] for j=0..3  → 4 reads   = 8 total MIO ops
//     FMAs:      4×4 = 16
//     FMA/MIO = 16/8 = 2.0   (vs 1.0 for 2x2, vs 0.5 for baseline)
//
// Layout:
//   Block: 16×16 = 256 threads
//   Each block covers (16×4) × (16×4) = 64×64 output tile
//   Grid:  (512/64) × (512/64) = 8×8 = 64 blocks  (vs 256 for 2x2)
//
// Shared memory per block:
//   tileA [64][16] = 4096 bytes
//   tileB [16][64] = 4096 bytes
//   Total = 8 KB  (32 KB available → 4 blocks/SM → 100% occupancy)
// ============================================================================
__global__ void registerTiledMatMul4x4(float *A, float *B, float *C, int width) {
    __shared__ float tileA[TILE_WIDTH * THREAD_TILE_4][TILE_WIDTH];      // [64][16]
    __shared__ float tileB[TILE_WIDTH][TILE_WIDTH * THREAD_TILE_4];      // [16][64]

    // Starting row/col of this thread's 4×4 output block
    int row = blockIdx.y * (TILE_WIDTH * THREAD_TILE_4) + threadIdx.y * THREAD_TILE_4;
    int col = blockIdx.x * (TILE_WIDTH * THREAD_TILE_4) + threadIdx.x * THREAD_TILE_4;

    float accum[THREAD_TILE_4][THREAD_TILE_4] = {0.0f};  // 16 accumulators in registers
    int numTiles = (width + TILE_WIDTH - 1) / TILE_WIDTH;

    for (int t = 0; t < numTiles; t++) {
        int aCol = t * TILE_WIDTH + threadIdx.x;
        int bRow = t * TILE_WIDTH + threadIdx.y;

        // ---- Load tileA: each thread loads 4 rows, 1 column of the A tile ----
        // ty=0..15, i=0..3 → rows ty*4+0 .. ty*4+3: covers all 64 rows
        // tx=0..15: covers all 16 columns
        for (int i = 0; i < THREAD_TILE_4; i++) {
            int loadRow = row + i;
            tileA[threadIdx.y * THREAD_TILE_4 + i][threadIdx.x] =
                (loadRow < width && aCol < width) ? A[loadRow * width + aCol] : 0.0f;
        }

        // ---- Load tileB: each thread loads 1 row, 4 columns of the B tile ----
        // ty=0..15: covers all 16 rows (bRow = t*16+ty)
        // tx=0..15, j=0..3 → cols tx*4+0 .. tx*4+3: covers all 64 columns
        for (int j = 0; j < THREAD_TILE_4; j++) {
            tileB[threadIdx.y][threadIdx.x * THREAD_TILE_4 + j] =
                (bRow < width && col + j < width) ? B[bRow * width + col + j] : 0.0f;
        }

        __syncthreads();

        // ---- Compute 4×4 output block ----
        // Each k step: 8 shared mem reads, 16 FMAs → FMA/MIO = 2.0
        #pragma unroll
        for (int k = 0; k < TILE_WIDTH; k++) {
            for (int i = 0; i < THREAD_TILE_4; i++) {
                for (int j = 0; j < THREAD_TILE_4; j++) {
                    accum[i][j] += tileA[threadIdx.y * THREAD_TILE_4 + i][k] *
                                   tileB[k][threadIdx.x * THREAD_TILE_4 + j];
                }
            }
        }

        __syncthreads();
    }

    // ---- Store 4×4 output block ----
    for (int i = 0; i < THREAD_TILE_4; i++) {
        for (int j = 0; j < THREAD_TILE_4; j++) {
            int storeRow = row + i;
            int storeCol = col + j;
            if (storeRow < width && storeCol < width)
                C[storeRow * width + storeCol] = accum[i][j];
        }
    }
}

// ============================================================================
// KERNEL 4: 1D Thread Block with Register Tiling
//
// Functionally identical to registerTiledMatMul2x2, but the block is declared
// as dim3(256) instead of dim3(16,16). The hardware sees no difference — CUDA
// linearizes 2D blocks the same way. This version is useful for understanding
// that threadIdx.y/x are just a convenience; the real linear index is what
// the scheduler uses.
//
// Thread mapping (1D → logical 2D):
//   ty = tid / TILE_WIDTH   (0..15) — logical row within block
//   tx = tid % TILE_WIDTH   (0..15) — logical col within block
//
// *** BUG in skeleton col formula (same class of bug as 2x2 had) ***
//   WRONG:   col = blockIdx.x * TILE_WIDTH + tx
//            → tx=0 stores cols {0,1}, tx=1 stores cols {1,2} — OVERLAP!
//   CORRECT: col = blockIdx.x * (TILE_WIDTH * RTILE) + tx * RTILE
//            → tx=0 stores cols {0,1}, tx=1 stores cols {2,3} — correct
//
// Tile loading (256 threads, 512 elements per tile — each thread loads 2):
//   tileA [32][16]: thread(ty,tx) loads rows ty*2+0 and ty*2+1 at column tx
//   tileB [16][32]: thread(ty,tx) loads row ty at columns tx*2+0 and tx*2+1
// ============================================================================
__global__ void registerTiledMatMul1D(float *A, float *B, float *C, int width) {
    const int RTILE = 2;

    __shared__ float tileA[TILE_WIDTH * RTILE][TILE_WIDTH];   // [32][16]
    __shared__ float tileB[TILE_WIDTH][TILE_WIDTH * RTILE];   // [16][32]

    int tid = threadIdx.x;
    int ty  = tid / TILE_WIDTH;   // logical row index: 0..15
    int tx  = tid % TILE_WIDTH;   // logical col index: 0..15

    // Starting row/col of this thread's 2x2 output block
    // BUG FIX: col must stride by RTILE so threads don't overlap output columns
    int row = blockIdx.y * (TILE_WIDTH * RTILE) + ty * RTILE;
    int col = blockIdx.x * (TILE_WIDTH * RTILE) + tx * RTILE; // FIXED

    float accum[RTILE][RTILE] = {0.0f};
    int numTiles = (width + TILE_WIDTH - 1) / TILE_WIDTH;

    for (int t = 0; t < numTiles; t++) {
        int aCol = t * TILE_WIDTH + tx;   // global A column this thread loads
        int bRow = t * TILE_WIDTH + ty;   // global B row this thread loads

        // ---- Load tileA ----
        // Thread (ty, tx) loads 2 rows of A into tileA[ty*2+i][tx].
        // ty=0..15, i=0,1 → covers all 32 rows of tileA.
        // tx=0..15         → covers all 16 columns.
        // Each warp (same ty, tx=0..15): loads A[row+i][aCol+0..15]
        // → 16 consecutive floats → coalesced (half-warp granularity).
        for (int i = 0; i < RTILE; i++) {
            int loadRow = row + i;
            tileA[ty * RTILE + i][tx] =
                (loadRow < width && aCol < width) ? A[loadRow * width + aCol] : 0.0f;
        }

        // ---- Load tileB ----
        // Thread (ty, tx) loads 1 row, 2 columns of B into tileB[ty][tx*2+j].
        // ty=0..15         → covers all 16 rows of tileB (bRow = t*16+ty).
        // tx=0..15, j=0,1  → covers all 32 columns of tileB (tx*2+j = 0..31).
        for (int j = 0; j < RTILE; j++) {
            int bCol = col + j;   // col = blockIdx.x*32 + tx*2, so bCol = tx*2+j in tile
            tileB[ty][tx * RTILE + j] =
                (bRow < width && bCol < width) ? B[bRow * width + bCol] : 0.0f;
        }

        __syncthreads();

        // ---- Compute 2x2 output block ----
        // Same FMA/MIO ratio as 2x2 kernel: 4 MIO reads → 4 FMAs per k step = 1.0
        #pragma unroll
        for (int k = 0; k < TILE_WIDTH; k++) {
            for (int i = 0; i < RTILE; i++) {
                for (int j = 0; j < RTILE; j++) {
                    accum[i][j] += tileA[ty * RTILE + i][k] *
                                   tileB[k][tx * RTILE + j];
                }
            }
        }

        __syncthreads();
    }

    // ---- Store 2x2 output block ----
    for (int i = 0; i < RTILE; i++) {
        for (int j = 0; j < RTILE; j++) {
            int storeRow = row + i;
            int storeCol = col + j;
            if (storeRow < width && storeCol < width)
                C[storeRow * width + storeCol] = accum[i][j];
        }
    }
}

// ============================================================================
// KERNEL 5: Double-Buffered Register Tiling
//
// Problem being solved: Barrier stall.
//   Standard tiling: load → __sync → compute → __sync → load → __sync ...
//   The __sync after the load stalls all threads until the SLOWEST thread's
//   global memory load arrives (~600 cycles). Compute sits idle.
//
// Solution — double buffering:
//   - Keep TWO copies of shared memory (ping and pong buffers)
//   - PROLOGUE: load tile 0 into buffer[0], sync
//   - LOOP:     issue loads for tile t+1 into buffer[nxt]   ← memory unit busy
//               compute with buffer[cur]                     ← ALU busy
//               __sync  ← by now tile t+1 is already loaded (hidden behind compute)
//   - No dependency between nxt loads and cur compute → hardware overlaps them
//
// Why it reduces Barrier stalls in Nsight:
//   __sync stall = (load latency) - (compute time)
//   If compute time ≥ load latency → __sync stall ≈ 0
//
// Shared memory cost: 2× vs single-buffer (8 KB total, still 4 blocks/SM)
//   tileA [2][32][16] = 4096 bytes
//   tileB [2][16][32] = 4096 bytes
//   Total = 8 KB (vs 4 KB for 2×2) — still 4 blocks per SM on GTX 1660Ti
//
// *** BUG FIX (same col overlap bug as always) ***
//   col must include RTILE spacing:
//   col = blockIdx.x * (TILE_WIDTH * RTILE) + threadIdx.x * RTILE
// ============================================================================
__global__ void registerTiledMatMulDoubleBuffer(float *A, float *B, float *C, int width) {
    const int RTILE = 2;

    __shared__ float tileA[2][TILE_WIDTH * RTILE][TILE_WIDTH];   // [2][32][16] = 4 KB
    __shared__ float tileB[2][TILE_WIDTH][TILE_WIDTH * RTILE];   // [2][16][32] = 4 KB

    int ty = threadIdx.y;
    int tx = threadIdx.x;

    // BUG FIX: col must stride by RTILE so each thread's 2×2 outputs don't overlap
    int row = blockIdx.y * (TILE_WIDTH * RTILE) + ty * RTILE;
    int col = blockIdx.x * (TILE_WIDTH * RTILE) + tx * RTILE;   // FIXED

    float accum[RTILE][RTILE] = {0.0f};
    int numTiles = (width + TILE_WIDTH - 1) / TILE_WIDTH;

    // =========================================================================
    // PROLOGUE: Load tile 0 into buffer[0] and sync.
    // This primes the pipeline — every subsequent iteration can immediately
    // compute with the already-loaded buffer while prefetching the next tile.
    // =========================================================================
    {
        int aCol = tx;        // t=0: A column = 0*TILE_WIDTH + tx
        int bRow = ty;        // t=0: B row    = 0*TILE_WIDTH + ty
        for (int i = 0; i < RTILE; i++) {
            int loadRow = row + i;
            tileA[0][ty * RTILE + i][tx] =
                (loadRow < width && aCol < width) ? A[loadRow * width + aCol] : 0.0f;
        }
        for (int j = 0; j < RTILE; j++) {
            int bCol = col + j;
            tileB[0][ty][tx * RTILE + j] =
                (bRow < width && bCol < width) ? B[bRow * width + bCol] : 0.0f;
        }
    }
    __syncthreads();

    // =========================================================================
    // MAIN LOOP — true software double buffering
    //
    // Structure per iteration:
    //   1. PREFETCH: issue global→shared loads for tile t+1 into buffer[nxt]
    //      No __sync here — loads go into the memory pipeline immediately
    //   2. COMPUTE: use buffer[cur] for FMAs
    //      No dependency on buffer[nxt] — hardware can overlap with step 1
    //   3. __sync: ensures prefetch is complete before next iteration uses nxt
    //      If compute time ≥ load latency, this sync costs 0 cycles (no stall)
    // =========================================================================
    for (int t = 0; t < numTiles; t++) {
        int cur = t % 2;         // buffer holding tile t   (already loaded)
        int nxt = (t + 1) % 2;  // buffer to receive tile t+1

        // ---- PREFETCH tile t+1 into buffer[nxt] ----
        // Issued BEFORE compute so the memory system gets maximum lead time.
        // Writes to tileA[nxt] / tileB[nxt], reads from tileA[cur] / tileB[cur]
        // → zero data hazard, hardware can pipeline freely.
        if (t + 1 < numTiles) {
            int aCol = (t + 1) * TILE_WIDTH + tx;
            int bRow = (t + 1) * TILE_WIDTH + ty;
            for (int i = 0; i < RTILE; i++) {
                int loadRow = row + i;
                tileA[nxt][ty * RTILE + i][tx] =
                    (loadRow < width && aCol < width) ? A[loadRow * width + aCol] : 0.0f;
            }
            for (int j = 0; j < RTILE; j++) {
                int bCol = col + j;
                tileB[nxt][ty][tx * RTILE + j] =
                    (bRow < width && bCol < width) ? B[bRow * width + bCol] : 0.0f;
            }
        }

        // ---- COMPUTE with buffer[cur] ----
        // These FMAs run while the prefetch loads above are in-flight.
        // FMA/MIO ratio = 1.0 (same as 2×2) — MIO here is from shared mem reads.
        #pragma unroll
        for (int k = 0; k < TILE_WIDTH; k++) {
            for (int i = 0; i < RTILE; i++) {
                for (int j = 0; j < RTILE; j++) {
                    accum[i][j] += tileA[cur][ty * RTILE + i][k] *
                                   tileB[cur][k][tx * RTILE + j];
                }
            }
        }

        // ---- SYNC: ensure prefetch is complete before next iter uses nxt ----
        // If compute took longer than the DRAM fetch (~600 cycles for L2 miss,
        // ~200 cycles for L2 hit), this sync completes instantly = 0 barrier stall.
        __syncthreads();
    }

    // ---- Store 2×2 output block ----
    for (int i = 0; i < RTILE; i++) {
        for (int j = 0; j < RTILE; j++) {
            int storeRow = row + i;
            int storeCol = col + j;
            if (storeRow < width && storeCol < width)
                C[storeRow * width + storeCol] = accum[i][j];
        }
    }
}

// ============================================================================
// CPU reference and utilities
// ============================================================================
void cpuMatMul(float *A, float *B, float *C, int n) {
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            float s = 0.0f;
            for (int k = 0; k < n; k++) s += A[i*n+k] * B[k*n+j];
            C[i*n+j] = s;
        }
}

bool verify(float *gpu, float *ref, int n, const char *label) {
    for (int i = 0; i < n*n; i++) {
        if (fabsf(gpu[i] - ref[i]) > 1e-1f) {
            printf("  [%s] FAILED at [%d]: got %.4f expected %.4f\n",
                   label, i, gpu[i], ref[i]);
            return false;
        }
    }
    printf("  [%s] PASSED\n", label);
    return true;
}

void initMatrix(float *M, int n) {
    for (int i = 0; i < n*n; i++) M[i] = (float)(i % 13) * 0.1f;
}

// ============================================================================
// MAIN
// ============================================================================
int main() {
    const int N = 512;
    const int PEAK_GFLOPS = 4027;
    size_t bytes = (size_t)N * N * sizeof(float);

    // =========================================================================
    // Correctness check at N=64 (CPU verifies all 6 kernels)
    // In Nsight: these are IDs 0-5  (tiny duration ~4-8 us, ignore for profiling)
    // =========================================================================
    printf("=== Correctness Check (64x64) ===\n");
    const int VN = 64;
    size_t vbytes = VN * VN * sizeof(float);
    float *vA = (float*)malloc(vbytes), *vB = (float*)malloc(vbytes);
    float *vC = (float*)malloc(vbytes), *vRef = (float*)malloc(vbytes);
    float *d_vA, *d_vB, *d_vC;
    cudaMalloc(&d_vA, vbytes); cudaMalloc(&d_vB, vbytes); cudaMalloc(&d_vC, vbytes);
    initMatrix(vA, VN); initMatrix(vB, VN);
    cpuMatMul(vA, vB, vRef, VN);
    cudaMemcpy(d_vA, vA, vbytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_vB, vB, vbytes, cudaMemcpyHostToDevice);

    dim3 vBlkNaive(16, 16);
    dim3 vBlk2D(TILE_WIDTH, TILE_WIDTH);
    dim3 vBlk1D(TILE_WIDTH * TILE_WIDTH);
    dim3 vGrd_naive((VN + 15) / 16, (VN + 15) / 16);
    dim3 vGrd_base((VN + TILE_WIDTH - 1) / TILE_WIDTH,
                   (VN + TILE_WIDTH - 1) / TILE_WIDTH);
    dim3 vGrd_reg2((VN + TILE_WIDTH*THREAD_TILE - 1) / (TILE_WIDTH*THREAD_TILE),
                   (VN + TILE_WIDTH*THREAD_TILE - 1) / (TILE_WIDTH*THREAD_TILE));
    dim3 vGrd_reg4((VN + TILE_WIDTH*THREAD_TILE_4 - 1) / (TILE_WIDTH*THREAD_TILE_4),
                   (VN + TILE_WIDTH*THREAD_TILE_4 - 1) / (TILE_WIDTH*THREAD_TILE_4));

    sgemmNaive<<<vGrd_naive, vBlkNaive>>>(d_vA, d_vB, d_vC, VN);
    cudaDeviceSynchronize();
    cudaMemcpy(vC, d_vC, vbytes, cudaMemcpyDeviceToHost);
    verify(vC, vRef, VN, "sgemmNaive");

    baselineMatMul<<<vGrd_base, vBlk2D>>>(d_vA, d_vB, d_vC, VN);
    cudaDeviceSynchronize();
    cudaMemcpy(vC, d_vC, vbytes, cudaMemcpyDeviceToHost);
    verify(vC, vRef, VN, "baselineMatMul");

    registerTiledMatMul2x2<<<vGrd_reg2, vBlk2D>>>(d_vA, d_vB, d_vC, VN);
    cudaDeviceSynchronize();
    cudaMemcpy(vC, d_vC, vbytes, cudaMemcpyDeviceToHost);
    verify(vC, vRef, VN, "registerTiledMatMul2x2");

    registerTiledMatMul4x4<<<vGrd_reg4, vBlk2D>>>(d_vA, d_vB, d_vC, VN);
    cudaDeviceSynchronize();
    cudaMemcpy(vC, d_vC, vbytes, cudaMemcpyDeviceToHost);
    verify(vC, vRef, VN, "registerTiledMatMul4x4");

    registerTiledMatMul1D<<<vGrd_reg2, vBlk1D>>>(d_vA, d_vB, d_vC, VN);
    cudaDeviceSynchronize();
    cudaMemcpy(vC, d_vC, vbytes, cudaMemcpyDeviceToHost);
    verify(vC, vRef, VN, "registerTiledMatMul1D");

    registerTiledMatMulDoubleBuffer<<<vGrd_reg2, vBlk2D>>>(d_vA, d_vB, d_vC, VN);
    cudaDeviceSynchronize();
    cudaMemcpy(vC, d_vC, vbytes, cudaMemcpyDeviceToHost);
    verify(vC, vRef, VN, "registerTiledMatMulDoubleBuffer");

    cudaFree(d_vA); cudaFree(d_vB); cudaFree(d_vC);
    free(vA); free(vB); free(vC); free(vRef);

    // =========================================================================
    // Benchmark — single pass at N=512, NO warmup
    // In Nsight: IDs 6-11, one per kernel — clean, easy to compare
    //
    //  ID 6  sgemmNaive            ← Long Scoreboard stall (DRAM latency)
    //  ID 7  baselineMatMul        ← MIO Throttle ~25 cycles
    //  ID 8  registerTiledMatMul2x2← MIO Throttle ~19 cycles
    //  ID 9  registerTiledMatMul4x4← lower MIO / register pressure
    //  ID 10 registerTiledMatMul1D ← same as 2x2 (different block shape)
    //  ID 11 registerTiledMatMulDoubleBuffer ← Barrier stall reduced
    // =========================================================================
    printf("\n=== Benchmark N=%d (no warmup — each kernel appears once in Nsight) ===\n", N);
    printf("  %-28s  %10s  %s\n", "Kernel", "GFLOP/s", "Nsight: top stall to look for");
    printf("  %-28s  %10s  %s\n",
           "----------------------------", "----------", "------------------------------");

    float *hA = (float*)malloc(bytes), *hB = (float*)malloc(bytes);
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes); cudaMalloc(&d_B, bytes); cudaMalloc(&d_C, bytes);
    initMatrix(hA, N); initMatrix(hB, N);
    cudaMemcpy(d_A, hA, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, hB, bytes, cudaMemcpyHostToDevice);
    free(hA); free(hB);

    dim3 blkNaive(16, 16);
    dim3 blk2D(TILE_WIDTH, TILE_WIDTH);
    dim3 blk1D(TILE_WIDTH * TILE_WIDTH);
    dim3 grd_naive((N + 15) / 16, (N + 15) / 16);
    dim3 grd_base((N + TILE_WIDTH - 1) / TILE_WIDTH,
                  (N + TILE_WIDTH - 1) / TILE_WIDTH);
    dim3 grd_reg2((N + TILE_WIDTH*THREAD_TILE - 1)   / (TILE_WIDTH*THREAD_TILE),
                  (N + TILE_WIDTH*THREAD_TILE - 1)   / (TILE_WIDTH*THREAD_TILE));
    dim3 grd_reg4((N + TILE_WIDTH*THREAD_TILE_4 - 1) / (TILE_WIDTH*THREAD_TILE_4),
                  (N + TILE_WIDTH*THREAD_TILE_4 - 1) / (TILE_WIDTH*THREAD_TILE_4));

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    float ms = 0.0f;
    float gflops = 2.0f * N * N * N;

    // ID 6 in Nsight
    cudaEventRecord(start);
    sgemmNaive<<<grd_naive, blkNaive>>>(d_A, d_B, d_C, N);
    cudaEventRecord(stop); cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    printf("  %-28s  %10.0f  Long Scoreboard (DRAM every iter)\n",
           "sgemmNaive", gflops/(ms*1e6f));

    // ID 7 in Nsight
    cudaEventRecord(start);
    baselineMatMul<<<grd_base, blk2D>>>(d_A, d_B, d_C, N);
    cudaEventRecord(stop); cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    printf("  %-28s  %10.0f  MIO Throttle ~25 cyc  FMA/MIO=0.5\n",
           "baselineMatMul", gflops/(ms*1e6f));

    // ID 8 in Nsight
    cudaEventRecord(start);
    registerTiledMatMul2x2<<<grd_reg2, blk2D>>>(d_A, d_B, d_C, N);
    cudaEventRecord(stop); cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    printf("  %-28s  %10.0f  MIO Throttle ~19 cyc  FMA/MIO=1.0\n",
           "registerTiledMatMul2x2", gflops/(ms*1e6f));

    // ID 9 in Nsight
    cudaEventRecord(start);
    registerTiledMatMul4x4<<<grd_reg4, blk2D>>>(d_A, d_B, d_C, N);
    cudaEventRecord(stop); cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    printf("  %-28s  %10.0f  MIO Throttle lower    FMA/MIO=2.0\n",
           "registerTiledMatMul4x4", gflops/(ms*1e6f));

    // ID 10 in Nsight
    cudaEventRecord(start);
    registerTiledMatMul1D<<<grd_reg2, blk1D>>>(d_A, d_B, d_C, N);
    cudaEventRecord(stop); cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    printf("  %-28s  %10.0f  MIO Throttle ~19 cyc  (same as 2x2)\n",
           "registerTiledMatMul1D", gflops/(ms*1e6f));

    // ID 11 in Nsight
    cudaEventRecord(start);
    registerTiledMatMulDoubleBuffer<<<grd_reg2, blk2D>>>(d_A, d_B, d_C, N);
    cudaEventRecord(stop); cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    printf("  %-28s  %10.0f  Barrier stall reduced (double buf)\n",
           "registerTiledDoubleBuffer", gflops/(ms*1e6f));

    printf("\n  Peak = %d GFLOP/s (GTX 1660Ti). cuBLAS reaches ~85%%.\n", PEAK_GFLOPS);

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    return 0;
}
