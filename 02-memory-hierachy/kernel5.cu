#include <cuda_runtime.h>
#include <stdio.h>

#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

#define MATRIX_SIZE 512
#define TILE_SIZE 32

// Constant memory for matrix dimensions
__constant__ int d_matrixSize;

__global__ void optimizedMatrixMultiply(float *A, float *B, float *C, int size) {
    // Shared memory for tiles
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];

    // Calculate row and column of C element to compute
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    // Thread's position within the tile
    int ty = threadIdx.y;
    int tx = threadIdx.x;

    float accumulator = 0.0f;

    // Loop over tiles
    int numTiles = (size + TILE_SIZE - 1) / TILE_SIZE;
    for (int t = 0; t < numTiles; t++) {
        // Load tile of A into shared memory
        // Each thread loads one element cooperatively
        int aCol = t * TILE_SIZE + tx;
        if (row < size && aCol < size) {
            tileA[ty][tx] = A[row * size + aCol];
        } else {
            tileA[ty][tx] = 0.0f;
        }

        // Load tile of B into shared memory
        int bRow = t * TILE_SIZE + ty;
        if (bRow < size && col < size) {
            tileB[ty][tx] = B[bRow * size + col];
        } else {
            tileB[ty][tx] = 0.0f;
        }

        // Synchronize to ensure all data is loaded
        __syncthreads();

        // Compute partial product for this tile
        for (int k = 0; k < TILE_SIZE; k++) {
            accumulator += tileA[ty][k] * tileB[k][tx];
        }

        // Synchronize before loading next tile
        __syncthreads();
    }

    // Write result to global memory (with bounds check)
    if (row < size && col < size) {
        C[row * size + col] = accumulator;
    }
}
// Simple naive matrix multiplication for comparison
__global__ void naiveMatrixMultiply(float *A, float *B, float *C, int size) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < size && col < size) {
        float sum = 0.0f;
        for (int k = 0; k < size; k++) {
            sum += A[row * size + k] * B[k * size + col];
        }
        C[row * size + col] = sum;
    }
}

void initializeMatrix(float *matrix, int size, float seed) {
    for (int i = 0; i < size * size; i++) {
        matrix[i] = (float)(i % 100) / 100.0f + seed;
    }
}

bool verifyMatrix(float *C, float *reference, int size) {
    float tolerance = 1e-3f;
    for (int i = 0; i < size * size; i++) {
        float diff = fabsf(C[i] - reference[i]);
        if (diff > tolerance) {
            printf("Mismatch at (%d, %d): expected %f, got %f\n", 
                   i / size, i % size, reference[i], C[i]);
            return false;
        }
    }
    return true;
}

void printMatrixCorner(float *matrix, int size, const char *label) {
    printf("%s (top-left 5x5 corner):\n", label);
    for (int i = 0; i < 5; i++) {
        printf("  ");
        for (int j = 0; j < 5; j++) {
            printf("%8.4f ", matrix[i * size + j]);
        }
        printf("\n");
    }
}

int main() {
    float *h_A, *h_B, *h_C, *h_C_ref;
    float *d_A, *d_B, *d_C;
    
    int size = MATRIX_SIZE;
    int matrixBytes = size * size * sizeof(float);
    
    // Allocate host memory
    h_A = (float*)malloc(matrixBytes);
    h_B = (float*)malloc(matrixBytes);
    h_C = (float*)malloc(matrixBytes);
    h_C_ref = (float*)malloc(matrixBytes);
    
    // Initialize matrices
    initializeMatrix(h_A, size, 1.0f);
    initializeMatrix(h_B, size, 2.0f);
    
    // Allocate device memory
    cudaMalloc(&d_A, matrixBytes);
    cudaMalloc(&d_B, matrixBytes);
    cudaMalloc(&d_C, matrixBytes);
    
    // Copy to device
    cudaMemcpy(d_A, h_A, matrixBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, matrixBytes, cudaMemcpyHostToDevice);
    
    // TODO: Copy matrix size to constant memory
    // Hint: cudaMemcpyToSymbol(d_matrixSize, &size, sizeof(int));
    /* YOUR CODE HERE */
    
    // Configure blocks and threads
    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 blocksPerGrid((size + TILE_SIZE - 1) / TILE_SIZE, 
                       (size + TILE_SIZE - 1) / TILE_SIZE);
    
    printf("=== Memory Hierarchy Combined Exercise ===\n");
    printf("Matrix size: %d x %d\n", size);
    printf("Tile size: %d x %d\n\n", TILE_SIZE, TILE_SIZE);
    
    // Run naive version for reference
    printf("1. Running naive matrix multiplication...\n");
    naiveMatrixMultiply<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, size);
    cudaDeviceSynchronize();
    cudaMemcpy(h_C_ref, d_C, matrixBytes, cudaMemcpyDeviceToHost);
    printMatrixCorner(h_C_ref, size, "   Reference result");
    
    // Run optimized version
    printf("\n2. Running optimized matrix multiplication (with shared memory)...\n");
    optimizedMatrixMultiply<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, size);
    cudaDeviceSynchronize();
    
    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("   CUDA Error: %s\n", cudaGetErrorString(err));
    }
    
    cudaMemcpy(h_C, d_C, matrixBytes, cudaMemcpyDeviceToHost);
    printMatrixCorner(h_C, size, "   Optimized result");
    
    // Verify results
    printf("\n3. Verifying results...\n");
    if (verifyMatrix(h_C, h_C_ref, size)) {
        printf("   ✓ Results match!\n");
    } else {
        printf("   ✗ Results differ!\n");
    }
    
    // TODO: Optional - Add timing comparison
    // Hint: Use cudaEventRecord to time both kernels
    /* YOUR CODE HERE */
    
    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C_ref);
    
    printf("\n=== Key Takeaways ===\n");
    printf("- Use shared memory for data reused within a block\n");
    printf("- Use constant memory for read-only uniform parameters\n");
    printf("- Keep accumulators in registers\n");
    printf("- Minimize global memory accesses through tiling\n");
    printf("- Proper synchronization is critical with shared memory\n");
    
    return 0;
}
