#include <cuda_runtime.h>
#include <stdio.h>

#define TILE_SIZE 32

// ============================================================================
// KERNEL 1: Naive Transpose (Baseline - UNOPTIMIZED)
//  This has uncoalesced writes - use as performance baseline
// ============================================================================
__global__ void naiveTranspose(float *input, float *output, int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (col < width && row < height) {
        // Read is coalesced (row-major), but WRITE is uncoalesced!
        output[col * height + row] = input[row * width + col];
    }
}

// ============================================================================
// KERNEL 2: Shared Memory Transpose (Incomplete)
// Use shared memory to achieve coalesced reads AND writes
// TODO: Complete the implementation
// ============================================================================
__global__ void sharedMemTranspose(float *input, float *output, int width, int height) {
    // TODO: Declare shared memory tile
    __shared__ float tile[TILE_SIZE][TILE_SIZE];
    
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    // TODO: Load data into shared memory (coalesced read)
    // Each thread loads one element
    if (col < width && row < height) {
        // TODO: Store in tile with correct indexing
        tile[threadIdx.y][threadIdx.x] = input[row * width + col];
    }
    
    __syncthreads();
    
    // TODO: Calculate transposed coordinates for writing
    int transposedCol = row;
    int transposedRow = col;
    
    // TODO: Read from shared memory and write to output (coalesced write)
    if (transposedX < height && transposedY < width) {
      output[transposedY * height + transposedX] = tile[threadIdx.x][threadIdx.y];
    }
}

// KERNEL 3: Bank Conflict-Free Transpose (Challenge)
// Add padding to avoid shared memory bank conflicts
// ============================================================================
__global__ void bankConflictFreeTranspose(float *input, float *output, int width, int height) {
    // TODO: Declare shared memory with padding to avoid bank conflicts
    // Hint: Add one extra column: tile[TILE_SIZE][TILE_SIZE + 1]
    __shared__ float tile[TILE_SIZE][TILE_SIZE + 1];
    
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (col < width && row < height) {
        // TODO: Load with padding consideration
        tile[threadIdx.y][threadIdx.x] = input[row * width + col];
    }
    
    __syncthreads();
    
    // Calculate transposed coordinates
    int transposedX = blockIdx.y * blockDim.x + threadIdx.x;
    int transposedY = blockIdx.x * blockDim.y + threadIdx.y;

    if (transposedX < height && transposedY < width) {
        // Read with padded row index (was column before transpose)
        output[transposedY * height + transposedX] = tile[threadIdx.x][threadIdx.y];
    }
}

// ============================================================================
// KERNEL 4: 1D Block Transpose (Optimization)
// Use 1D block configuration for better occupancy
// ============================================================================
__global__ void transpose1DBlock(float *input, float *output, int width, int height) {
    __shared__ float tile[TILE_SIZE][TILE_SIZE];
    
    int x = blockIdx.x * TILE_SIZE + threadIdx.x;
    int y = blockIdx.y * TILE_SIZE + (threadIdx.x / TILE_SIZE);  // 1D to 2D mapping
    
    // TODO: Complete the implementation
    // Load, sync, transpose, sync, store
}