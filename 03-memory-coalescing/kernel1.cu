#include <cuda_runtime.h>
#include <stdio.h>

// ============================================================================
// KERNEL 1: Strided Access Problem
// This kernel has BAD memory access pattern - fix it!
// Current: Thread 0 accesses index 0, Thread 1 accesses index stride, etc.
// Goal: Thread 0 accesses index 0, Thread 1 accesses index 1, etc.
// ============================================================================
__global__ void fixStridedAccess(float *input, float *output, int n, int stride) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // TODO: Fix this strided access pattern
    // Current (BAD): Each thread accesses memory stride apart
    // for (int i = tid * stride; i < n; i += stride) {
    //     output[i] = input[i] * 2.0f;
    // }
    
    // TODO: Implement coalesced access
    // Each thread should process consecutive elements
    // Hint: Use grid-stride loop instead
    for (int i = tid; i < n; i += stride) {
        output[i] = input[i] * 2.0f;
    }
}

// ============================================================================
// KERNEL 2: Column-Major vs Row-Major Access
// Fix the access pattern for row-major stored matrix
// ============================================================================
__global__ void fixColumnAccess(float *matrix, int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    // TODO: This accesses memory in column-major order (BAD for row-major data)
    // Current (BAD): matrix[row][col] accessed as matrix[col][row]
    // Fix: Ensure consecutive threads access consecutive memory
    
    if (row < height && col < width) {
        // TODO: Fix the indexing for row-major storage
        // Row-major: index = row * width + col
        int idx = row * width + col;
        matrix[idx] = matrix[idx] * 2.0f;
    }
}

// ============================================================================
// KERNEL 3: Interleaved Access Pattern
// Fix access pattern for interleaved data (e.g., RGB pixels)
// ============================================================================
__global__ void fixInterleavedAccess(float *data, int n, int channels) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // TODO: This kernel processes one channel at a time with strided access
    // Current (BAD): Processing channel 0: indices 0, channels, 2*channels...
    // Goal: Process all channels for consecutive pixels
    
    // TODO: Restructure to have consecutive threads process consecutive pixels
    // Each thread handles all channels for one or more pixels
    if (tid * channels < n) {
        int pixelIdx = tid;
        for (int c = 0; c < channels; c++) {
            int idx = pixelIdx * channels + c;
            if (idx < n) {
                data[idx] = data[idx] * 1.5f;
            }
        }
    }
}

