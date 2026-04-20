#include <cuda_runtime.h>
#include <stdio.h>

#define N 1000000
#define BLOCK_SIZE 256

// TODO: Complete this kernel using shared memory
// Task: 
//   1. Load data cooperatively from global to shared memory
//   2. Synchronize threads
//   3. Process data from shared memory (multiply by 2)
//   4. Write results back to global memory
__global__ void sharedMemoryExample(float *input, float *output, int n) {
    // TODO: Declare shared memory array
    // Hint: __shared__ float sharedData[BLOCK_SIZE];
    __shared__ float sharedData[BLOCK_SIZE];
    
    // Calculate global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // TODO: Load data from global to shared memory cooperatively
    // Each thread loads one element
    // Don't forget bounds checking!
    if (idx < n) {
        sharedData[threadIdx.x] = input[idx];
    }
    else {
        sharedData[threadIdx.x] = 0.0f;
    }
    __syncthreads();
    
    // TODO: Process data from shared memory
    // Multiply the shared memory value by 2 and store to output
    if (idx < n) {
        output[idx] = sharedData[threadIdx.x] * 2.0f;
    }
}

// Advanced: Multi-pass shared memory usage
// TODO: Complete this kernel that processes data in multiple passes
__global__ void sharedMemoryMultiPass(float *input, float *output, int n, int passes) {
    // TODO: Declare shared memory
    __shared__ float sharedData[BLOCK_SIZE];
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    
    // Load initial data
    if (idx < n) {
        sharedData[tid] = input[idx];
    } else {
        sharedData[tid] = 0.0f;
    }
    __syncthreads();
    
    // TODO: Process for 'passes' number of iterations
    // In each pass:
    //   1. Synchronize threads
    //   2. Each thread adds its left neighbor's value (if exists)
    //   3. Synchronize again before next pass
    // Hint: Be careful of boundary conditions!
    for (int p = 0; p < passes; p++) {
        if (tid > 0) {
            sharedData[tid] += sharedData[tid - 1];
        }
        __syncthreads();
    }
    __syncthreads();
    
    if (idx < n) {
        output[idx] = sharedData[tid];
    }
}
void printFirstElements(float *arr, int n, const char *label) {
    printf("%s: ", label);
    for (int i = 0; i < 10 && i < n; i++) {
        printf("%.2f ", arr[i]);
    }
    printf("\n");
}

int main() {
    printf("=== Memory Hierarchy Level 2: Shared Memory Introduction ===\n\n");

    float *h_input, *h_output;
    float *d_input, *d_output;

    // Allocate and initialize
    h_input = (float*)malloc(N * sizeof(float));
    h_output = (float*)malloc(N * sizeof(float));

    for (int i = 0; i < N; i++) {
        h_input[i] = 1.0f;
    }

    // Allocate device memory
    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_output, N * sizeof(float));

    // Copy to device
    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);

    // Launch configuration
    int threadsPerBlock = BLOCK_SIZE;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Test 1: Basic shared memory
    printf("Test 1: Basic shared memory example\n");
    printf("  Array size: %d elements\n", N);
    printf("  Block size: %d threads\n", threadsPerBlock);
    printf("  Grid size: %d blocks\n\n", blocksPerGrid);

    sharedMemoryExample<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, N);

    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }

    // Copy results back
    cudaMemcpy(h_output, d_output, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Verify (should be 2.0 for all elements)
    printFirstElements(h_output, 10, "Results (expected 2.00)");

    // Test 2: Multi-pass version
    printf("\nTest 2: Multi-pass shared memory (3 passes)\n");
    sharedMemoryMultiPass<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, N, 3);
    cudaMemcpy(h_output, d_output, N * sizeof(float), cudaMemcpyDeviceToHost);
    printFirstElements(h_output, 10, "Multi-pass results");

    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
    free(h_output);

    printf("\n=== Key Takeaways ===\n");
    printf("- Shared memory is declared with __shared__ keyword\n");
    printf("- __syncthreads() ensures all threads reach the barrier\n");
    printf("- Shared memory is much faster than global memory\n");

    return 0;
}
