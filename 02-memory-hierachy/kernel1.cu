#include <cuda_runtime.h>
#include <stdio.h>

#define N 1000000
#define SCALE_FACTOR 2.5f

// TODO: Complete this kernel
// Task: Copy input to output, multiplying each element by scale_factor
// Requirements:
//   1. Calculate global thread ID
//   2. Add bounds check (thread ID < n)
//   3. Perform: output[i] = input[i] * scale_factor
__global__ void globalMemoryTransform(float *input, float *output, int n, float scale_factor) {
    // TODO: Calculate global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // TODO: Add bounds check and perform the transformation
    if (idx < n) {
        output[idx] = input[idx] * scale_factor;
    }
}

__global__ void globalMemoryTransformStrided(float *input, float *output, int n, float scale_factor) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n; i += stride) {
        output[i] = input[i] * scale_factor;
    }
}
void verifyResults(float *output, int n, float expected) {
    bool success = true;
    for (int i = 0; i < n; i++) {
        if (output[i] != expected) {
            printf("Error at index %d: expected %f, got %f\n", i, expected, output[i]);
            success = false;
            break;
        }
    }
    if (success) {
        printf("✓ Verification passed!\n");
    }
}

int main() {
    float *h_input, *h_output;
    float *d_input, *d_output;
    
    // Allocate host memory
    h_input = (float*)malloc(N * sizeof(float));
    h_output = (float*)malloc(N * sizeof(float));
    
    // Initialize input data
    for (int i = 0; i < N; i++) {
        h_input[i] = 1.0f;
    }
    
    // TODO: Allocate device memory using cudaMalloc
    // Hint: cudaMalloc(&pointer, size_in_bytes)
    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_output, N * sizeof(float));
    
    // TODO: Copy input data to device
    // Hint: cudaMemcpy(dest, src, size, cudaMemcpyHostToDevice)
    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);
    
    // Launch configuration
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    
    // Launch kernel
    globalMemoryTransform<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, N, SCALE_FACTOR);
    
    // TODO: Check for kernel launch errors
    // Hint: cudaGetLastError()
    cudaGetLastError();
    
    // TODO: Copy results back to host
    // Hint: cudaMemcpy(dest, src, size, cudaMemcpyDeviceToHost)
    cudaMemcpy(h_output, d_output, N * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Verify results (first 10 elements)
    printf("Checking first 10 results (expected: %f):\n", SCALE_FACTOR);
    for (int i = 0; i < 10; i++) {
        printf("%.2f ", h_output[i]);
    }
    printf("\n");
    
    // TODO: Free device memory
    // Hint: cudaFree(pointer)
    cudaFree(d_input);
    cudaFree(d_output);
    
    // Free host memory
    free(h_input);
    free(h_output);
    
    printf("Global memory exercise completed!\n");
    return 0;
}

