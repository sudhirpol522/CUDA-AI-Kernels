#include <cuda_runtime.h>
#include <stdio.h>

#define N 100000
#define ARRAY_SIZE 100  // Large array that may cause spilling

// TODO: Complete this kernel that demonstrates local memory usage
// Task:
//   1. Declare a large local array (causes spilling to local memory)
//   2. Initialize the array
//   3. Process the array
//   4. Store result
// 
// WARNING: This kernel intentionally uses local memory (slow!)
// Your goal is to understand the performance impact
__global__ void localMemoryExample(float *input, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= n) return;
    // TODO: Declare a large local array
    // Hint: float localArray[ARRAY_SIZE];
    // This will spill to local memory because it's too large for registers
    float localArray[ARRAY_SIZE];
    
    // TODO: Initialize the array with input values (with bounds checking)
    if (idx < n) {
        localArray[idx] = input[idx];
    }
    
    // TODO: Process the array (e.g., sum all elements)
    float sum = 0.0f;
    for (int i = 0; i < ARRAY_SIZE; i++) {
        sum += localArray[i];
    }
    
    // Store result
    output[idx] = sum;
}
// TODO: Complete this optimized version that reduces local memory usage
// Task: Use registers more efficiently by processing in smaller chunks
__global__ void localMemoryOptimized(float *input, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= n) return;
    
    // TODO: Instead of storing all values, process them on-the-fly
    // Use a single accumulator variable (stays in register)
    float sum = 0.0f;
    
    // Hint: Loop through ARRAY_SIZE iterations but don't store all values
    // Just accumulate the sum directly
    for (int i = 0; i < ARRAY_SIZE; i++) {
        int dataIdx = idx * ARRAY_SIZE + i;
        if (dataIdx < n) {
            sum += input[dataIdx];
        }
    }
    
    output[idx] = sum;
}
// TODO: Complete this kernel that uses dynamic indexing (causes local memory)
__global__ void dynamicIndexingExample(float *input, float *output, int n, int *indices) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= n) return;
    
    // TODO: Declare local array for temporary storage
    float temp[ARRAY_SIZE];
    
    // TODO: Use dynamic indexing (index determined at runtime)
    // This prevents compiler from optimizing to registers
    int dynamicIdx = indices[idx % 10];  // Runtime-determined index
    
    for (int i = 0; i < ARRAY_SIZE; i++) {
        // Dynamic indexing causes local memory usage
        temp[i] = input[idx * ARRAY_SIZE + (i + dynamicIdx) % ARRAY_SIZE];
    }
    
    // TODO: Compute sum and store to output
    float sum = 0.0f;
    for (int i = 0; i < ARRAY_SIZE; i++) {
        sum += temp[i];
    }
    output[idx] = sum;
}
void printArrayStats(float *arr, int n, const char *label) {
    float sum = 0.0f, min = arr[0], max = arr[0];
    for (int i = 0; i < n && i < 100; i++) {
        sum += arr[i];
        if (arr[i] < min) min = arr[i];
        if (arr[i] > max) max = arr[i];
    }
    printf("%s: sum=%.2f, min=%.2f, max=%.2f (first 100 elements)\n", label, sum, min, max);
}

int main() {
    float *h_input, *h_output;
    float *d_input, *d_output;
    int *d_indices;
    
    int dataSize = N * ARRAY_SIZE;
    
    // Allocate host memory
    h_input = (float*)malloc(dataSize * sizeof(float));
    h_output = (float*)malloc(N * sizeof(float));
    int h_indices[10] = {0, 5, 10, 15, 20, 25, 30, 35, 40, 45};
    
    // Initialize
    for (int i = 0; i < dataSize; i++) {
        h_input[i] = 1.0f;
    }
    
    // Allocate device memory
    cudaMalloc(&d_input, dataSize * sizeof(float));
    cudaMalloc(&d_output, N * sizeof(float));
    cudaMalloc(&d_indices, 10 * sizeof(int));
    
    // Copy to device
    cudaMemcpy(d_input, h_input, dataSize * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_indices, h_indices, 10 * sizeof(int), cudaMemcpyHostToDevice);
    
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    
    printf("=== Local Memory Exercise ===\n\n");
    
    // Test local memory version (slow due to spilling)
    printf("1. Running local memory version (with spilling)...\n");
    localMemoryExample<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_output, d_output, N * sizeof(float), cudaMemcpyDeviceToHost);
    printArrayStats(h_output, N, "   Results");
    
    // Test optimized version
    printf("\n2. Running optimized version (register-friendly)...\n");
    localMemoryOptimized<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_output, d_output, N * sizeof(float), cudaMemcpyDeviceToHost);
    printArrayStats(h_output, N, "   Results");
    
    // Test dynamic indexing version
    printf("\n3. Running dynamic indexing version (local memory)...\n");
    dynamicIndexingExample<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, N, d_indices);
    cudaDeviceSynchronize();
    cudaMemcpy(h_output, d_output, N * sizeof(float), cudaMemcpyDeviceToHost);
    printArrayStats(h_output, N, "   Results");
    
    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_indices);
    free(h_input);
    free(h_output);
    
    printf("\n=== Key Takeaways ===\n");
    printf("- Large local arrays spill to local memory (slow!)\n");
    printf("- Dynamic indexing prevents register allocation\n");
    printf("- Process data on-the-fly when possible to stay in registers\n");
    printf("- Use -Xptxas=-v to see register usage and spilling\n");
    
    return 0;
}
