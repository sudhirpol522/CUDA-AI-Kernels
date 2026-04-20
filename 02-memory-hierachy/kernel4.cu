#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

#define N 1000000
#define COEFF_COUNT 10
#define TABLE_SIZE 256
#define M_PI 3.14159265358979323846

// TODO: Declare constant memory for coefficients
// Hint: __constant__ float d_coeffs[COEFF_COUNT];
__constant__ float d_coeffs[COEFF_COUNT];


// Constant memory for sine lookup table
__constant__ float sinTable[TABLE_SIZE];

// TODO: Complete this kernel using constant memory for coefficients
// Task:
//   1. Apply polynomial transformation using constant memory coefficients
//   2. Formula: output[i] = sum(coeffs[j] * pow(input[i], j)) for j = 0 to COEFF_COUNT-1
//   3. All threads read the same coefficients (broadcast optimization)
__global__ void constantMemoryTransform(float *input, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= n) return;
    
    float x = input[idx];
    float result = 0.0f;
    
    // TODO: Loop through coefficients in constant memory
    // Apply polynomial: result += d_coeffs[j] * pow(x, j)
    for (int j = 0; j < COEFF_COUNT; j++) {
        result += d_coeffs[j] * powf(x, j);
    }
    
    output[idx] = result;
}
// Compare with global memory version
__global__ void globalMemoryTransform(float *input, float *output, int n, float *coeffs) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= n) return;
    
    float x = input[idx];
    float result = 0.0f;
    
    // Same computation but coefficients in global memory
    for (int j = 0; j < COEFF_COUNT; j++) {
        result += coeffs[j] * powf(x, j);
    }
    
    output[idx] = result;
}

// TODO: Complete this kernel that uses constant memory for a lookup table
// Task: Implement a sine approximation using constant memory lookup table
__global__ void constantMemoryLookup(float *input, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= n) return;
    
    // TODO: Assume constant memory has a precomputed sine table
    // Access the table based on input value (normalized to 0-255)
    // Hint: int tableIdx = (int)(input[idx] * 255.0f) % 256;
    //       output[idx] = sinTable[tableIdx];
    int tableIdx = (int)(input[idx] * 255.0f) % 256;
    output[idx] = sinTable[tableIdx];
}
void initializeCoefficients(float *coeffs) {
    // Initialize polynomial coefficients
    for (int i = 0; i < COEFF_COUNT; i++) {
        coeffs[i] = 1.0f / (i + 1);  // 1, 1/2, 1/3, 1/4, ...
    }
}

void initializeSinTable(float *table) {
    for (int i = 0; i < 256; i++) {
        table[i] = sinf(2.0f * M_PI * i / 256.0f);
    }
}

void printFirstElements(float *arr, int n, const char *label) {
    printf("%s: ", label);
    for (int i = 0; i < 5 && i < n; i++) {
        printf("%.4f ", arr[i]);
    }
    printf("\n");
}

int main() {
    float *h_input, *h_output;
    float *d_input, *d_output;
    float *d_coeffs_global;
    float h_coeffs[COEFF_COUNT];
    float h_sinTable[TABLE_SIZE];
    
    // Allocate host memory
    h_input = (float*)malloc(N * sizeof(float));
    h_output = (float*)malloc(N * sizeof(float));
    
    // Initialize input data (values between 0 and 1)
    for (int i = 0; i < N; i++) {
        h_input[i] = (float)(i % 100) / 100.0f;
    }
    
    // Initialize coefficients
    initializeCoefficients(h_coeffs);
     // Initialize sine lookup table
     initializeSinTable(h_sinTable);
    
    // Allocate device memory
    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_output, N * sizeof(float));
    cudaMalloc(&d_coeffs_global, COEFF_COUNT * sizeof(float));
    
    // Copy to device
    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_coeffs_global, h_coeffs, COEFF_COUNT * sizeof(float), cudaMemcpyHostToDevice);
    
    // TODO: Copy coefficients to constant memory
    // Hint: cudaMemcpyToSymbol(symbol_name, source, size)
    // Example: cudaMemcpyToSymbol(d_coeffs, h_coeffs, COEFF_COUNT * sizeof(float));
    cudaMemcpyToSymbol(d_coeffs, h_coeffs, COEFF_COUNT * sizeof(float));
    cudaMemcpyToSymbol(sinTable, h_sinTable, TABLE_SIZE * sizeof(float));
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float elapsedMs = 0.0f;

    printf("=== Constant Memory Exercise ===\n\n");
    
    // Test constant memory version
    printf("1. Running constant memory version...\n");
    cudaEventRecord(start);
    constantMemoryTransform<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedMs, start, stop);
    cudaMemcpy(h_output, d_output, N * sizeof(float), cudaMemcpyDeviceToHost);
    printFirstElements(h_output, 5, "   Results (first 5)");
    printf("   Time: %.3f ms\n", elapsedMs);
    
    // Test global memory version (for comparison)
    printf("\n2. Running global memory version...\n");
    cudaEventRecord(start);
    globalMemoryTransform<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, N, d_coeffs_global);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedMs, start, stop);
    cudaMemcpy(h_output, d_output, N * sizeof(float), cudaMemcpyDeviceToHost);
    printFirstElements(h_output, 5, "   Results (first 5)");
    printf("   Time: %.3f ms\n", elapsedMs);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    
    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_coeffs_global);
    free(h_input);
    free(h_output);
    
    printf("\n=== Key Takeaways ===\n");
    printf("- Constant memory is cached and optimized for broadcast reads\n");
    printf("- When all threads read same address, constant memory is very fast\n");
    printf("- Use cudaMemcpyToSymbol to copy data to constant memory\n");
    printf("- Limited to 64KB total constant memory\n");
    
    return 0;
}