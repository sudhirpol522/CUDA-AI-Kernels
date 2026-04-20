#include <cuda_runtime.h>
#include <iostream>

__global__ void gridStrideBasic(float *output, int n) {
    // TODO: Calculate global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // TODO: Calculate grid stride (total threads across all blocks)
    // Hint: stride = blockDim.x * gridDim.x
    int stride = blockDim.x * gridDim.x;
    
    // TODO: Implement grid-stride loop
    // Each thread processes multiple elements spaced by 'stride'
    for (int i = idx; i < n; i += stride) {
        output[i] = i * 3.0f;
    }
}

// ============================================================================
// KERNEL 2: 2D Grid-Stride Loop for Image Processing
// Complete the 2D grid-stride pattern
// ============================================================================
__global__ void gridStride2D(float *output, int width, int height, float multiplier) {
    // TODO: Calculate starting column and row
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    // TODO: Calculate strides in each dimension
    int strideX = blockDim.x * gridDim.x;
    int strideY = blockDim.y * gridDim.y;
    
    // TODO: Implement 2D grid-stride loop
    for (int y = row; y < height; y += strideY) {
        for (int x = col; x < width; x += strideX) {
            int idx = y * width + x;
            output[idx] = (y * width + x) * multiplier;
        }
    }
}
__global__ void gridStrideAccumulate(float *input, float *output, int n) {
    // TODO: Calculate thread index and stride
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    // Each output element corresponds to one thread
    // Each thread accumulates sum of elements it visits
    float sum = 0.0f;
    
    // TODO: Implement accumulation loop
     for (int i = idx; i < n; i += stride) {
       sum += input[i];
     }
    
    output[idx] = sum;
}

void initArray(float *arr, int n, float val) {
    for (int i = 0; i < n; i++) arr[i] = val;
}

bool verifyBasic(float *result, int n) {
    for (int i = 0; i < n; i++) {
        if (result[i] != i * 3.0f) return false;
    }
    return true;
}

bool verify2D(float *result, int width, int height, float mult) {
    for (int i = 0; i < width * height; i++) {
        if (result[i] != i * mult) return false;
    }
    return true;
}

int main() {
    printf("=== Thread Hierarchy Level 2: Grid-Stride Loops ===\n\n");
    
    // Test 1: Basic grid-stride with small grid (forces multiple iterations)
    printf("Testing basic grid-stride loop...\n");
    const int N = 10000;
    float *d_out, *d_in;
    cudaMalloc(&d_out, N * sizeof(float));
    cudaMalloc(&d_in, N * sizeof(float));
    
    // Use small grid to force multiple iterations per thread
    int blockSize = 256;
    int gridSize = 10;  // Only 10 blocks, so each thread handles multiple elements
    
    cudaMemset(d_out, 0, N * sizeof(float));
    gridStrideBasic<<<gridSize, blockSize>>>(d_out, N);
    cudaDeviceSynchronize();
    
    float *h_result = (float*)malloc(N * sizeof(float));
    cudaMemcpy(h_result, d_out, N * sizeof(float), cudaMemcpyDeviceToHost);
    
    if (verifyBasic(h_result, N)) {
        printf("✓ Basic grid-stride PASSED\n");
    } else {
        printf("✗ Basic grid-stride FAILED - Check your stride calculation\n");
    }
    
    // Test 2: 2D grid-stride
    printf("\nTesting 2D grid-stride loop...\n");
    const int W = 256, H = 256;
    const int N2D = W * H;
    cudaMemset(d_out, 0, N2D * sizeof(float));
    
    dim3 block2D(16, 16);
    dim3 grid2D(8, 8);  // Small grid forces multiple iterations
    gridStride2D<<<grid2D, block2D>>>(d_out, W, H, 2.5f);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_result, d_out, N2D * sizeof(float), cudaMemcpyDeviceToHost);
    
    if (verify2D(h_result, W, H, 2.5f)) {
        printf("✓ 2D grid-stride PASSED\n");
    } else {
        printf("✗ 2D grid-stride FAILED - Check your 2D stride implementation\n");
    }
    
    // Test 3: Grid-stride accumulation
    printf("\nTesting grid-stride accumulation...\n");
    const int N_ACC = 1000;
    float *h_input = (float*)malloc(N_ACC * sizeof(float));
    for (int i = 0; i < N_ACC; i++) h_input[i] = 1.0f;
    
    float *d_input;
    cudaMalloc(&d_input, N_ACC * sizeof(float));
    cudaMemcpy(d_input, h_input, N_ACC * sizeof(float), cudaMemcpyHostToDevice);
    
    int blockSizeAcc = 256;
    int gridSizeAcc = (N_ACC + blockSizeAcc - 1) / blockSizeAcc;
    
    float *d_outputAcc;
    cudaMalloc(&d_outputAcc, gridSizeAcc * sizeof(float));
    
    gridStrideAccumulate<<<gridSizeAcc, blockSizeAcc>>>(d_input, d_outputAcc, N_ACC);
    cudaDeviceSynchronize();
    
    float *h_outputAcc = (float*)malloc(gridSizeAcc * sizeof(float));
    cudaMemcpy(h_outputAcc, d_outputAcc, gridSizeAcc * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Each thread should have summed all elements it visited
    bool passAcc = true;
    for (int t = 0; t < gridSizeAcc && passAcc; t++) {
        float expected = 0.0f;
        for (int i = t; i < N_ACC; i += gridSizeAcc * blockSizeAcc) {
            expected += 1.0f;
        }
        if (h_outputAcc[t] != expected) passAcc = false;
    }
    
    if (passAcc) {
        printf("✓ Grid-stride accumulation PASSED\n");
    } else {
        printf("✗ Grid-stride accumulation FAILED - Check your accumulation logic\n");
    }
    
    // Cleanup
    free(h_result);
    free(h_input);
    free(h_outputAcc);
    cudaFree(d_out);
    cudaFree(d_in);
    cudaFree(d_input);
    cudaFree(d_outputAcc);
    
    printf("\n=== Level 2 Complete ===\n");
    printf("Next: Try level3_multidim_data.cu for image/volume processing\n");
    
    return 0;}