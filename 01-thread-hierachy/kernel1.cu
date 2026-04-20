#include <cuda_runtime.h>
#include <iostream>


__global__ void process1D(float* d_in, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        d_in[idx] = idx * 2;
    }
}

__global__ void process2D(float *output, int width, int height) {
    // TODO: Calculate column and row from 2D thread configuration
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    
 
    int idx = row * width + col;
    
    if (row < height && col < width) {
        output[idx] = (row * width + col) * 2.0f;
    }
}
__global__ void process3D(float *output, int width, int height, int depth) {
    // TODO: Calculate x, y, z coordinates from 3D thread configuration
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    int idx = z * (width * height) + y * width + x;
    
    if (x < width && y < height && z < depth) {
        output[idx] = (z * width * height + y * width + x) * 2.0f;
    }
}
// Utility functions
void initArray(float *arr, int n) {
    for (int i = 0; i < n; i++) arr[i] = 0.0f;
}

bool verify1D(float *result, int n) {
    for (int i = 0; i < n; i++) {
        if (result[i] != i * 2.0f) return false;
    }
    return true;
}

bool verify2D(float *result, int width, int height) {
    for (int row = 0; row < height; row++) {
        for (int col = 0; col < width; col++) {
            int idx = row * width + col;
            if (result[idx] != idx * 2.0f) return false;
        }
    }
    return true;
}

int main() {
    printf("=== Thread Hierarchy Level 1: Basic Indexing ===\n\n");
    
    // Test 1D configuration
    printf("Testing 1D thread configuration...\n");
    const int N = 1024;
    float *d_out1D;
    cudaMalloc(&d_out1D, N * sizeof(float));
    cudaMemset(d_out1D, 0, N * sizeof(float));
    
    process1D<<<(N + 255) / 256, 256>>>(d_out1D, N);
    cudaDeviceSynchronize();
    
    float *h_result = (float*)malloc(N * sizeof(float));
    cudaMemcpy(h_result, d_out1D, N * sizeof(float), cudaMemcpyDeviceToHost);
    
    if (verify1D(h_result, N)) {
        printf("✓ 1D indexing PASSED\n");
    } else {
        printf("✗ 1D indexing FAILED - Check your index calculation\n");
    }
    
    // Test 2D configuration
    printf("\nTesting 2D thread configuration...\n");
    const int WIDTH = 32, HEIGHT = 32;
    const int N2D = WIDTH * HEIGHT;
    float *d_out2D;
    cudaMalloc(&d_out2D, N2D * sizeof(float));
    cudaMemset(d_out2D, 0, N2D * sizeof(float));
    
    dim3 block2D(16, 16);
    dim3 grid2D((WIDTH + 15) / 16, (HEIGHT + 15) / 16);
    process2D<<<grid2D, block2D>>>(d_out2D, WIDTH, HEIGHT);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_result, d_out2D, N2D * sizeof(float), cudaMemcpyDeviceToHost);
    
    if (verify2D(h_result, WIDTH, HEIGHT)) {
        printf("✓ 2D indexing PASSED\n");
    } else {
        printf("✗ 2D indexing FAILED - Check your 2D to 1D mapping\n");
    }
    // Test 3D configuration
    printf("\nTesting 3D thread configuration...\n");
    const int W = 8, H = 8, D = 8;
    const int N3D = W * H * D;
    float *d_out3D;
    cudaMalloc(&d_out3D, N3D * sizeof(float));
    cudaMemset(d_out3D, 0, N3D * sizeof(float));
    
    dim3 block3D(4, 4, 4);
    dim3 grid3D((W + 3) / 4, (H + 3) / 4, (D + 3) / 4);
    process3D<<<grid3D, block3D>>>(d_out3D, W, H, D);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_result, d_out3D, N3D * sizeof(float), cudaMemcpyDeviceToHost);
    
    bool pass3D = true;
    for (int z = 0; z < D && pass3D; z++) {
        for (int y = 0; y < H && pass3D; y++) {
            for (int x = 0; x < W && pass3D; x++) {
                int idx = z * W * H + y * W + x;
                if (h_result[idx] != idx * 2.0f) pass3D = false;
            }
        }
    }
    
    if (pass3D) {
        printf("✓ 3D indexing PASSED\n");
    } else {
        printf("✗ 3D indexing FAILED - Check your 3D to 1D mapping\n");
    }
    
    
    // Cleanup
    free(h_result);
    cudaFree(d_out1D);
    cudaFree(d_out2D);
    cudaFree(d_out3D);
    printf("\n=== Level 1 Complete ===\n");
    
    return 0;
}
