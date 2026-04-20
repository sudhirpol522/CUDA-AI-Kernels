#include <cuda_runtime.h>
#include <stdio.h>


__global__ void adjustBrightness(unsigned char *image, int width, int height, float delta) {
    // Calculate pixel coordinates
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Add bounds check
    if (x < width && y < height) {
        int idx = y * width + x;
        float adjusted = image[idx] + delta;
        // Clamp to valid range [0, 255]
        image[idx] = (unsigned char)(adjusted < 0 ? 0 : (adjusted > 255 ? 255 : adjusted));
    }
}
// KERNEL 2: RGB Image Channel Swap (RGB -> BGR)
// Swap red and blue channels in an interleaved RGB image
// ============================================================================
__global__ void swapRBChannels(unsigned char *image, int width, int height) {
    // TODO: Calculate pixel coordinates
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Add bounds check
    if (x < width && y < height) {
        // TODO: Calculate base index for this pixel (3 channels per pixel)
        int idx = (y * width + x)*3;
        
        // Swap R (channel 0) and B (channel 2)
        unsigned char temp = image[idx];
        image[idx] = image[idx + 2];
        image[idx + 2] = temp;
    }
}

__global__ void convoltion(float* din, float* dout, int width, int height, float* kernel, int kernelSize) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (row < height && col < width) {
       int halfKernelSize = kernelSize / 2;
       float sum = 0.0f;
       for (int y=0;y<kernelSize;y++) {
        for (int x=0;x<kernelSize;x++) {
            int pr=row+y-halfKernelSize;
            int pc=col+x-halfKernelSize;
            if (pr>=0 && pr<height && pc>=0 && pc<width) {
                int kidx = y * kernelSize + x;
                int didx = (pr * width + pc);
                sum += din[didx] * kernel[kidx];
            }
        }
       }
       dout[row * width + col] = sum;
    }
    
    }




// Utility functions
void initImage(unsigned char *img, int width, int height) {
    for (int i = 0; i < width * height; i++) {
        img[i] = (i % 256);
    }
}

void initRGBImage(unsigned char *img, int width, int height) {
    for (int i = 0; i < width * height * 3; i++) {
        img[i] = (i % 256);
    }
}

bool verifyBrightness(unsigned char *result, int width, int height, float delta) {
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int idx = y * width + x;
            float expected = (idx % 256) + delta;
            unsigned char clamped = (expected < 0) ? 0 : (expected > 255 ? 255 : (unsigned char)expected);
            if (result[idx] != clamped) return false;
        }
    }
    return true;
}

bool verifySwapRB(unsigned char *result, int width, int height) {
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int base = (y * width + x) * 3;
            // Original: R=base%256, G=(base+1)%256, B=(base+2)%256
            // After swap: B should be at base, R at base+2
            unsigned char origR = (base) % 256;
            unsigned char origB = (base + 2) % 256;
            if (result[base] != origB || result[base + 2] != origR) return false;
        }
    }
    return true;
}
int main() {
    printf("=== Thread Hierarchy Level 3: Multi-Dimensional Data ===\n\n");
    
    // Test 1: Brightness adjustment
    printf("Testing brightness adjustment...\n");
    const int W = 256, H = 256;
    unsigned char *h_image, *d_image;
    h_image = (unsigned char*)malloc(W * H);
    initImage(h_image, W, H);
    
    cudaMalloc(&d_image, W * H);
    cudaMemcpy(d_image, h_image, W * H, cudaMemcpyHostToDevice);
    
    dim3 block(W / 16, H / 16);
    adjustBrightness<<<16, 16>>>(d_image, W, H, 50.0f);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_image, d_image, W * H, cudaMemcpyDeviceToHost);
    
    if (verifyBrightness(h_image, W, H, 50.0f)) {
        printf("✓ Brightness adjustment PASSED\n");
    } else {
        printf("✗ Brightness adjustment FAILED - Check pixel indexing\n");
    }
    
    // Test 2: RGB channel swap
    printf("\nTesting RGB channel swap...\n");
    unsigned char *h_rgb, *d_rgb;
    h_rgb = (unsigned char*)malloc(W * H * 3);
    initRGBImage(h_rgb, W, H);
    
    cudaMalloc(&d_rgb, W * H * 3);
    cudaMemcpy(d_rgb, h_rgb, W * H * 3, cudaMemcpyHostToDevice);
    
    dim3 blockRGB(16, 16);
    dim3 gridRGB((W + 15) / 16, (H + 15) / 16);
    swapRBChannels<<<gridRGB, blockRGB>>>(d_rgb, W, H);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_rgb, d_rgb, W * H * 3, cudaMemcpyDeviceToHost);
    
    if (verifySwapRB(h_rgb, W, H)) {
        printf("✓ RGB channel swap PASSED\n");
    } else {
        printf("✗ RGB channel swap FAILED - Check channel indexing\n");
    }
    
    // Cleanup
    free(h_image);
    free(h_rgb);
    cudaFree(d_image);
    cudaFree(d_rgb);
    
    printf("\n=== Level 3 Complete ===\n");
    printf("Next: Try level4_warp_aware.cu for warp-level optimization\n");
    
    return 0;
}
