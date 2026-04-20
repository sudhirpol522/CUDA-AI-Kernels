#include <cuda_runtime.h>
#include <stdio.h>

// ============================================================================
// Demo: float4 loads on a 4x4 matrix
//
// Launch config: 1 block, dim3(1, 4) threads  → 4 threads total
//   threadIdx.x = 0 for all threads
//   threadIdx.y = 0, 1, 2, 3
//
// Each thread (ty) loads an entire ROW of 4 floats using float4.
// Total: 4 threads × 4 floats = 16 floats = entire 4x4 matrix in one pass.
// ============================================================================

#define N 4  // 4x4 matrix

__global__ void float4LoadDemo(float *A, float *out) {
    int ty = threadIdx.y;   // 0, 1, 2, 3 — one thread per row

    // Each thread loads its entire row as a single float4 (16-byte load)
    float4 row = *reinterpret_cast<float4*>(&A[ty * N]);

    // Print from GPU side what each thread loaded
    printf("  Thread ty=%d loaded row %d:  %.1f  %.1f  %.1f  %.1f\n",
           ty, ty, row.x, row.y, row.z, row.w);

    // Write to output so we can verify on CPU too
    out[ty * N + 0] = row.x;
    out[ty * N + 1] = row.y;
    out[ty * N + 2] = row.z;
    out[ty * N + 3] = row.w;
}

int main() {
    // -----------------------------------------------------------------------
    // Setup: 4x4 matrix with values 1..16
    // -----------------------------------------------------------------------
    float h_A[N * N];
    float h_out[N * N];

    printf("=== Input Matrix (4x4) ===\n");
    for (int i = 0; i < N; i++) {
        printf("  Row %d: ", i);
        for (int j = 0; j < N; j++) {
            h_A[i * N + j] = (float)(i * N + j + 1);  // 1..16
            printf("%5.1f ", h_A[i * N + j]);
        }
        printf("\n");
    }

    // -----------------------------------------------------------------------
    // Device memory
    // -----------------------------------------------------------------------
    float *d_A, *d_out;
    cudaMalloc(&d_A,   N * N * sizeof(float));
    cudaMalloc(&d_out, N * N * sizeof(float));
    cudaMemcpy(d_A, h_A, N * N * sizeof(float), cudaMemcpyHostToDevice);

    // -----------------------------------------------------------------------
    // Launch: 1 block, (x=1, y=4) threads → 4 threads total
    // Each thread handles 1 row → 4 rows covered
    // -----------------------------------------------------------------------
    printf("\n=== Kernel Output (what each thread loaded via float4) ===\n");
    dim3 block(1, 4);   // x=1, y=4
    dim3 grid(1, 1);    // single block
    float4LoadDemo<<<grid, block>>>(d_A, d_out);
    cudaDeviceSynchronize();

    // -----------------------------------------------------------------------
    // Copy back and verify
    // -----------------------------------------------------------------------
    cudaMemcpy(h_out, d_out, N * N * sizeof(float), cudaMemcpyDeviceToHost);

    printf("\n=== Output Matrix (written back from GPU) ===\n");
    bool match = true;
    for (int i = 0; i < N; i++) {
        printf("  Row %d: ", i);
        for (int j = 0; j < N; j++) {
            printf("%5.1f ", h_out[i * N + j]);
            if (h_out[i * N + j] != h_A[i * N + j]) match = false;
        }
        printf("\n");
    }

    printf("\n=== Verification: %s ===\n", match ? "PASSED ✓" : "FAILED ✗");

    printf("\n=== Thread → Memory Mapping ===\n");
    printf("  Block config : dim3(x=1, y=4) → %d threads total\n", 1 * 4);
    printf("  Each thread  : loads 1 row × 4 floats = float4 (128-bit load)\n");
    printf("  Total loaded : 4 threads × 4 floats = %d floats = entire matrix\n", N * N);
    printf("  Address math : thread ty=%d reads &A[ty*%d] = &A[%d]\n",
           2, N, 2 * N);

    cudaFree(d_A);
    cudaFree(d_out);
    return 0;
}
