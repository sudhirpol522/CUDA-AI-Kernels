// =============================================================================
// softmax.cu  --  Numerically Stable Parallel Softmax on CUDA
// =============================================================================
//
// WHAT IS SOFTMAX?
// ----------------
// Given a row of N numbers  x = [x0, x1, ..., x_{N-1}],
// softmax turns them into a probability distribution:
//
//       softmax(x)_i  =  exp(x_i) / sum_j( exp(x_j) )
//
// Every output is in (0, 1) and all outputs sum to exactly 1.
// It is used at the output of classifiers and inside attention heads.
//
// =============================================================================
// THE NUMERICAL STABILITY PROBLEM
// =============================================================================
// If any x_i is large (e.g. 1000), exp(1000) = +INF --> overflow.
// If any x_i is very negative (e.g. -1000), exp(-1000) ≈ 0 --> underflow.
//
// FIX: subtract the row maximum before exponentiating.
//
//   softmax(x)_i = exp(x_i - max_x) / sum_j( exp(x_j - max_x) )
//
// This is mathematically identical because the max cancels:
//
//   exp(x_i - max_x) / sum_j(exp(x_j - max_x))
//   = [exp(x_i)/exp(max_x)] / [sum_j(exp(x_j)/exp(max_x))]
//   = exp(x_i) / sum_j(exp(x_j))                          ✓
//
// Now the largest shifted value is exp(0) = 1.0, no overflow.
//
// =============================================================================
// CONCRETE EXAMPLE  (walk through this before reading the GPU code)
// =============================================================================
//
// Input row: x = [1.0,  2.0,  3.0,  4.0]   N = 4
//
// ---- PASS 1: find the row maximum ----
//   max_x = 4.0
//
// ---- PASS 2: shift, exponentiate, accumulate sum ----
//   e[0] = exp(1 - 4) = exp(-3) ≈ 0.0498
//   e[1] = exp(2 - 4) = exp(-2) ≈ 0.1353
//   e[2] = exp(3 - 4) = exp(-1) ≈ 0.3679
//   e[3] = exp(4 - 4) = exp( 0) = 1.0000
//   sum  = 0.0498 + 0.1353 + 0.3679 + 1.0000 = 1.5530
//
// ---- PASS 3: normalise ----
//   out[0] = 0.0498 / 1.5530 ≈ 0.0321
//   out[1] = 0.1353 / 1.5530 ≈ 0.0871
//   out[2] = 0.3679 / 1.5530 ≈ 0.2369
//   out[3] = 1.0000 / 1.5530 ≈ 0.6439
//   check: 0.0321 + 0.0871 + 0.2369 + 0.6439 = 1.0000  ✓
//
// The GPU kernel below runs all three passes on each row in parallel.
//
// =============================================================================
// HOW THE GPU PARALLELISES THIS
// =============================================================================
//
// Grid layout
// -----------
//   - One CUDA block  = one row of the matrix.
//   - `threads` threads inside that block cooperate to reduce that row.
//   - threads = TILE * TILE  (e.g. TILE=16 → 256 threads/block)
//
// When N > threads, each thread "owns" multiple columns via a strided loop:
//   thread 0 owns columns 0, 256, 512, ...
//   thread 1 owns columns 1, 257, 513, ...
//
// After the strided loop every thread holds a partial result.
// Those partials must be combined (reduced) to get the single row maximum
// or row sum.  The reduction uses two steps:
//
//   Step A – Shared-memory tree reduction (for threads >= 32)
//   ----------------------------------------------------------
//   Imagine 256 threads:
//     round 1 (s=128):  thread 0 = max(t0, t128),  thread 1 = max(t1, t129), ...
//     round 2 (s=64):   thread 0 = max(t0, t64),   ...
//     round 3 (s=32):   thread 0 = max(t0, t32),   ...
//   Now only threads 0-31 (one warp) hold meaningful values.
//
//   Step B – Warp-shuffle reduction (for the final 32 threads)
//   -----------------------------------------------------------
//   Threads in the same warp (32 threads) can exchange registers directly
//   using __shfl_down_sync without going through shared memory.
//   This is faster because it avoids cache/memory traffic entirely.
//
//     offset=16: thread 0 = max(t0, t16), thread 1 = max(t1, t17), ...
//     offset=8:  thread 0 = max(t0, t8),  ...
//     offset=4:  thread 0 = max(t0, t4),  ...
//     offset=2:  thread 0 = max(t0, t2),  ...
//     offset=1:  thread 0 = max(t0, t1)   ← final answer in thread 0
//   Thread 0 writes the result back to sdata[0] so all threads can read it.
//
// The same two-step reduction is done for the sum.
//
// =============================================================================

#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <cassert>

// ---------------------------------------------------------------------------
// Helper: check CUDA calls
// ---------------------------------------------------------------------------
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d  %s\n",                      \
                    __FILE__, __LINE__, cudaGetErrorString(err));              \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

// ---------------------------------------------------------------------------
// The kernel
//   input  : [M x N] row-major matrix   (M rows, N columns)
//   output : [M x N] result (each row is a probability distribution)
//   N      : number of columns
//
// TILE is a compile-time constant so the compiler knows the shared-memory size
// and can unroll/optimise the loops.
// ---------------------------------------------------------------------------
template <int TILE>
__global__ void softmax_kernel(const float *input, float *output, int N)
{
    // -----------------------------------------------------------------------
    // SETUP
    // -----------------------------------------------------------------------
    // Each block handles exactly one row.
    int row     = blockIdx.x;           // which row this block owns
    int tid     = threadIdx.x;          // this thread's ID within the block
    constexpr int threads = TILE * TILE; // total threads per block (e.g. 256)

    const float *row_in  = input  + (long long)row * N; // pointer to this row's input
    float       *row_out = output + (long long)row * N; // pointer to this row's output

    // Shared memory: every thread writes its partial result here so neighbours
    // can read it during the tree reduction.
    __shared__ float sdata[threads];

    // -----------------------------------------------------------------------
    // PASS 1 — find the row maximum
    // -----------------------------------------------------------------------
    // Each thread scans its own "stripe" of columns and tracks its local max.
    //
    // Example: N=1024, threads=256
    //   thread 0 checks columns 0, 256, 512, 768
    //   thread 1 checks columns 1, 257, 513, 769
    //   ...
    float local_max = -__int_as_float(0x7f800000); // -INF as a float literal
    for (int i = tid; i < N; i += threads) {
        float v = row_in[i];
        if (v > local_max) local_max = v;
    }

    // Write each thread's partial max into shared memory.
    sdata[tid] = local_max;
    __syncthreads(); // wait for ALL threads to finish writing

    // --- Shared-memory tree reduction to narrow down to 32 values ---
    // After each iteration the "active" range halves.
    // s=128: 128 active threads each merge with a partner 128 away
    // s=64:   64 active threads merge
    // s=32:   32 active threads merge  → stops here, hand off to warp shuffle
    for (int s = threads / 2; s >= 32; s >>= 1) {
        if (tid < s)
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        __syncthreads(); // barrier between rounds so no thread reads stale data
    }

    // --- Warp-level shuffle reduction (no shared memory needed) ---
    // Only threads 0-31 participate (one warp).
    // __shfl_down_sync(mask, val, offset) returns the value of the thread
    // that is `offset` lanes ahead of the current thread.
    if (tid < 32) {
        float v = sdata[tid];
        //  offset 16: combine t0+t16, t1+t17, ... (8 meaningful values remain)
        //  offset  8: combine t0+t8,  ...         (4)
        //  offset  4: combine t0+t4,  ...         (2)
        //  offset  2: combine t0+t2,  ...         (1, in t0)
        //  offset  1: combine t0+t1              (done)
        for (int offset = 16; offset > 0; offset >>= 1)
            v = fmaxf(v, __shfl_down_sync(0xFFFFFFFFu, v, offset));
        // After this loop, thread 0 holds the true row maximum.
        if (tid == 0) sdata[0] = v;
    }
    __syncthreads(); // broadcast sdata[0] to every thread

    float max_val = sdata[0]; // every thread now knows the row maximum

    // -----------------------------------------------------------------------
    // PASS 2 — compute exp(x_i - max_val) and accumulate partial sums
    // -----------------------------------------------------------------------
    // We also STORE the exponential into the output array so we don't need
    // to recompute it in pass 3.
    float local_sum = 0.0f;
    for (int i = tid; i < N; i += threads) {
        float e = expf(row_in[i] - max_val); // shift makes this safe
        row_out[i]  = e;                     // temporarily store in output
        local_sum  += e;
    }

    // Same two-step reduction for the sum.
    sdata[tid] = local_sum;
    __syncthreads();

    for (int s = threads / 2; s >= 32; s >>= 1) {
        if (tid < s)
            sdata[tid] = sdata[tid] + sdata[tid + s];
        __syncthreads();
    }

    if (tid < 32) {
        float v = sdata[tid];
        for (int offset = 16; offset > 0; offset >>= 1)
            v += __shfl_down_sync(0xFFFFFFFFu, v, offset);
        if (tid == 0) sdata[0] = v;
    }
    __syncthreads();

    float exp_sum = sdata[0]; // every thread now knows sum(exp(x_i - max))

    // -----------------------------------------------------------------------
    // PASS 3 — normalise: divide every stored exponential by the sum
    // -----------------------------------------------------------------------
    for (int i = tid; i < N; i += threads)
        row_out[i] = row_out[i] / exp_sum;
}

// ---------------------------------------------------------------------------
// Launch wrapper
//   M : number of rows  (one block per row)
//   N : number of columns
// ---------------------------------------------------------------------------
template <int TILE>
void launch_softmax(const float *d_in, float *d_out,
                    int M, int N, cudaStream_t stream = 0)
{
    constexpr int threadsPerBlock = TILE * TILE;
    softmax_kernel<TILE><<<M, threadsPerBlock, 0, stream>>>(d_in, d_out, N);
    CUDA_CHECK(cudaGetLastError());
}

// ---------------------------------------------------------------------------
// CPU reference (for verification)
// ---------------------------------------------------------------------------
void softmax_cpu(const float *h_in, float *h_out, int M, int N)
{
    for (int r = 0; r < M; ++r) {
        const float *row_in  = h_in  + r * N;
        float       *row_out = h_out + r * N;

        // find max
        float mx = row_in[0];
        for (int c = 1; c < N; ++c) if (row_in[c] > mx) mx = row_in[c];

        // exp and sum
        float s = 0.f;
        for (int c = 0; c < N; ++c) {
            row_out[c] = expf(row_in[c] - mx);
            s += row_out[c];
        }
        // normalise
        for (int c = 0; c < N; ++c) row_out[c] /= s;
    }
}

// ---------------------------------------------------------------------------
// main: run the small example from the comment block, then verify a larger one
// ---------------------------------------------------------------------------
int main()
{
    // ------------------------------------------------------------------
    // Tiny worked example: one row, [1, 2, 3, 4]
    // Expected output ≈ [0.0321, 0.0871, 0.2369, 0.6439]
    // ------------------------------------------------------------------
    printf("=== Tiny example: row = [1, 2, 3, 4] ===\n");
    {
        const int M = 1, N = 4;
        float h_in[N]  = {1.f, 2.f, 3.f, 4.f};
        float h_out[N] = {};

        float *d_in, *d_out;
        CUDA_CHECK(cudaMalloc(&d_in,  M * N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_out, M * N * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_in, h_in, M * N * sizeof(float), cudaMemcpyHostToDevice));

        launch_softmax<4>(d_in, d_out, M, N);   // TILE=4 → 16 threads
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(h_out, d_out, M * N * sizeof(float), cudaMemcpyDeviceToHost));

        float s = 0.f;
        for (int i = 0; i < N; ++i) { printf("  out[%d] = %.4f\n", i, h_out[i]); s += h_out[i]; }
        printf("  sum = %.4f  (should be 1.0)\n\n", s);

        CUDA_CHECK(cudaFree(d_in));
        CUDA_CHECK(cudaFree(d_out));
    }

    // ------------------------------------------------------------------
    // Larger correctness check: 128 rows × 1024 columns
    // ------------------------------------------------------------------
    printf("=== Correctness check: 128 rows x 1024 cols ===\n");
    {
        const int M = 128, N = 1024;
        size_t bytes = (size_t)M * N * sizeof(float);

        float *h_in  = new float[M * N];
        float *h_ref = new float[M * N];
        float *h_out = new float[M * N];

        // fill with some values
        for (int i = 0; i < M * N; ++i) h_in[i] = sinf((float)i) * 5.f;

        softmax_cpu(h_in, h_ref, M, N);

        float *d_in, *d_out;
        CUDA_CHECK(cudaMalloc(&d_in,  bytes));
        CUDA_CHECK(cudaMalloc(&d_out, bytes));
        CUDA_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

        launch_softmax<16>(d_in, d_out, M, N);  // TILE=16 → 256 threads
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));

        float max_err = 0.f;
        for (int i = 0; i < M * N; ++i)
            max_err = fmaxf(max_err, fabsf(h_out[i] - h_ref[i]));
        printf("  max absolute error vs CPU = %.2e  (%s)\n",
               max_err, max_err < 1e-5f ? "PASS" : "FAIL");

        delete[] h_in; delete[] h_ref; delete[] h_out;
        CUDA_CHECK(cudaFree(d_in));
        CUDA_CHECK(cudaFree(d_out));
    }

    return 0;
}
