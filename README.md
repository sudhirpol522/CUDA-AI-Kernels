# CUDA AI Kernels

A collection of CUDA kernels implementing common building blocks for deep learning workloads, built incrementally from naive baselines to heavily optimized versions. Each kernel is benchmarked on real hardware and accompanied by commentary on why each optimization produces the measured speedup.

The repository is organized as a learning progression. The same operation is implemented several times with increasingly aggressive optimizations, and the benchmark files print time, bandwidth, and percent of peak for direct comparison.

## Hardware used for benchmarks

- GPU: NVIDIA GeForce GTX 1660 Ti (Turing, sm_75)
- Peak memory bandwidth: 288 GB/s (192-bit bus, 12 Gbps GDDR6)
- VRAM: 6 GB

## Benchmark results

### Matrix-vector multiplication (GEMV), 8192 x 8192

| Kernel                     | Time (ms) | Bandwidth (GB/s) | Percent of peak | Speedup |
|----------------------------|-----------|------------------|-----------------|---------|
| Naive (one thread per row) | 8.35      | 32               | 11.2            | 1.0x    |
| Coalesced warp             | 2.18      | 123              | 42.9            | 3.8x    |
| Coalesced warp plus block  | 1.33      | 201              | 70.0            | 6.3x    |
| Vectorized float4          | 1.00      | 268              | 93.1            | 8.4x    |

### Softmax, 8192 x 8192

| Kernel                             | Passes | Time (ms) | Bandwidth (GB/s) | Percent of peak | Speedup |
|------------------------------------|--------|-----------|------------------|-----------------|---------|
| Naive three-pass                   | 4      | 77.7      | 14               | 4.8             | 1.0x    |
| Online two-pass                    | 3      | 68.4      | 12               | 4.1             | 1.1x    |
| Coalesced warp online              | 3      | 6.0       | 134              | 46.6            | 13.0x   |
| Shared memory block reduce (1024t) | 3      | 8.0       | 100              | 34.9            | 9.7x    |
| Vectorized float4 plus shuffle     | 3      | 3.2       | 251              | 87.3            | 24.3x   |

## Repository structure

```
01-thread-hierachy/       Thread indexing and grid and block layouts
02-memory-hierachy/       Global, shared, and register memory examples
03-memory-coalescing/     Coalesced vs non-coalesced access patterns
05-reduction-patterns/    Tree reduction, warp shuffle, block reduction
06-deep-learning-kernels/ Softmax with numerical stability, Flash Attention
08_warp_primitives/       Warp-level shuffle and vote intrinsics
practice/                 Benchmark drivers and reference implementations
compile.bat               Build helper for Windows with MSVC and nvcc
PROFILING-GUIDE.md        Notes on using Nsight Compute (ncu) on Windows
```

### Key files in practice/

| File                      | Description                                                           |
|---------------------------|-----------------------------------------------------------------------|
| sgemv_bench.cu            | Four GEMV kernels with side-by-side timing                            |
| softmax_bench.cu          | Five softmax kernels with side-by-side timing                         |
| sgemv_block_reduce.cu     | Instrumented block reduction that prints intermediate values          |
| warp_reduce_max.cu        | Warp-level max reduction using shuffle intrinsics                     |
| coasled_access.cu         | Reference coalesced GEMV kernel used by other drivers                 |

### Key files in 06-deep-learning-kernels/

| File                          | Description                                                                          |
|-------------------------------|--------------------------------------------------------------------------------------|
| softmax.cu                    | Numerically stable three-pass softmax with worked numerical example                  |
| flash_attention_explained.cu  | Flash Attention with online softmax, block tiling, and step-by-step commentary       |

## How to build

The project targets Windows with MSVC and the NVIDIA CUDA Toolkit. A helper batch script is included.

```
compile.bat path\to\kernel.cu path\to\output_name
```

For example:

```
compile.bat practice\softmax_bench.cu practice\softmax_bench
practice\softmax_bench.exe
```

The script internally invokes nvcc with optimization flags and links against the CUDA runtime. Any kernel file in the repository can be built this way.

## How to run the benchmarks

After building, run the executable. Each benchmark runs a warmup phase and then times two hundred iterations using cudaEvent timing, which has microsecond resolution.

```
practice\sgemv_bench.exe
practice\softmax_bench.exe
```

Output includes wall-clock time, effective memory bandwidth, percent of theoretical peak bandwidth, and a correctness check against a CPU reference.

## Design notes

The optimizations demonstrated in this repository follow a consistent pattern that applies to most memory-bound GPU kernels:

1. Coalesce global memory accesses. A warp of thirty-two threads should read thirty-two consecutive floats so that the memory controller can combine them into a single cache-line transaction. This alone accounts for the largest single speedup in both GEMV and softmax.
2. Increase occupancy by using more threads per row or column. More active warps per streaming multiprocessor allow the hardware to hide memory latency by switching to a ready warp while another waits for data.
3. Replace shared memory tree reductions with warp-shuffle reductions where possible. Shuffles operate directly on registers and do not require syncthreads barriers, reducing stall cycles at the warp level.
4. Use vectorized loads (float4) to cut the instruction count in half or more. One load instruction fetches sixteen bytes instead of four, keeping the memory pipeline more continuously saturated.
5. For functions like softmax that call expf internally, compute throughput also matters. Even with perfect memory access the transcendental unit caps throughput below pure memory-bound kernels.

## License

The reference implementations are inspired by and in parts based on the CUDA optimization blog by Maharshi Pandya, released under the Apache 2.0 license. Original derivations, commentary, and benchmark drivers in this repository are provided for educational use.
