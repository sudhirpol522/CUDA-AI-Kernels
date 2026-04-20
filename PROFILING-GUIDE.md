# CUDA Kernel Profiling Guide
> Built from real profiling sessions on GTX 1660Ti (sm_75)
> Kernels: practice.cu, register_tiling.cu, float4_demo.cu

---

## Step 0 — The Workflow (Do This Every Time)

```
1. Write kernel
2. Compile:   .\compile.bat yourfile.cu
3. Profile:   .\compile.bat yourfile.cu profile   ← compiles with -lineinfo + generates .ncu-rep
4. Open:      Nsight Compute → File → Open → yourfile-analysis.ncu-rep
5. Select kernel from dropdown → Details tab → read top to bottom
6. Fix the TOP stall reason → re-profile → repeat
```

**Quick terminal-only checks (no GUI needed):**
```powershell
.\compile.bat yourfile.cu verbose     # register usage + spilling
.\compile.bat yourfile.cu banks       # shared memory bank conflicts
.\compile.bat yourfile.cu divergence  # warp divergence metrics
```

**⚠️ Source tab requires `-lineinfo`:**
The `profile` mode in compile.bat already adds `-lineinfo` automatically.
Without it, the Source tab in Nsight will show SASS only — no C++ source correlation.
If Source tab shows no file or greyed lines → recompile with `profile` mode.

**Note on dropdown:** Nsight shows every kernel LAUNCH, not just every kernel function.
Keep your benchmark simple: one correctness pass (N=64) + one benchmark pass (N=512), no warmups.
This gives you clean sequential IDs — one per kernel, easy to find.

**ID layout for register_tiling.cu (current structure):**
```
IDs 0–5:  Correctness check N=64  (tiny ~0.01ms — ignore for profiling)
IDs 6–11: Benchmark N=512         (these are the ones to profile)
  ID 6  = sgemmNaive              ← LG Throttle stall
  ID 7  = baselineMatMul          ← MIO Throttle ~25 cycles
  ID 8  = registerTiledMatMul2x2  ← MIO Throttle ~19 cycles
  ID 9  = registerTiledMatMul4x4  ← lower MIO, register pressure
  ID 10 = registerTiledMatMul1D   ← same as 2x2 (different block layout)
  ID 11 = registerTiledDoubleBuffer ← Barrier stall reduced
```

---

## How to Read GFLOP/s

```
GFLOP/s = FLOPs / time_in_seconds / 1,000,000,000

For matrix multiply (N×N):
  FLOPs = 2 × N³   (each of N² outputs needs N multiplies + N adds = 2N ops)

Example N=512:
  FLOPs = 2 × 512³ = 268,435,456
  Time  = 0.280 ms = 0.000280 s
  GFLOP/s = 268,435,456 / 0.000280 / 1e9 = 958.7

In code:  (2.0f * N * N * N) / (ms * 1e6f)
```

**What is FMA (Fused Multiply-Add)?**
```
FMA = a * b + c computed in ONE instruction (not two)
1 FMA instruction = 2 FLOPs (1 multiply + 1 add)

accumulator += tileA[ty][k] * tileB[k][tx]
↑ this IS an FMA — the compiler converts it automatically

Peak GFLOP/s = CUDA_cores × clock_Hz × 2 FLOPs_per_FMA
GTX 1660Ti  = 1408 × 1,430,000,000 × 2 = 4,027 GFLOP/s
```

**% of peak = Achieved GFLOP/s / Theoretical peak × 100**
Nsight shows this in the Roofline section. Always verify your achieved % matches.

---

## The 6 Sections to Read (in order)

---

### SECTION 1 — GPU Speed of Light
**What:** High-level health check. Two bars — Compute % and Memory %.

| Scenario | Diagnosis | Fix |
|----------|-----------|-----|
| Compute high, Memory low | Memory bound | Add tiling, improve coalescing, use shared mem |
| Memory high, Compute low | Compute bound | Reduce FLOPs, use `__fmaf`, `rsqrtf` |
| Both equal (~70%+) | Balanced ✓ | Minor tuning only |
| Both low (<40%) | Latency bound | Fix stalls first (check Warp State) |

**Ideal:** Both bars equal and as high as possible.

**Why important:** Tells you immediately WHERE to focus. No point optimizing memory
if you're compute bound and vice versa.

### ⚠️ MISCONCEPTION: "Low throughput % = slow kernel"
```
register_tiling.cu results:
  baseline:      Compute 71.56%, Memory 71.56% → "Balanced" → 500 GFLOP/s
  register 2×2:  Compute 46.71%, Memory 47.22% → "Latency Issue" → 958 GFLOP/s

The "worse" looking kernel was 2× FASTER.
```
Why? Throughput % is measured against peak hardware capacity.
Register tiled uses 256 blocks vs baseline's 1024 blocks.
Last wave has only 64/96 = 67% of SMs busy → average utilization looks low.
**Always check actual ms and GFLOP/s, not just throughput %.**

---

### SECTION 2 — Memory Workload Analysis
**What:** Where your memory traffic lives — DRAM → L2 → L1 → Shared Memory.

| Metric | Ideal | Bad Value Means | Fix |
|--------|-------|-----------------|-----|
| **L1 Hit Rate** | >50% | Data not in L1, falling to L2 | Improve locality / tile size |
| **L2 Hit Rate** | >85% | Data not reused, going to DRAM | Add tiling / shared memory |
| **DRAM Throughput** | As LOW as possible | Too much global memory traffic | Tile your data |
| **Memory Throughput (GB/s)** | Near GPU peak (if memory bound) | Bandwidth underutilized | Improve coalescing |
| **Local Memory Spilling** | **0** | Registers overflowed to DRAM (very slow) | Reduce variables, use `__launch_bounds__` |
| **Shared Mem Spilling** | **0** | Shared memory overflowed to global | Reduce tile size |

### ⚠️ Hit Rate vs Throughput — They Measure Different Things

```
Hit Rate   = QUALITY  — "of all requests, what % were already cached?"
Throughput = SPEED    — "how busy is the cache hardware unit?"

These are INDEPENDENT. A kernel can have:
  Low Hit Rate  + High Throughput = cache unit flooded with misses (uncoalesced access)
  High Hit Rate + Low Throughput  = data reused well, unit not the bottleneck
```

**Example from sgemmNaive:**
```
L1 Hit Rate   =  10%   ← almost nothing found in L1
L1 Throughput =  99%   ← but unit is completely maxed out

Why? Uncoalesced A access sends 16 cache requests per warp instead of 1.
Even though most miss, the VOLUME of requests saturates the L1 unit.
This shows up as "Stall LG Throttle" in Warp State — not Long Scoreboard.
```

**Which level to watch per kernel:**
```
sgemmNaive      → L1 Throughput (flooded by uncoalesced requests)
baselineMatMul  → L2 Hit Rate   (is tiling keeping data in L2?)
register tiled  → L2 Hit Rate still fine; bottleneck has moved to Warp State
```

**Results from register_tiling.cu:**
```
sgemmNaive:      L1 Hit Rate ~10%,  L1 Throughput 99%,  L2 Hit Rate ~low
baselineMatMul:  L1 Hit Rate ~10%,  L2 Hit Rate 96.58% ✓  DRAM 2.20% ✓
```

### ⚠️ Shared Memory is NOT Part of the Cache Hierarchy

```
Shared Memory ≠ L1 Cache

Shared Memory:          L1/L2 Cache:
  Lives inside the SM     Lives inside the SM (L1) or chip (L2)
  YOU manage it           Hardware manages it automatically
  __shared__ keyword      No keyword needed
  ~5 cycles               L1 ~30 cycles, L2 ~200 cycles
  Private to your block   L2 shared across all SMs
  Guaranteed hit          May miss (unknown until runtime)
```

Full hierarchy with locations:
```
Registers   ~1 cycle   ← inside SM, per-thread
Shared Mem  ~5 cycles  ← inside SM, per-block, YOU control
L1 Cache    ~30 cycles ← inside SM, hardware managed
L2 Cache   ~200 cycles ← on chip, shared by all SMs, hardware managed
DRAM       ~600 cycles ← off chip, avoid
```

**Coalescing rule:** 32 threads in a warp reading 32 consecutive floats
= 1 coalesced 128-byte transaction = Sectors/Request of 4. This is OPTIMAL.
If Sectors/Request > 4 → threads are accessing non-consecutive addresses → fix it.

### How Warp Threads Linearize — Critical for Coalescing

CUDA assigns thread IDs in **x-first order**:
```
tid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y
```

For a 2D block (16, 16):
```
Warp 0 = tid 0..31:
  tid  0-15: threadIdx.x=0..15, threadIdx.y=0   ← first row of block
  tid 16-31: threadIdx.x=0..15, threadIdx.y=1   ← second row of block
```

**Consequence:** threadIdx.x varies fastest within a warp.

**Rule:** For coalesced access, let **threadIdx.x select the column** (consecutive memory):
```c
// GOOD — threadIdx.x picks column → consecutive addresses across warp
A[row * width + threadIdx.x]    // thread 0→col 0, thread 1→col 1 ... ✓

// BAD — threadIdx.x picks row → stride-N addresses across warp
A[threadIdx.x * width + col]    // thread 0→row 0 (offset 0),
                                 // thread 1→row 1 (offset 512 floats) ✗
```

This is exactly why sgemmNaive's `A[row * width + i]` is uncoalesced:
row = blockIdx.x * blockDim.x + threadIdx.x → threadIdx.x drives the ROW,
so consecutive threads access addresses 512 floats apart.

---

### SECTION 3 — Shared Memory & Bank Conflicts
**What:** Detailed breakdown of shared memory loads/stores.

| Metric | Ideal | Bad Value Means | Fix |
|--------|-------|-----------------|-----|
| **Bank Conflicts (Load)** | **0** | Warp serialized N times | Pad arrays or change access pattern |
| **Bank Conflicts (Store)** | **0** | Same — warp store serialized | Same fix |
| **Shared Load % Peak** | 30–80% | Too low = underusing it, too high = bottleneck | Balance tile size |
| **Sectors/Request** | 1 (shared), 4 (global coalesced) | Replays = wasted cycles | Fix access stride |

**How bank conflicts happen:**
```cuda
// GOOD — threads read different banks (tx varies across warp)
sum += tileA[ty][k] * tileB[k][tx];   // 0 conflicts ✓
//   tileB[k][tx]: bank = tx → thread 0→bank 0, thread 1→bank 1... all different

// BAD — stride-2 access: threads 0 and 16 both hit bank 0
shared_data[threadIdx.x * 2];   // 2-way conflict ✗
```

**Fix for bank conflicts — pad by 1:**
```cuda
__shared__ float tileA[TILE_WIDTH][TILE_WIDTH + 1];  // +1 shifts banks, breaks conflicts
```

**Results from practice.cu:** Bank Conflicts Load = 0, Store = 0 ✓

### ⚠️ MISCONCEPTION: "Using shared memory causes bank conflicts"
The original `tileB[k][tx]` access was ALREADY conflict-free:
```
tileB[k][tx]: bank = (k*32 + tx) % 32 = tx
→ thread 0 → bank 0, thread 1 → bank 1, ... thread 31 → bank 31
→ All different → 0 conflicts
```
Transposing tileB to `tileBT[tx][k]` didn't improve anything because
there was nothing to improve. Always CHECK first before "fixing."

---

### SECTION 4 — Warp State Statistics (Most Diagnostic Section)
**What:** Where warps are spending their time — useful work vs stalling.

**Top two numbers to check first:**

| Metric | Ideal | Bad if |
|--------|-------|--------|
| **Warp Cycles Per Issued Instruction** | <10 | >20 means heavy stalling |
| **Avg Active Threads Per Warp** | **32** | <32 means warp divergence |

**Stall types — what each one means:**

| Stall | Typical Cycles | Meaning | Real Fix |
|-------|---------------|---------|----------|
| **LG Throttle** | 200–300 | Global load/store unit congested — too many requests queued (uncoalesced access) | Fix coalescing, add shared memory tiling |
| **MIO Throttle** | 20–30 | Too many shared mem ops per cycle | **Register tiling** (increase FMA/MIO ratio) |
| **Long Scoreboard** | 10–40 | Waiting for a specific global memory result | Increase occupancy to hide latency |
| **Barrier** | 5–15 | Waiting at `__syncthreads()` | Double buffering |
| **Not Selected** | 5–10 | Ready but scheduler chose another warp | Normal — means occupancy is good |
| **No Instructions** | <5 | Instruction cache miss | Keep kernels smaller |

**LG Throttle vs Long Scoreboard — key difference:**
```
LG Throttle     = the LOAD UNIT ITSELF is congested
                  Too many requests queued up, unit can't keep up
                  Caused by: uncoalesced access (16 requests per warp instead of 1)
                  Fix: fix coalescing first, then add tiling

Long Scoreboard = waiting for ONE specific load to RETURN
                  Unit is fine, just waiting for the data (latency)
                  Caused by: cache miss, DRAM latency
                  Fix: increase occupancy so other warps run while waiting
```

**What sgemmNaive shows (confirmed from profiling):**
```
Stall LG Throttle:   ~270 cycles  ← dominant (uncoalesced A access floods LG unit)
Stall Long Scoreboard: ~20 cycles
smsp__average_lg_throttle = 209.514 cycles  ← Nsight reports this directly
smsp__issue_active = 0.034 instructions/cycle (healthy = ~4.0)
```

### ⚠️ MISCONCEPTION: "#pragma unroll fixes MIO Throttle"
```
Before #pragma unroll: MIO throttle = 24.7032 cycles
After  #pragma unroll: MIO throttle = 24.7767 cycles  ← essentially unchanged

Timing improvement was only 3% (from removing loop overhead: branch + counter)
NOT from reducing MIO pressure.
```
`#pragma unroll` removes loop bookkeeping instructions. It does NOT reduce
the number of shared memory operations — those are fixed by the algorithm.

**The REAL fix for MIO Throttle = Register Tiling:**
```
Problem: not enough FMAs per shared memory load (FMA/MIO ratio too low)

Baseline (1 output/thread):
  32 MIO ops per outer iteration, 16 FMAs → ratio = 0.5

Register 2×2 (4 outputs/thread):
  64 MIO ops per outer iteration, 64 FMAs → ratio = 1.0

Result: MIO stall dropped 24.7 → 19.0 cycles, kernel went 500 → 958 GFLOP/s
```

**Warp divergence — when it's a problem:**
```cuda
// DIVERGENT — threads in same warp take different paths
if (threadIdx.x % 2 == 0) {
    doA();   // 16 threads run, 16 idle → 2× slower
} else {
    doB();
}

// FINE — all threads in warp take same path
if (blockIdx.x > 5) { doA(); }  // entire block same direction
```

### ⚠️ MISCONCEPTION: "float4 loads fix MIO Throttle"
```
tiledMatMulVectorized (float4 loads): 0.600 ms | 447 GFLOP/s ← SLOWER
baseline:                             0.494 ms | 543 GFLOP/s
```
Why float4 HURT performance:
- Requires `if ((tx & 3) == 0)` → 3/4 of threads idle per warp (warp underutilization)
- Global loads were ALREADY coalesced → same bytes moved, no bandwidth gain
- Added branch overhead on top

**float4 only helps when:** every thread loads exactly 4 contiguous elements
(e.g., 1D processing, each thread handles 4 adjacent array elements).

---

### SECTION 5 — Occupancy
**What:** How many warps are active on each SM vs the maximum possible.

| Metric | Ideal | Meaning |
|--------|-------|---------|
| **Theoretical Occupancy** | 100% | Hardware max given your config |
| **Achieved Occupancy** | >50% ideally, match theoretical | Low = latency not hidden |
| **Achieved Active Warps/SM** | = Max warps per SM | More warps = better stall hiding |

### ⚠️ MISCONCEPTION: "Higher occupancy = better performance"
```
register_tiling.cu results:
  baseline:      96.98% occupancy → 500 GFLOP/s  (SLOWER)
  register 2×2:  89.21% occupancy → 958 GFLOP/s  (FASTER)

Lower occupancy, 2× better performance.
```
Occupancy is a **latency-hiding** tool, NOT a performance metric.
It only matters when your kernel is **latency bound** (Long Scoreboard stalls).
When compute bound or MIO throttle bound, more warps don't help.

**Why register 2×2 achieved only 89.21% occupancy:**
```
256 total blocks, 24 SMs × 4 blocks/SM = 96 blocks per wave:
  Wave 1: 96 blocks → all 24 SMs full → 100%
  Wave 2: 96 blocks → 100%
  Wave 3: 64 blocks → only 67% of SMs have work
  Average: (100 + 100 + 67) / 3 = 89%  ← matches Nsight exactly
```

#### How to Calculate Occupancy Manually

```
Step 1: Registers limit
  Blocks/SM = floor(Registers_per_SM / (Regs_per_thread × Threads_per_block))

Step 2: Shared memory limit
  Blocks/SM = floor(SharedMem_per_SM / SharedMem_per_block)

Step 3: Warp limit
  Blocks/SM = floor(Max_warps_per_SM / (Threads_per_block / 32))

Step 4: Binding constraint = minimum of all three
Step 5: Occupancy = (Blocks/SM × Warps/block) / Max_warps_per_SM × 100
```

**Example (practice.cu tiledMatMul, GTX 1660Ti):**
```
Registers:    floor(65536 / (42 × 1024)) = floor(1.52) = 1 block
Shared mem:   floor(32770 / 8192)        = floor(3.99) = 3 blocks
Warps:        floor(32 / 32)             = 1 block
→ min = 1 block × 32 warps = 32/32 = 100% ← matches Nsight's 99.75% ✓
```

**The register cliff explained:**
```
At 42 regs/thread → fits 1 block/SM → 100% occupancy    (you are here ✓)
At 64 regs/thread → fits 1 block/SM → 100% occupancy    (still fine)
At 65 regs/thread → block doesn't fit → cliff drop!     (never go here)
```
The graph in Nsight shows 100% flat until 64, then sharp drop — this is why.

**The three occupancy killers:**

| Killer | Symptom in Nsight | Fix |
|--------|-------------------|-----|
| Too many registers | Block Limit Registers = 1 | Use `__launch_bounds__(threads, blocks)` |
| Too much shared memory | Block Limit Shared Mem = 1 | Reduce tile size |
| Block size too small | Achieved << Theoretical | Use larger blocks (256–1024) |

---

### SECTION 6 — Compute Workload / Roofline
**What:** Chart showing if you're near hardware performance limits.

```
Y-axis: Achieved GFLOP/s
X-axis: Arithmetic Intensity (FLOPs per byte of memory traffic)

Two ceilings:
  Compute roofline  = max GFLOP/s of your GPU (diagonal)
  Memory roofline   = max GB/s × your intensity (diagonal, shallower)
```

| Your dot position | Diagnosis | Fix |
|-------------------|-----------|-----|
| Near compute roofline | Compute bound — near peak ✓ | Nothing, you're optimal |
| Near memory roofline | Memory bound | Add tiling, increase arithmetic intensity |
| Far below both | Latency bound | Fix stalls first (warp state section) |

**Arithmetic intensity for matrix multiply:**
```
FLOPs  = 2 × N³
Bytes  = 3 × N² × 4  (load A, B, write C)
Intensity = 2N³ / (12N²) = N/6

For N=512: intensity = 512/6 ≈ 85 FLOP/byte → compute bound
```

**% of peak calculation:**
```
% of peak = Achieved GFLOP/s / (CUDA_cores × clock_Hz × 2) × 100

practice.cu tiledMatMul:       958 / 4027 = 23.8% of peak
register_tiling.cu register2x2: same ~23.8%  (Nsight showed 23%)

Both confirm ~24% of peak. cuBLAS reaches ~85-90%.
Significant headroom remaining → 4×4 register tiling would be next step.
```

---

## Reading the Nsight Summary Tab Columns

When you first open Nsight you see the Summary table with all kernel launches. Here's what each column means:

| Column | What it measures | Ideal | How to read it |
|--------|-----------------|-------|----------------|
| **ID** | Launch sequence number | — | IDs 0–5 = correctness (ignore), 6–11 = benchmark |
| **Estimated Speedup [%]** | How much faster Nsight thinks you COULD be if top bottleneck fixed | 0% | Higher = more room to improve. Not a performance grade. |
| **Duration [ms]** | Actual wall-clock GPU time | As low as possible | **The only metric that actually matters.** Everything else explains this. |
| **Runtime Improvement [ms]** | Absolute time saved if top issue fixed (= Speedup% × Duration) | 0 ms | More useful than % — it's real time |
| **Compute Throughput [%]** | How busy the CUDA math units are vs peak | High, but… | **Misleading alone** — fewer blocks = lower % even if kernel is faster |
| **Memory Throughput [%]** | How busy the memory system is vs peak | Match Compute % | Compare to Compute. Equal = balanced. One much higher = bound by that |
| **# Registers** | Registers used per thread | <64 on sm_75 | Controls occupancy. Above 64 = cliff drop on GTX 1660Ti |

**⚠️ Most important rule for the Summary tab:**
```
NEVER judge a kernel by Compute% or Memory% alone.
Always look at Duration [ms] first.

register 4x4: Compute 38%, Memory 42%  → looks mediocre
              Duration 0.17ms           → actually the FASTEST kernel
```

---

## Register Tiling — The Real Fix for MIO Throttle

**Concept:** Each thread computes more output elements per shared memory load.

```
1 thread → 1 output (baseline):
  k-loop: 2 MIO loads, 1 FMA  →  FMA/MIO = 0.5

1 thread → 2×2 outputs (register tiling):
  k-loop: 4 MIO loads, 4 FMAs →  FMA/MIO = 1.0

1 thread → 4×4 outputs (next level):
  k-loop: 8 MIO loads, 16 FMAs → FMA/MIO = 2.0
```

**Results from register_tiling.cu:**

| Kernel | Time | GFLOP/s | MIO stall |
|--------|------|---------|-----------|
| baseline | 0.537 ms | 500 | 24.7 cycles |
| register 2×2 | **0.280 ms** | **958** | 19.0 cycles |

**Trade-off:** More registers per thread → fewer blocks fit per SM → slightly lower occupancy.
But since the kernel is MIO-bound (not latency-bound), this is acceptable.

**Grid must change with register tiling:**
```
baseline:      grid = (N/TILE_WIDTH)² = (512/16)² = 32×32 = 1024 blocks
register 2×2:  grid = (N/(TILE_WIDTH×THREAD_TILE))² = (512/32)² = 16×16 = 256 blocks

Each block covers more output (32×32 instead of 16×16)
→ fewer blocks needed for same total output
```

---

## Your Personal Profiling Checklist

Run through this for every kernel you write:

```
□ 1. GPU Speed of Light
      → Compute% and Memory% balanced?
      → Both >60%? If not, identify which is lower → that's your bottleneck
      ⚠ Low % doesn't mean slow — check actual ms and GFLOP/s first

□ 2. Memory Workload
      → L2 Hit Rate >85%?
      → DRAM Throughput low?
      → Local Memory Spilling = 0?
      → Shared Memory Spilling = 0?
      → Sectors/Request = 4 for global loads? (coalescing check)

□ 3. Shared Memory
      → Bank Conflicts (Load) = 0?
      → Bank Conflicts (Store) = 0?
      ⚠ Don't "fix" bank conflicts that don't exist — profile first

□ 4. Warp State Statistics  ← MOST IMPORTANT
      → Avg Active Threads Per Warp = 32? (no divergence)
      → Warp Cycles Per Instruction <15?
      → Top stall reason identified?
      → LG Throttle?  → fix coalescing first, then add shared memory tiling
      → MIO Throttle? → use register tiling (NOT float4, NOT pragma unroll)
      → Long Scoreboard? → increase occupancy
      → Barrier? → try double buffering

□ 5. Occupancy
      → Achieved Occupancy >50%?
      → Theoretical = Achieved? (if not, find the limiter)
      → Block Limit — which resource is binding (regs/shared/warps)?
      ⚠ High occupancy ≠ high performance. Only matters if latency bound.

□ 6. Roofline
      → Near compute roofline? (compute-heavy kernels)
      → Near memory roofline? (means you need more data reuse)
      → % of peak? cuBLAS = ~85-90%, hand-written = 20-40% typical
```

---

## Common Problems and Fixes — Quick Lookup

| Problem | Nsight Shows | Fix | What DOESN'T work |
|---------|-------------|-----|-------------------|
| Uncoalesced + no tiling | **LG Throttle >100 cycles**, L1 Throughput ~99% | Fix coalescing (threadIdx.x → column), add shared memory | — |
| DRAM traffic high | L2 Hit Rate <50% | Add shared memory tiling | — |
| Bank conflicts | Bank Conflicts >0 | Pad arrays: `[N][M+1]` | — |
| Warp divergence | Active Threads/Warp <32 | Restructure branches | — |
| Register spilling | Local Memory Spilling >0 | `__launch_bounds__` | — |
| Low occupancy (registers) | Block Limit Registers low | `__launch_bounds__` | — |
| Low occupancy (shared mem) | Block Limit Shared Mem low | Reduce tile size | — |
| **MIO Throttle** | Stall MIO Throttle >20 | **Register tiling (THREAD_TILE)** | `#pragma unroll` alone, float4 loads |
| Long Scoreboard | Stall Long Scoreboard >20 | Increase occupancy | — |
| Barrier stall | Stall Barrier >10 | Double buffering | — |

---

## Misconceptions We Proved Wrong (From Real Profiling)

| Misconception | Reality | Evidence |
|--------------|---------|----------|
| float4 = faster loads | Only if ALL threads load 4 elements. Otherwise warp underutilization hurts more | float4 kernel: 447 vs 543 GFLOP/s |
| Higher occupancy = better performance | Occupancy is latency-hiding, not performance | 89% occupancy → 2× faster than 97% |
| #pragma unroll fixes MIO throttle | Only reduces loop overhead (~3%). MIO op count unchanged | MIO cycles: 24.7 → 24.8 (unchanged) |
| Low throughput % = slow kernel | Fewer blocks → last wave underutilized → % looks low | 46% throughput → 2× faster kernel |
| Transposing B tile reduces MIO throttle | Original tileB[k][tx] already had 0 bank conflicts | No improvement measured |
| Shared memory always has bank conflicts | Standard tiled matmul naturally has 0 conflicts | Confirmed 0 in every kernel |
| Low % of peak = bad optimization | 20-25% is typical for hand-written CUDA matmul | cuBLAS at 85% uses 10+ years of tricks |
| High L1 Throughput = fast kernel | L1 unit saturated by UNCOALESCED requests = slow kernel | sgemmNaive: L1 99% throughput → 1% of peak |
| Shared memory is part of the cache hierarchy | Shared mem is a separate programmer-managed scratchpad inside the SM | ~5 cycles vs L1 ~30 cycles, you control what goes in |
| B access in sgemmNaive is fine (broadcast) | Broadcast is efficient per-warp but B has no temporal reuse across loop iterations | Each of 512 iterations hits a new cache line for B |
| LG Throttle and Long Scoreboard are the same stall | LG = load UNIT congested (too many requests). Long Scoreboard = waiting for one specific load result | sgemmNaive: LG 270 cycles dominant, Long Scoreboard only 20 cycles |

---

## GTX 1660Ti (sm_75) Hardware Limits Reference

```
SMs                    : 24
Max warps per SM       : 32
Max threads per SM     : 1024
Max blocks per SM      : 16
Registers per SM       : 65,536
Max shared mem per SM  : 32 KB (configurable up to 64 KB)
L2 Cache               : 1 MB
Memory bandwidth       : ~192 GB/s
FP32 Peak              : ~4,027 GFLOP/s (at 1.43 GHz boost)
Warp size              : 32 threads
Max threads per block  : 1024
Max block dims         : 1024 × 1024 × 64
FP32 FMAs per SM/clock : 128
```

---

## Profiling Results Log

### register_tiling.cu — All 6 Kernels (N=512, GTX 1660Ti)

| Kernel | Duration | GFLOP/s | Top Stall | Root Cause |
|--------|----------|---------|-----------|------------|
| sgemmNaive | ~4.58 ms | ~30 | LG Throttle ~270 cyc | Uncoalesced A access, no shared mem |
| baselineMatMul | ~0.59 ms | ~230 | MIO Throttle ~25 cyc | FMA/MIO ratio 0.5 |
| registerTiled2x2 | ~0.26 ms | ~500 | MIO Throttle ~19 cyc | FMA/MIO ratio 1.0 |
| registerTiled4x4 | ~0.17 ms | ~760 | MIO lower / reg pressure | FMA/MIO ratio 2.0 |
| registerTiled1D | ~0.26 ms | ~500 | MIO Throttle ~19 cyc | Same as 2x2 |
| doubleBuffer | ~0.26 ms | ~500 | Barrier stall reduced | Software pipelining |

### sgemmNaive — Key Metrics

| Check | Value | Status |
|-------|-------|--------|
| Compute Throughput | 11.71% | ❌ Near idle |
| Memory Throughput | 49.72% | ⚠️ Moderate |
| L1/TEX Throughput | 99.44% | ❌ Saturated (uncoalesced) |
| DRAM Throughput | 2.27% | ✅ Low (data in L2 eventually) |
| Top Stall | LG Throttle ~270 cycles | ❌ Load unit flooded |
| SM Busy | 3.91% | ❌ SM idle 96% of time |
| IPC (instructions/cycle) | 0.034 | ❌ Should be ~4.0 |
| Roofline | 1% of FP32 peak | ❌ 100× from optimal |

### baselineMatMul — Key Metrics

| Check | Value | Status |
|-------|-------|--------|
| Compute / Memory balance | ~73% / ~73% | ✅ Balanced |
| L1 Hit Rate | ~10% | ⚠️ Low (tiles are fresh each block) |
| L2 Hit Rate | 96.58% | ✅ Excellent — tiling is working |
| DRAM Throughput | 2.20% | ✅ Near zero |
| Bank Conflicts | 0 | ✅ Perfect |
| Active Threads/Warp | 32.0 | ✅ No divergence |
| Local Memory Spilling | 0 | ✅ Clean |
| Achieved Occupancy | ~89% | ✅ Good |
| Top Stall | MIO Throttle ~25 cycles | ⚠️ New bottleneck |
| Roofline | ~12% of FP32 peak | ⚠️ Room to grow |

### Occupancy Deep Dive — 2x2 vs 4x4

| Metric | register 2x2 | register 4x4 |
|--------|-------------|-------------|
| Registers/thread | ~62 | ~66 |
| Blocks/SM (register limit) | 4 | 3 |
| Blocks/SM (shared mem limit) | 8 | 4 |
| **Binding constraint** | **Registers** | **Registers** |
| Theoretical Occupancy | 100% | 75% |
| Achieved Occupancy | 88.89% | 55.28% |
| Total blocks (N=512) | 256 | 64 |
| Duration | ~0.26 ms | **~0.17 ms** |

**Why 4x4 is faster despite lower occupancy:**
```
register 2x2: 3 waves of 96 blocks → last wave 67% full → 89% avg occupancy
register 4x4: only 64 blocks total  → doesn't even fill one full wave
              BUT FMA/MIO = 2.0 vs 1.0 → each warp does 2× more per stall cycle
              The extra compute per warp outweighs the occupancy loss
```

**Tail effect (Nsight warns about this):**
```
register 2x2: Est. Speedup 33.33% from tail effect
  → 256 blocks / (24 SMs × 4 blocks) = 2.67 waves → last wave is 67% full
  → if you could eliminate the partial wave, 33% speedup possible

register 4x4: only 64 blocks for 24 SMs → severe tail effect
  → barely fills even one wave across all SMs
```

### practice.cu — tiledMatMul (32×32 tile, 2D block)

| Check | Value | Status |
|-------|-------|--------|
| Compute / Memory balance | 71.56% / 71.56% | ✅ Balanced |
| L2 Hit Rate | 93.94% | ✅ Excellent |
| DRAM Throughput | 2.29% | ✅ Near zero |
| Bank Conflicts | 0 | ✅ Perfect |
| Active Threads/Warp | 32.0 | ✅ No divergence |
| Local Memory Spilling | 0 | ✅ Clean |
| Achieved Occupancy | 99.75% | ✅ Near perfect |
| Top Stall | MIO Throttle 24.7 cycles | ⚠️ Structural bottleneck |
| Roofline | 23% of FP32 peak | ⚠️ Room to grow |

### Optimization History (tiled matmul, 512×512)

```
0. sgemmNaive (no shared mem):       4.58 ms |  30 GFLOP/s  ← LG Throttle
1. baseline (16×16 tile):            0.59 ms | 230 GFLOP/s  ← MIO Throttle
2. + #pragma unroll:                 0.479ms | 560 GFLOP/s  (+3%)
3. + transposed B tile:              0.491ms | 547 GFLOP/s  (~0%)
4. + float4 loads:                   0.600ms | 448 GFLOP/s  (-17% ← hurt!)
5. register 2×2 (16×16 tile):        0.26 ms | 500 GFLOP/s  ← MIO still top stall
6. register 4×4 (16×16 tile):        0.17 ms | 760 GFLOP/s  ← fastest hand-written
```
