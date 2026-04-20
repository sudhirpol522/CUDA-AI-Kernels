// =============================================================================
// flash_attention_explained.cu
// =============================================================================
// READ TOP TO BOTTOM. Every section builds on the previous.
// One concrete 6-token example is traced through EVERY single step so you
// can check each number by hand before reading any GPU code.
// =============================================================================


// =============================================================================
// SECTION 1 — WHAT IS ATTENTION?
// =============================================================================
//
// Each "token" (word, pixel, etc.) in the sequence produces three vectors:
//   Query  q_i  — "what am I looking for?"
//   Key    k_j  — "what do I offer?"
//   Value  v_j  — "what do I actually contain?"
//
// Token i attends to token j with strength = dot(q_i, k_j).
// All those strengths are turned into a probability distribution (softmax),
// then used to build a weighted sum of values.
//
// FULL FORMULA  (one head, no batching):
//
//   Input:   Q [N×d],  K [N×d],  V [N×d]
//
//   Step 1:  S = Q @ K^T              [N×N]   S[i,j] = dot(q_i, k_j)
//   Step 2:  S = S / sqrt(d)          [N×N]   scale to prevent huge dot products
//   Step 3:  P = softmax_rowwise(S)   [N×N]   each row is a probability dist.
//   Step 4:  O = P @ V                [N×d]   O[i] = weighted mix of all values
//
// PIPELINE VISUALIZATION:
//
//   Q [N×d]              K [N×d]               V [N×d]
//   ┌──────────┐         ┌──────────┐           ┌──────────┐
//   │ q0 ──── │         │ k0 ──── │           │ v0 ──── │
//   │ q1 ──── │   K^T   │ k1 ──── │           │ v1 ──── │
//   │ q2 ──── │ ──────► │ k2 ──── │           │ v2 ──── │
//   │ ...     │         │ ...     │           │ ...     │
//   └──────────┘         └──────────┘           └──────────┘
//        │                    │                      │
//        └────── Q @ K^T ─────┘                      │
//                    │                               │
//             S [N×N] (scores)                       │
//             ┌──────────────┐                       │
//             │s00 s01 s02...│ ← each row i =        │
//             │s10 s11 s12...│   [dot(qi,k0),         │
//             │s20 s21 s22...│    dot(qi,k1), ...]    │
//             │ ...          │                       │
//             └──────────────┘                       │
//                    │ ÷ sqrt(d) + softmax per row    │
//             P [N×N] (weights)                      │
//             ┌──────────────┐                       │
//             │.03 .60 .37...│ ← each row sums to 1  │
//             │.10 .25 .65...│                       │
//             │ ...          │                       │
//             └──────────────┘                       │
//                    │                               │
//                    └──────────── P @ V ────────────┘
//                                      │
//                               O [N×d] (output)
//                               ┌──────────┐
//                               │ o0 ──── │  o_i = weighted mix of
//                               │ o1 ──── │         all value rows
//                               │ ...     │
//                               └──────────┘
//
// WHY SCALE BY sqrt(d)?
//   With d=64, each dot product sums 64 products. Variance grows with d.
//   Dividing by sqrt(d) keeps softmax in a well-behaved region (not too
//   "peaky" or too "flat").
//
// =============================================================================
// SECTION 2 — THE MEMORY EXPLOSION PROBLEM
// =============================================================================
//
// The scores matrix S is [N×N].
//   N=1024  → 4 MB         (fine)
//   N=8192  → 256 MB       (per head! 32 heads × batch 32 = 256 GB)
//   N=32768 → 4 GB         (impossible on a single GPU)
//
// FLASH ATTENTION (Dao et al., 2022) avoids storing S at all.
// Key insight: tile Q into blocks of Br rows, tile K,V into blocks of Bc rows,
// and compute the output incrementally using "online softmax".
//
// Memory drops from O(N²) to O(N). Runtime is the same (actually faster due
// to better memory access patterns).
//
// NAIVE vs FLASH ATTENTION  (N=6, Br=2, Bc=2):
//
//  NAIVE: materialise full 6×6 score matrix in memory
//
//         K-cols→  k0  k1  k2  k3  k4  k5
//         Q-row0 [  .   .   .   .   .   .  ]   ← needs ALL 6 scores before softmax
//         Q-row1 [  .   .   .   .   .   .  ]
//         Q-row2 [  .   .   .   .   .   .  ]   6×6 = 36 values in memory
//         Q-row3 [  .   .   .   .   .   .  ]
//         Q-row4 [  .   .   .   .   .   .  ]
//         Q-row5 [  .   .   .   .   .   .  ]
//
//  FLASH: process one (Br×Bc) tile at a time, NEVER store full matrix
//
//         K-cols→ [k0,k1] [k2,k3] [k4,k5]   ← Bc=2 KV-cols per iteration
//                  ───────────────────────
//         [q0,q1] │ tile  │ tile  │ tile  │  ← CUDA block 0 (Br=2 Q-rows)
//                 ├───────┼───────┼───────┤
//         [q2,q3] │ tile  │ tile  │ tile  │  ← CUDA block 1
//                 ├───────┼───────┼───────┤
//         [q4,q5] │ tile  │ tile  │ tile  │  ← CUDA block 2
//                  ───────────────────────
//
//  Each CUDA block handles ONE row of tiles (one Br-chunk of Q).
//  It loops over ALL KV-tiles (left to right), accumulating online.
//  The Br×Bc score tile is computed, used, then DISCARDED.
//  Only the running stats (m, s, O) are kept between iterations.
//
// =============================================================================
// SECTION 3 — THE ONLINE SOFTMAX MATH  (read this first, then the GPU code)
// =============================================================================
//
// PROBLEM: softmax of row i is  exp(s_j) / sum_j(exp(s_j))
//          You need ALL s_j before you can compute the denominator.
//          But in Flash Attention you only see Bc scores at a time.
//
// SOLUTION: Maintain two running statistics per query row:
//   m  = running maximum of scaled scores seen so far
//   s  = running sum of exp(score - m) seen so far
//
// When a new block of scores arrives, the max might increase.
// If it does, ALL previous exponentials were computed relative to the OLD max
// and must be RESCALED to be relative to the NEW max.
//
// THE CORRECTION FACTOR:
//   Old contribution at column j:  exp(score_j - m_old)
//   We want it expressed as:       exp(score_j - m_new)
//
//   exp(score_j - m_new)
//   = exp(score_j - m_old + m_old - m_new)
//   = exp(score_j - m_old) × exp(m_old - m_new)
//   ^^^^^^^^^^^^^^^^^^^^       ^^^^^^^^^^^^^^^^
//   already computed           correction factor c = exp(m_old - m_new)
//
//   Since m_new >= m_old, we have m_old - m_new <= 0, so c is always in (0,1].
//   If m_new == m_old, c = exp(0) = 1 (no rescaling needed).
//   If m_new >  m_old, c < 1 (shrink old values).
//
// =============================================================================
// ONLINE UPDATE RULE — WITH FULL SHAPES AND WORKED EXAMPLE
// =============================================================================
//
// ── PARAMETERS FOR THIS EXAMPLE ──────────────────────────────────────────────
//
//   N  = 6   total tokens in the sequence
//   d  = 2   head dimension  (each Q/K/V row is a 2-element vector)
//   Br = 2   rows of Q per CUDA block  (one block handles 2 query rows)
//   Bc = 2   rows of K,V per KV iteration  (we load 2 K/V rows at a time)
//
//   Number of CUDA blocks = N/Br = 6/2 = 3   (blocks 0,1,2 in the grid)
//   Number of KV iterations per block = N/Bc = 6/2 = 3
//
// ── ALL MATRIX SHAPES ─────────────────────────────────────────────────────────
//
//   Q         [N × d]   = [6 × 2]   full query matrix (global memory)
//   K         [N × d]   = [6 × 2]   full key   matrix (global memory)
//   V         [N × d]   = [6 × 2]   full value matrix (global memory)
//   O         [N × d]   = [6 × 2]   output      (global memory, written at end)
//
//   q_block   [Br × d]  = [2 × 2]   Q tile in shared memory (loaded once)
//   kv_block  [Bc × d]  = [2 × 2]   K^T or V tile in shared memory (reloaded each iter)
//   scores    [Br × Bc] = [2 × 2]   attention score tile (shared memory, scratch)
//
//   Per query row (scalars, also in shared memory):
//     m_prev  [Br]      = [2]        running max, one per query row
//     sum_exp [Br]      = [2]        running sum of exp, one per query row
//     output  [Br × d]  = [2 × 2]   running unnormalised output
//
// ── INPUT DATA ────────────────────────────────────────────────────────────────
//
//   Q [6×2]:          K [6×2]:          V [6×2]:
//   row0: [1, 0]      row0: [1, 0]      row0: [ 1,  0]
//   row1: [0, 1]      row1: [0, 1]      row1: [ 0,  1]
//   row2: [1, 1]      row2: [1, 1]      row2: [ 1,  1]
//   row3: [2, 0]      row3: [2, 0]      row3: [ 2,  0]
//   row4: [0, 2]      row4: [0, 2]      row4: [ 0,  2]
//   row5: [1, 1]      row5: [1, 1]      row5: [ 1,  1]
//
// ── WE FOCUS ON CUDA BLOCK 0 ─────────────────────────────────────────────────
//   This block handles Q rows {0, 1}.
//   q_block [2×2] = Q[0:2] = [[1,0],[0,1]]   loaded once, stays all iterations.
//
//   We trace query row 0 (q = [1,0]) in full detail.
//   query row 1 (q = [0,1]) runs in parallel — same logic, different numbers.
//
// ── INITIAL STATE (before any KV iteration) ───────────────────────────────────
//
//   m_prev  [2]   = [-INF, -INF]     no scores seen yet
//   sum_exp [2]   = [0.0,  0.0]      no exps accumulated
//   output  [2×2] = [[0,0],[0,0]]    no weighted values accumulated
//
// =============================================================================
// KV ITERATION 0  — loads K rows {0,1} then V rows {0,1}
// =============================================================================
//
//   kv_block loaded as K^T [2×2]:         (transposed so Q@K^T works)
//     kv_block = [[K[0][0], K[1][0]],  =  [[1, 0],
//                 [K[0][1], K[1][1]]]       [0, 1]]
//         d_col→   0    1
//    bc_row=0  [  1    0  ]   ← d-feature 0 of K-rows {0,1}
//    bc_row=1  [  0    1  ]   ← d-feature 1 of K-rows {0,1}
//
//   STEP A — scores [Br×Bc] = q_block [2×2] @ kv_block [2×2]
//
//     scores [2×2] before scaling:
//       scores[0,0] = dot(Q[0], K[0]) = 1×1 + 0×0 = 1
//       scores[0,1] = dot(Q[0], K[1]) = 1×0 + 0×1 = 0
//       scores[1,0] = dot(Q[1], K[0]) = 0×1 + 1×0 = 0
//       scores[1,1] = dot(Q[1], K[1]) = 0×0 + 1×1 = 1
//
//     scores [2×2]:        q-row index
//        ↓        k0    k1   ← k-col index (which K-row in this block)
//       q0       [ 1     0  ]
//       q1       [ 0     1  ]
//
//   STEP B — scale scores by inv_sqrt_d = 1/sqrt(2) ≈ 0.707  (in-place)
//
//     scores [2×2]:
//        ↓        k0      k1
//       q0       [0.707   0.000]
//       q1       [0.000   0.707]
//
//   STEP ① — m_local per query row = max across each row of scores
//     m_local[0] = max(0.707, 0.000) = 0.707   (for query row 0)
//     m_local[1] = max(0.000, 0.707) = 0.707   (for query row 1)
//
//   STEP ② — m_new = max(m_prev, m_local)
//     m_new[0] = max(-INF, 0.707) = 0.707
//     m_new[1] = max(-INF, 0.707) = 0.707
//
//   STEP ③ — c = exp(m_prev - m_new)   [scalar per query row]
//     c[0] = exp(-INF - 0.707) = exp(-INF) = 0.000
//     c[1] = exp(-INF - 0.707) = 0.000
//     (first iteration → c=0, old state wiped out — nothing was there anyway)
//
//   kv_block reloaded with V rows {0,1} [2×2]:   (overwrite K^T)
//     kv_block = [[V[0][0], V[0][1]],  = [[1, 0],
//                 [V[1][0], V[1][1]]]      [0, 1]]
//
//   STEP ④ — probs [Br×Bc] = exp(scores - m_new)   (overwrite scores in-place)
//
//     probs [2×2]:
//        ↓       k0                    k1
//       q0      [exp(0.707-0.707)=1.000  exp(0.000-0.707)=0.493]
//       q1      [exp(0.000-0.707)=0.493  exp(0.707-0.707)=1.000]
//
//   STEP ⑤ — new_sum [Br] = row-sum of probs
//     new_sum[0] = 1.000 + 0.493 = 1.493
//     new_sum[1] = 0.493 + 1.000 = 1.493
//
//   STEP ⑥ — sum_exp [Br] = sum_exp × c + new_sum
//     sum_exp[0] = 0.0 × 0.0 + 1.493 = 1.493
//     sum_exp[1] = 0.0 × 0.0 + 1.493 = 1.493
//
//   STEP ⑦part1 — output [2×2] *= c   (rescale old, c=0 so zeroes it)
//     output = [[0,0],[0,0]] × 0 = [[0,0],[0,0]]   (already zero, no change)
//
//   STEP ⑦part2 — output [2×2] += probs [2×2] @ kv_block [2×2]
//
//     probs [2×2] @ V_block [2×2]:
//       out_row0 = 1.000×V[0] + 0.493×V[1] = 1.000×[1,0] + 0.493×[0,1] = [1.000, 0.493]
//       out_row1 = 0.493×V[0] + 1.000×V[1] = 0.493×[1,0] + 1.000×[0,1] = [0.493, 1.000]
//
//     output [2×2] after iteration 0:
//       row0: [1.000, 0.493]
//       row1: [0.493, 1.000]
//
//   STATE after KV iter 0:
//     m_prev  = [0.707,  0.707]
//     sum_exp = [1.493,  1.493]
//     output  = [[1.000, 0.493],
//                [0.493, 1.000]]
//
// =============================================================================
// KV ITERATION 1  — loads K rows {2,3} then V rows {2,3}
// =============================================================================
//
//   kv_block loaded as K^T [2×2]:
//       d-feat→   K[2]   K[3]
//    d_col=0  [   1       2  ]
//    d_col=1  [   1       0  ]
//
//   STEP A — scores [2×2] = q_block @ kv_block:
//     scores[0,0] = dot(Q[0],K[2]) = 1×1+0×1 = 1
//     scores[0,1] = dot(Q[0],K[3]) = 1×2+0×0 = 2
//     scores[1,0] = dot(Q[1],K[2]) = 0×1+1×1 = 1
//     scores[1,1] = dot(Q[1],K[3]) = 0×2+1×0 = 0
//
//     scores [2×2]:
//        ↓        K[2]   K[3]
//       q0       [ 1      2  ]
//       q1       [ 1      0  ]
//
//   STEP B — scale ×0.707:
//     scores [2×2]:
//        ↓        K[2]    K[3]
//       q0       [0.707   1.414]   ← q0 sees a NEW high score: 1.414 > 0.707
//       q1       [0.707   0.000]   ← q1's max stays at 0.707
//
//   STEP ① — m_local:
//     m_local[0] = max(0.707, 1.414) = 1.414
//     m_local[1] = max(0.707, 0.000) = 0.707
//
//   STEP ② — m_new = max(m_prev, m_local):
//     m_new[0] = max(0.707, 1.414) = 1.414   ← INCREASED for q0 !
//     m_new[1] = max(0.707, 0.707) = 0.707   ← unchanged for q1
//
//   STEP ③ — c = exp(m_prev - m_new):
//     c[0] = exp(0.707 - 1.414) = exp(-0.707) = 0.493   ← shrink q0's old output
//     c[1] = exp(0.707 - 0.707) = exp(0)      = 1.000   ← keep  q1's old output
//
//   WHY c IS NEEDED FOR q0:
//     output[0] currently stores: exp(s0-0.707)×V[0] + exp(s1-0.707)×V[1]
//     We need it to store:        exp(s0-1.414)×V[0] + exp(s1-1.414)×V[1]
//     Since exp(s-1.414) = exp(s-0.707) × exp(0.707-1.414) = (stored) × 0.493,
//     just multiply the whole vector by 0.493. No recomputation needed.
//
//   kv_block reloaded with V rows {2,3} [2×2]:
//     [[V[2][0], V[2][1]],  =  [[1, 1],
//      [V[3][0], V[3][1]]]      [2, 0]]
//
//   STEP ④ — probs [2×2] = exp(scores - m_new):
//     probs[0,0] = exp(0.707-1.414) = 0.493    probs[0,1] = exp(1.414-1.414) = 1.000
//     probs[1,0] = exp(0.707-0.707) = 1.000    probs[1,1] = exp(0.000-0.707) = 0.493
//
//     probs [2×2]:
//        ↓        K[2]    K[3]
//       q0       [0.493   1.000]
//       q1       [1.000   0.493]
//
//   STEP ⑤ — new_sum [2]:
//     new_sum[0] = 0.493+1.000 = 1.493
//     new_sum[1] = 1.000+0.493 = 1.493
//
//   STEP ⑥ — sum_exp = sum_exp × c + new_sum:
//     sum_exp[0] = 1.493 × 0.493 + 1.493 = 0.736 + 1.493 = 2.229
//     sum_exp[1] = 1.493 × 1.000 + 1.493 = 1.493 + 1.493 = 2.986
//
//   STEP ⑦part1 — output [2×2] *= c  (each row scaled by its own c):
//     output[0] = [1.000, 0.493] × 0.493 = [0.493, 0.243]   ← q0 rescaled
//     output[1] = [0.493, 1.000] × 1.000 = [0.493, 1.000]   ← q1 unchanged
//
//   STEP ⑦part2 — output += probs [2×2] @ V_block [2×2]:
//     new contribution row0: 0.493×[1,1] + 1.000×[2,0] = [0.493,0.493]+[2,0] = [2.493, 0.493]
//     new contribution row1: 1.000×[1,1] + 0.493×[2,0] = [1,1]+[0.986,0]     = [1.986, 1.000]
//
//     output [2×2] after iteration 1:
//       row0: [0.493+2.493, 0.243+0.493] = [2.986, 0.736]
//       row1: [0.493+1.986, 1.000+1.000] = [2.479, 2.000]
//
//   STATE after KV iter 1:
//     m_prev  = [1.414,  0.707]
//     sum_exp = [2.229,  2.986]
//     output  = [[2.986, 0.736],
//                [2.479, 2.000]]
//
// =============================================================================
// KV ITERATION 2  — loads K rows {4,5} then V rows {4,5}
// =============================================================================
//
//   kv_block loaded as K^T [2×2]:
//       d-feat→   K[4]   K[5]
//    d_col=0  [   0       1  ]
//    d_col=1  [   2       1  ]
//
//   STEP A — scores [2×2] = q_block @ kv_block:
//     scores[0,0] = dot(Q[0],K[4]) = 1×0+0×2 = 0
//     scores[0,1] = dot(Q[0],K[5]) = 1×1+0×1 = 1
//     scores[1,0] = dot(Q[1],K[4]) = 0×0+1×2 = 2
//     scores[1,1] = dot(Q[1],K[5]) = 0×1+1×1 = 1
//
//   STEP B — scale ×0.707:
//     scores [2×2]:
//        ↓        K[4]    K[5]
//       q0       [0.000   0.707]   ← below q0's current max 1.414, no change
//       q1       [1.414   0.707]   ← q1 sees a NEW high score: 1.414 > 0.707
//
//   STEP ① — m_local:
//     m_local[0] = max(0.000, 0.707) = 0.707
//     m_local[1] = max(1.414, 0.707) = 1.414
//
//   STEP ② — m_new = max(m_prev, m_local):
//     m_new[0] = max(1.414, 0.707) = 1.414   ← unchanged for q0
//     m_new[1] = max(0.707, 1.414) = 1.414   ← INCREASED for q1 !
//
//   STEP ③ — c = exp(m_prev - m_new):
//     c[0] = exp(1.414 - 1.414) = 1.000   ← q0's max didn't change, no rescaling
//     c[1] = exp(0.707 - 1.414) = 0.493   ← q1's max grew, shrink old output
//
//   kv_block reloaded with V rows {4,5} [2×2]:
//     [[V[4][0], V[4][1]],  =  [[0, 2],
//      [V[5][0], V[5][1]]]      [1, 1]]
//
//   STEP ④ — probs [2×2]:
//     probs[0,0] = exp(0.000-1.414) = 0.243    probs[0,1] = exp(0.707-1.414) = 0.493
//     probs[1,0] = exp(1.414-1.414) = 1.000    probs[1,1] = exp(0.707-1.414) = 0.493
//
//     probs [2×2]:
//        ↓        K[4]    K[5]
//       q0       [0.243   0.493]
//       q1       [1.000   0.493]
//
//   STEP ⑤ — new_sum [2]:
//     new_sum[0] = 0.243+0.493 = 0.736
//     new_sum[1] = 1.000+0.493 = 1.493
//
//   STEP ⑥ — sum_exp = sum_exp × c + new_sum:
//     sum_exp[0] = 2.229 × 1.000 + 0.736 = 2.965
//     sum_exp[1] = 2.986 × 0.493 + 1.493 = 1.472 + 1.493 = 2.965
//
//   STEP ⑦part1 — output [2×2] *= c:
//     output[0] = [2.986, 0.736] × 1.000 = [2.986, 0.736]   ← q0 unchanged
//     output[1] = [2.479, 2.000] × 0.493 = [1.222, 0.986]   ← q1 rescaled
//
//   STEP ⑦part2 — output += probs [2×2] @ V_block [2×2]:
//     new contribution row0: 0.243×[0,2] + 0.493×[1,1] = [0,0.486]+[0.493,0.493] = [0.493, 0.979]
//     new contribution row1: 1.000×[0,2] + 0.493×[1,1] = [0,2]+[0.493,0.493]     = [0.493, 2.493]
//
//     output [2×2] after iteration 2:
//       row0: [2.986+0.493, 0.736+0.979] = [3.479, 1.715]
//       row1: [1.222+0.493, 0.986+2.493] = [1.715, 3.479]
//
//   STATE after KV iter 2 (all blocks done):
//     m_prev  = [1.414,  1.414]
//     sum_exp = [2.965,  2.965]
//     output  = [[3.479, 1.715],
//                [1.715, 3.479]]
//
// =============================================================================
// EPILOGUE — NORMALISE AND WRITE BACK
// =============================================================================
//
//   output [2×2] /= sum_exp [2]:   (each row divided by its scalar)
//
//     O[0] = [3.479/2.965, 1.715/2.965] = [1.173, 0.578]
//     O[1] = [1.715/2.965, 3.479/2.965] = [0.578, 1.173]
//
//   Write O[0] and O[1] back to global memory O[N×d].
//
// =============================================================================
// VERIFICATION — compare to naive full attention
// =============================================================================
//
//   Naive softmax weights row 0: softmax([0.707, 0, 0.707, 1.414, 0, 0.707])
//     = [0.166, 0.082, 0.166, 0.337, 0.082, 0.166]
//   O[0] = Σ weight[j]×V[j]
//         = 0.166×[1,0]+0.082×[0,1]+0.166×[1,1]+0.337×[2,0]+0.082×[0,2]+0.166×[1,1]
//         = [1.173, 0.578]  ✓  MATCHES
//
//   Naive softmax weights row 1: softmax([0, 0.707, 0.707, 0, 1.414, 0.707])
//     (by symmetry with row 0, result is the transpose)
//   O[1] = [0.578, 1.173]  ✓  MATCHES
//
// =============================================================================
// SUMMARY: what each shape is, when it exists, and where it lives
// =============================================================================
//
//  Shape         Name        Lives in        When it exists
//  ──────────────────────────────────────────────────────────────────────────
//  [6×2]         Q,K,V,O     Global memory   always (input/output)
//  [2×2]=Br×d    q_block     Shared memory   entire kernel lifetime (loaded once)
//  [2×2]=Bc×d    kv_block    Shared memory   ONE KV iteration (overwritten each time)
//  [2×2]=Br×Bc   scores      Shared memory   ONE KV iteration (overwritten each time)
//  [2  ]=Br      m_prev      Shared memory   entire kernel lifetime (updated per iter)
//  [2  ]=Br      sum_exp     Shared memory   entire kernel lifetime (updated per iter)
//  [2×2]=Br×d    output      Shared memory   entire kernel lifetime (updated per iter)
//  ──────────────────────────────────────────────────────────────────────────
//  KEY INSIGHT: only the bottom 3 rows grow with the sequence length N.
//  Everything else is bounded by Br, Bc, d — all compile-time constants.
//  This is why Flash Attention uses O(N) memory instead of O(N²).
//
// =============================================================================
// SECTION 3C — FULL BLOCK MAPPING  (N=9, d=6, Br=3, Bc=3)
// =============================================================================
//
// Parameters for this section:
//   N  = 9   total tokens
//   d  = 6   head dimension  (each Q/K/V row is a 6-element vector)
//   Br = 3   Q-rows per CUDA block
//   Bc = 3   KV-rows per iteration
//   → grid has  N/Br = 3  CUDA blocks   (blocks 0, 1, 2 run in parallel)
//   → each block iterates  N/Bc = 3  KV steps   (steps 0, 1, 2)
//
// ── INPUT SHAPES ─────────────────────────────────────────────────────────────
//
//   Q  [9×6]  (global memory, read-only)
//   K  [9×6]  (global memory, read-only)
//   V  [9×6]  (global memory, read-only)
//   O  [9×6]  (global memory, write-only, output)
//
// ── HOW Q IS PARTITIONED ACROSS CUDA BLOCKS ──────────────────────────────────
//
//   Q [9 rows × 6 cols]:
//   ┌────────────────────────────────────────────┐
//   │ row0: [q00, q01, q02, q03, q04, q05]       │ ← CUDA Block 0  (blockIdx.x=0)
//   │ row1: [q10, q11, q12, q13, q14, q15]       │   q_block = Q[0:3, 0:6]  [3×6]
//   │ row2: [q20, q21, q22, q23, q24, q25]       │   loaded ONCE at startup
//   ├────────────────────────────────────────────┤
//   │ row3: [q30, q31, q32, q33, q34, q35]       │ ← CUDA Block 1  (blockIdx.x=1)
//   │ row4: [q40, q41, q42, q43, q44, q45]       │   q_block = Q[3:6, 0:6]  [3×6]
//   │ row5: [q50, q51, q52, q53, q54, q55]       │   loaded ONCE at startup
//   ├────────────────────────────────────────────┤
//   │ row6: [q60, q61, q62, q63, q64, q65]       │ ← CUDA Block 2  (blockIdx.x=2)
//   │ row7: [q70, q71, q72, q73, q74, q75]       │   q_block = Q[6:9, 0:6]  [3×6]
//   │ row8: [q80, q81, q82, q83, q84, q85]       │   loaded ONCE at startup
//   └────────────────────────────────────────────┘
//
//   IMPORTANT: the Q block NEVER MOVES during the kernel.
//   Each CUDA block loads its own Q slice at startup and keeps it in shared
//   memory for the entire duration.  What "moves" is the KV block — it slides
//   through K and V from left to right (iteration 0 → 1 → 2).
//
// ── THE FULL SCORE MATRIX (NEVER STORED — just for understanding) ─────────────
//
//   Full score matrix S = Q @ K^T  would be [9×9]:
//
//            ← K rows (all 9 tokens) →
//            K[0..2]     K[3..5]     K[6..8]
//           (KV block 0)(KV block 1)(KV block 2)
//           ┌───────────┬───────────┬───────────┐
//   Q[0..2] │  tile A   │  tile B   │  tile C   │  CUDA Block 0 computes these
//  (block 0)│  [3×3]    │  [3×3]    │  [3×3]    │  one tile per KV iteration
//           ├───────────┼───────────┼───────────┤
//   Q[3..5] │  tile D   │  tile E   │  tile F   │  CUDA Block 1 computes these
//  (block 1)│  [3×3]    │  [3×3]    │  [3×3]    │
//           ├───────────┼───────────┼───────────┤
//   Q[6..8] │  tile G   │  tile H   │  tile I   │  CUDA Block 2 computes these
//  (block 2)│  [3×3]    │  [3×3]    │  [3×3]    │
//           └───────────┴───────────┴───────────┘
//
//   Each tile is [Br×Bc] = [3×3].
//   Each CUDA block visits its entire ROW of tiles (e.g. block 0: A → B → C).
//   Tiles across different CUDA-block rows are computed FULLY IN PARALLEL.
//
// ── WHAT EACH CUDA BLOCK DOES STEP BY STEP ───────────────────────────────────
//
//  CUDA BLOCK 0 (blockIdx.x=0)  ←  handles Q rows {0,1,2}
//  ════════════════════════════════════════════════════════
//
//  Startup (once):
//    q_block [3×6] ← Q[0:3, :]      load rows 0,1,2 from global Q
//    output  [3×6] = 0
//    m_prev  [3]   = [-INF,-INF,-INF]
//    sum_exp [3]   = [0, 0, 0]
//
//  ┌─────────────────────────────────────────────────────────────────────────┐
//  │ KV ITERATION 0 — processes K rows {0,1,2}  and  V rows {0,1,2}         │
//  │                                                                         │
//  │  Phase 1: load K[0:3] transposed → kv_block [6×3]                      │
//  │                                                                         │
//  │    kv_block [6×3]  (K^T layout: each column = one K-row):              │
//  │         K[0]  K[1]  K[2]        ← 3 key-vectors as columns             │
//  │    d0 [ k00   k10   k20 ]                                               │
//  │    d1 [ k01   k11   k21 ]                                               │
//  │    d2 [ k02   k12   k22 ]  6 rows = d features                         │
//  │    d3 [ k03   k13   k23 ]                                               │
//  │    d4 [ k04   k14   k24 ]                                               │
//  │    d5 [ k05   k15   k25 ]                                               │
//  │                                                                         │
//  │  scores [3×3] = q_block [3×6]  @  kv_block [6×3]                       │
//  │                                                                         │
//  │    scores:       K[0]  K[1]  K[2]                                       │
//  │    Q[0] row → [ s00   s01   s02 ]  s00 = dot(q0, k0), 6-element dot    │
//  │    Q[1] row → [ s10   s11   s12 ]  s01 = dot(q0, k1), etc.             │
//  │    Q[2] row → [ s20   s21   s22 ]                                       │
//  │                                                                         │
//  │  scale scores ×= 1/sqrt(6) ≈ 0.408   (in-place)                        │
//  │  online softmax → probs [3×3]                                           │
//  │                                                                         │
//  │  Phase 2: reload kv_block ← V[0:3]  [3×6]  (overwrites K^T)            │
//  │                                                                         │
//  │    kv_block [3×6]  (V layout: each row = one V-row):                   │
//  │         d0   d1   d2   d3   d4   d5                                     │
//  │    V[0][ v00  v01  v02  v03  v04  v05 ]                                 │
//  │    V[1][ v10  v11  v12  v13  v14  v15 ]  3 rows = Bc value-vectors     │
//  │    V[2][ v20  v21  v22  v23  v24  v25 ]                                 │
//  │                                                                         │
//  │  output [3×6] += probs [3×3] @ kv_block [3×6]                          │
//  │                                                                         │
//  │    After this:  output[i, :] = Σ_{j=0}^{2} probs[i,j] × V[j, :]       │
//  │    partial result for d-cols 0-5, using only K/V rows 0-2               │
//  └─────────────────────────────────────────────────────────────────────────┘
//         │  carry forward:  m_prev [3],  sum_exp [3],  output [3×6]
//         ▼
//  ┌─────────────────────────────────────────────────────────────────────────┐
//  │ KV ITERATION 1 — processes K rows {3,4,5}  and  V rows {3,4,5}         │
//  │                                                                         │
//  │  scores [3×3] = q_block [3×6] @ K[3:6]^T [6×3]                         │
//  │    (same q_block — still sitting in shared memory from startup)         │
//  │                                                                         │
//  │    scores:       K[3]  K[4]  K[5]                                       │
//  │    Q[0] row → [ s03   s04   s05 ]   ← scores vs K rows 3,4,5           │
//  │    Q[1] row → [ s13   s14   s15 ]                                       │
//  │    Q[2] row → [ s23   s24   s25 ]                                       │
//  │                                                                         │
//  │  online softmax with correction factor c = exp(m_prev - m_new):         │
//  │    if any new score > current max → c < 1 → rescale output              │
//  │    if max unchanged               → c = 1 → output unchanged            │
//  │                                                                         │
//  │  output [3×6] = output×c  +  probs [3×3] @ V[3:6] [3×6]               │
//  └─────────────────────────────────────────────────────────────────────────┘
//         │  carry forward:  m_prev [3],  sum_exp [3],  output [3×6]
//         ▼
//  ┌─────────────────────────────────────────────────────────────────────────┐
//  │ KV ITERATION 2 — processes K rows {6,7,8}  and  V rows {6,7,8}         │
//  │                                                                         │
//  │  scores [3×3] = q_block [3×6] @ K[6:9]^T [6×3]                         │
//  │                                                                         │
//  │    scores:       K[6]  K[7]  K[8]                                       │
//  │    Q[0] row → [ s06   s07   s08 ]   ← ALL 9 scores for Q[0] now seen   │
//  │    Q[1] row → [ s16   s17   s18 ]                                       │
//  │    Q[2] row → [ s26   s27   s28 ]                                       │
//  │                                                                         │
//  │  output [3×6] = output×c  +  probs [3×3] @ V[6:9] [3×6]               │
//  └─────────────────────────────────────────────────────────────────────────┘
//         │
//         ▼  EPILOGUE
//  output [3×6] /= sum_exp [3]    (normalise each row by its scalar)
//  O[0:3, 0:6]  ← output          (write 3 rows × 6 cols to global memory)
//
//  CUDA BLOCK 1 does IDENTICAL steps for Q rows {3,4,5} → O[3:6, :]
//  CUDA BLOCK 2 does IDENTICAL steps for Q rows {6,7,8} → O[6:9, :]
//
// ── OUTPUT ASSEMBLY ──────────────────────────────────────────────────────────
//
//   O [9×6]:
//   ┌────────────────────────────────────────────┐
//   │ row0: [o00, o01, o02, o03, o04, o05]       │  ← written by CUDA Block 0
//   │ row1: [o10, o11, o12, o13, o14, o15]       │
//   │ row2: [o20, o21, o22, o23, o24, o25]       │
//   ├────────────────────────────────────────────┤
//   │ row3: [o30, o31, o32, o33, o34, o35]       │  ← written by CUDA Block 1
//   │ row4: [o40, o41, o42, o43, o44, o45]       │
//   │ row5: [o50, o51, o52, o53, o54, o55]       │
//   ├────────────────────────────────────────────┤
//   │ row6: [o60, o61, o62, o63, o64, o65]       │  ← written by CUDA Block 2
//   │ row7: [o70, o71, o72, o73, o74, o75]       │
//   │ row8: [o80, o81, o82, o83, o84, o85]       │
//   └────────────────────────────────────────────┘
//
//   Each output row i = softmax(scores[i, :]) @ V  where scores[i,:] is the
//   full row of 9 dot-products, but it was NEVER stored all at once.
//
// ── WHAT EACH BLOCK READS FROM K AND V ───────────────────────────────────────
//
//   CUDA Block 0 reads:  CUDA Block 1 reads:  CUDA Block 2 reads:
//   K iter0: K[0:3]      K iter0: K[0:3]      K iter0: K[0:3]
//   V iter0: V[0:3]      V iter0: V[0:3]      V iter0: V[0:3]
//   K iter1: K[3:6]      K iter1: K[3:6]      K iter1: K[3:6]
//   V iter1: V[3:6]      V iter1: V[3:6]      V iter1: V[3:6]
//   K iter2: K[6:9]      K iter2: K[6:9]      K iter2: K[6:9]
//   V iter2: V[6:9]      V iter2: V[6:9]      V iter2: V[6:9]
//
//   ALL 3 blocks read the SAME K and V tiles (just different Q tiles).
//   K and V are fully read-shared across all blocks.
//
// ── SHARED MEMORY CONTENTS AT EACH POINT (Block 0 as example) ───────────────
//
//   Startup:
//   ┌──────────────────┬──────────────────┬──────────────────┐
//   │ q_block [3×6]    │ output [3×6]=0   │ kv_block [3×6]=? │
//   │ Q rows 0,1,2     │                  │ (not yet loaded) │
//   └──────────────────┴──────────────────┴──────────────────┘
//   m_prev=[−∞,−∞,−∞]   sum_exp=[0,0,0]
//
//   After KV iter 0 (K^T loaded, scores computed, V loaded, output updated):
//   ┌──────────────────┬──────────────────┬──────────────────┐
//   │ q_block [3×6]    │ output [3×6]     │ kv_block [3×6]   │
//   │ UNCHANGED        │ partial O[0..2]  │ V[0:3]           │
//   │                  │ from V rows 0-2  │                  │
//   └──────────────────┴──────────────────┴──────────────────┘
//   m_prev=[m0,m1,m2]   sum_exp=[s0,s1,s2]  (partial, seen 3 scores each)
//
//   After KV iter 1:
//   ┌──────────────────┬──────────────────┬──────────────────┐
//   │ q_block [3×6]    │ output [3×6]     │ kv_block [3×6]   │
//   │ UNCHANGED        │ updated, rescaled│ V[3:6]           │
//   └──────────────────┴──────────────────┴──────────────────┘
//   m_prev updated (may be larger)   sum_exp updated
//
//   After KV iter 2 + epilogue:
//   ┌──────────────────┬──────────────────┬──────────────────┐
//   │ q_block [3×6]    │ output [3×6]     │ kv_block [3×6]   │
//   │ UNCHANGED        │ FINAL O[0..2]    │ V[6:9]           │
//   └──────────────────┴──────────────────┴──────────────────┘
//   → write output[3×6] to global O[0:3, :]
//
// ── NUMERICAL EXAMPLE  (d=6, N=6, Br=2, Bc=3 — compact version) ──────────────
//
//   Using N=6, d=6, Br=2 (2 CUDA blocks), Bc=3 (2 KV iterations per block)
//   inv_sqrt_d = 1/sqrt(6) ≈ 0.408
//
//   Q [6×6]:                        V [6×6]:
//   r0: [2, 1, 0, 0, 0, 0]         r0: [10,  0,  0,  0,  0,  0]
//   r1: [0, 0, 2, 1, 0, 0]         r1: [ 0, 10,  0,  0,  0,  0]
//   r2: [0, 0, 0, 0, 2, 1]         r2: [ 0,  0, 10,  0,  0,  0]
//   r3: [1, 2, 0, 0, 0, 0]         r3: [ 0,  0,  0, 10,  0,  0]
//   r4: [0, 0, 1, 2, 0, 0]         r4: [ 0,  0,  0,  0, 10,  0]
//   r5: [0, 0, 0, 0, 1, 2]         r5: [ 0,  0,  0,  0,  0, 10]
//
//   K [6×6]:                        (same as Q for this example)
//   r0: [2, 1, 0, 0, 0, 0]
//   r1: [0, 0, 2, 1, 0, 0]
//   r2: [0, 0, 0, 0, 2, 1]
//   r3: [1, 2, 0, 0, 0, 0]
//   r4: [0, 0, 1, 2, 0, 0]
//   r5: [0, 0, 0, 0, 1, 2]
//
//   CUDA BLOCK 0 handles Q rows {0,1}:
//   q_block [2×6] = [[2,1,0,0,0,0],
//                    [0,0,2,1,0,0]]
//
//   ── KV ITER 0: K rows {0,1,2}, V rows {0,1,2} ──
//
//   K^T used for score matmul [6×3]:
//       K[0]  K[1]  K[2]
//   d0 [  2     0     0  ]
//   d1 [  1     0     0  ]
//   d2 [  0     2     0  ]
//   d3 [  0     1     0  ]
//   d4 [  0     0     2  ]
//   d5 [  0     0     1  ]
//
//   scores [2×3] = q_block [2×6] @ K^T [6×3]:
//     s[0,0]=dot([2,1,0,0,0,0],[2,1,0,0,0,0])=4+1=5   s[0,1]=dot(q0,k1)=0   s[0,2]=0
//     s[1,0]=dot([0,0,2,1,0,0],[2,1,0,0,0,0])=0       s[1,1]=dot(q1,k1)=5   s[1,2]=0
//
//   scores [2×3] (raw):           after ×0.408:
//     Q[0]→ [ 5    0    0 ]         [ 2.041   0.000   0.000 ]
//     Q[1]→ [ 0    5    0 ]         [ 0.000   2.041   0.000 ]
//
//   m_new = [2.041, 2.041]   (max score per query row)
//   c     = [exp(-INF), exp(-INF)] = [0, 0]   (first iter, wipe old state)
//
//   probs [2×3] = exp(scores - m_new):
//     Q[0]→ [exp(0)=1.000  exp(-2.041)=0.130  exp(-2.041)=0.130]
//     Q[1]→ [exp(-2.041)=0.130  exp(0)=1.000  exp(-2.041)=0.130]
//
//   sum_new = [1.260, 1.260]
//   sum_exp = [0×0 + 1.260,  0×0 + 1.260] = [1.260, 1.260]
//
//   V block loaded [3×6]:
//     V[0]: [10,  0,  0,  0,  0,  0]
//     V[1]: [ 0, 10,  0,  0,  0,  0]
//     V[2]: [ 0,  0, 10,  0,  0,  0]
//
//   output [2×6] = 0×0 (rescale) + probs [2×3] @ V_block [3×6]:
//     O[0]= 1.000×[10,0,0,0,0,0]+0.130×[0,10,0,0,0,0]+0.130×[0,0,10,0,0,0,0]
//         = [10.000,  1.300,  1.300,  0.000,  0.000,  0.000]
//     O[1]= 0.130×[10,0,0,0,0,0]+1.000×[0,10,0,0,0,0]+0.130×[0,0,10,0,0,0,0]
//         = [ 1.300, 10.000,  1.300,  0.000,  0.000,  0.000]
//
//   State after iter 0:
//     m_prev  = [2.041, 2.041]
//     sum_exp = [1.260, 1.260]
//     output  = [[10.000,  1.300,  1.300,  0.000,  0.000,  0.000],
//                [ 1.300, 10.000,  1.300,  0.000,  0.000,  0.000]]
//
//   ── KV ITER 1: K rows {3,4,5}, V rows {3,4,5} ──
//
//   scores [2×3] = q_block [2×6] @ K[3:6]^T [6×3]:
//     s[0,0]=dot([2,1,0,0,0,0],[1,2,0,0,0,0])=2+2=4   s[0,1]=0   s[0,2]=0
//     s[1,0]=0   s[1,1]=dot([0,0,2,1,0,0],[0,0,1,2,0,0])=2+2=4   s[1,2]=0
//
//   scores [2×3] (raw):            after ×0.408:
//     Q[0]→ [ 4    0    0 ]          [ 1.633   0.000   0.000 ]
//     Q[1]→ [ 0    4    0 ]          [ 0.000   1.633   0.000 ]
//
//   m_local=[1.633, 1.633]   m_new=max([2.041,2.041],[1.633,1.633])=[2.041,2.041]
//   c = exp([2.041,2.041]-[2.041,2.041]) = [1.000, 1.000]   (max unchanged!)
//
//   probs [2×3]:
//     Q[0]→ [exp(1.633-2.041)=0.665  exp(0-2.041)=0.130  exp(0-2.041)=0.130]
//     Q[1]→ [exp(0-2.041)=0.130      exp(1.633-2.041)=0.665  exp(0-2.041)=0.130]
//
//   sum_new=[0.925, 0.925]
//   sum_exp=[1.260×1+0.925, 1.260×1+0.925]=[2.185, 2.185]
//
//   V block loaded [3×6]:
//     V[3]: [ 0,  0,  0, 10,  0,  0]
//     V[4]: [ 0,  0,  0,  0, 10,  0]
//     V[5]: [ 0,  0,  0,  0,  0, 10]
//
//   output [2×6]:  (rescale by c=1 → no change, then add new P@V)
//     new[0] = 0.665×[0,0,0,10,0,0]+0.130×[0,0,0,0,10,0]+0.130×[0,0,0,0,0,10]
//            = [0.000,  0.000,  0.000,  6.650,  1.300,  1.300]
//     new[1] = 0.130×[0,0,0,10,0,0]+0.665×[0,0,0,0,10,0]+0.130×[0,0,0,0,0,10]
//            = [0.000,  0.000,  0.000,  1.300,  6.650,  1.300]
//
//     output[0] = [10.000,1.300,1.300,0]+[0,0,0,6.650,1.300,1.300]
//              = [10.000, 1.300, 1.300, 6.650, 1.300, 1.300]
//     output[1] = [ 1.300,10.000,1.300,0]+[0,0,0,1.300,6.650,1.300]
//              = [ 1.300,10.000, 1.300, 1.300, 6.650, 1.300]
//
//   ── EPILOGUE ──
//
//   O[0] = output[0] / sum_exp[0] = [10.0,1.3,1.3,6.65,1.3,1.3] / 2.185
//        = [ 4.577,  0.595,  0.595,  3.044,  0.595,  0.595]
//   O[1] = output[1] / sum_exp[1] = [1.3,10.0,1.3,1.3,6.65,1.3] / 2.185
//        = [ 0.595,  4.577,  0.595,  0.595,  3.044,  0.595]
//
//   READING THE RESULT:
//     Q[0] = [2,1,0,0,0,0] is most similar to K[0]=[2,1,...] (score=5)
//     and second most similar to K[3]=[1,2,...] (score=4).
//     So O[0] is a mix heavily weighted toward V[0]=[10,0,...] and V[3]=[0,0,0,10,...].
//     Result [4.577, 0.595, 0.595, 3.044, 0.595, 0.595] reflects exactly that:
//     strong in dimensions 0 and 3, weak in others. ✓
//
//   CUDA BLOCK 1 computes O[2:4] using Q rows {2,3} — same algorithm, different Q.
//
// =============================================================================
// SECTION 4 — FULL NUMERICAL TRACE  (6 tokens, d=2, Bc=2)
// =============================================================================
//
// We focus on QUERY ROW 0 only. All numbers are shown to 3 decimal places.
//
// INPUT MATRICES:
//   Q[0] = [1, 0]
//
//   K = row0:[1,0]  row1:[0,1]  row2:[1,1]  row3:[2,0]  row4:[0,2]  row5:[1,1]
//   V = row0:[1,0]  row1:[0,1]  row2:[1,1]  row3:[2,0]  row4:[0,2]  row5:[1,1]
//
// REFERENCE ANSWER (naive full attention, computed once for verification):
//   raw scores row0 = Q[0] @ K^T = [1,0,1,2,0,1]
//   scaled      = [×0.707] = [0.707, 0.000, 0.707, 1.414, 0.000, 0.707]
//
//   global max = 1.414
//   exp(score - max) = [exp(-0.707), exp(-1.414), exp(-0.707),
//                       exp( 0.000), exp(-1.414), exp(-0.707)]
//                    = [0.493,       0.243,       0.493,
//                       1.000,       0.243,       0.493]
//   sum = 0.493+0.243+0.493+1.000+0.243+0.493 = 2.965
//
//   weights = [0.166, 0.082, 0.166, 0.337, 0.082, 0.166]
//
//   O_ref = sum_j(weights[j] × V[j])
//         = 0.166×[1,0] + 0.082×[0,1] + 0.166×[1,1]
//           + 0.337×[2,0] + 0.082×[0,2] + 0.166×[1,1]
//   O_ref = [1.173, 0.578]           ← TARGET to match
//
// ---------------------------------------------------------------------------
// INITIAL STATE (before any KV block):
//   m_prev = -INF    (no scores seen yet)
//   s_prev = 0.0     (no exponentials accumulated)
//   O_prev = [0, 0]  (no output accumulated)
// ---------------------------------------------------------------------------
//
// ============ KV BLOCK 0: uses K-rows {0,1} and V-rows {0,1} ===============
//
// WHAT WE NEED FROM THE MATRICES:
//   From Q: Q[0] = [1, 0]                   (already in shared memory)
//   From K: K[0]=[1,0], K[1]=[0,1]          (load into kv_block as K^T)
//   From V: V[0]=[1,0], V[1]=[0,1]          (load into kv_block)
//
// STEP A — Compute raw scores (Q[0] @ K[0:2]^T):
//   score[0,0] = dot([1,0], [1,0]) = 1
//   score[0,1] = dot([1,0], [0,1]) = 0
//
// STEP B — Scale by inv_sqrt_d = 1/sqrt(2) ≈ 0.707:
//   scaled[0,0] = 1 × 0.707 = 0.707
//   scaled[0,1] = 0 × 0.707 = 0.000
//
// STEP C — Find max in this block:
//   m_local = max(0.707, 0.000) = 0.707
//   m_new   = max(m_prev, m_local) = max(-INF, 0.707) = 0.707
//
// STEP D — Correction factor  c = exp(m_prev - m_new):
//   c = exp(-INF - 0.707) = exp(-INF) = 0.000
//   (m_prev was -INF so the old accumulated O_prev and s_prev are zeroed out,
//    which is correct — nothing valid was accumulated before.)
//
// STEP E — Compute exp(score - m_new) for this block:
//   p[0] = exp(0.707 - 0.707) = exp( 0.000) = 1.000
//   p[1] = exp(0.000 - 0.707) = exp(-0.707) = 0.493
//   new_sum = 1.000 + 0.493 = 1.493
//
// STEP F — Update statistics:
//   s_new = s_prev × c + new_sum = 0.000 × 0 + 1.493 = 1.493
//   m_new = 0.707
//
// STEP G — Rescale old output and add new P@V:
//   O_new = O_prev × c   +   p[0]×V[0]   +   p[1]×V[1]
//         = [0,0]  × 0   +   1.000×[1,0] +   0.493×[0,1]
//         = [0, 0]        +   [1.000, 0]  +   [0, 0.493]
//         = [1.000, 0.493]
//
// STATE AFTER BLOCK 0:
//   m_prev = 0.707
//   s_prev = 1.493
//   O_prev = [1.000, 0.493]   ← represents 1.0×V[0] + 0.493×V[1]  (unnormalised)
//
// ============ KV BLOCK 1: uses K-rows {2,3} and V-rows {2,3} ===============
//
// WHAT WE NEED FROM THE MATRICES:
//   From Q: Q[0] = [1, 0]                   (still in shared memory)
//   From K: K[2]=[1,1], K[3]=[2,0]          (new load into kv_block as K^T)
//   From V: V[2]=[1,1], V[3]=[2,0]          (new load into kv_block)
//
// STEP A — Compute raw scores:
//   score[0,0] = dot([1,0], [1,1]) = 1
//   score[0,1] = dot([1,0], [2,0]) = 2
//
// STEP B — Scale by 0.707:
//   scaled[0,0] = 0.707
//   scaled[0,1] = 1.414
//
// STEP C — Find max in this block:
//   m_local = max(0.707, 1.414) = 1.414
//   m_new   = max(m_prev=0.707, m_local=1.414) = 1.414
//
//   **m_new (1.414) > m_prev (0.707)  ← THE IMPORTANT CASE!**
//   The old accumulated output was computed relative to 0.707.
//   We now have a larger max, so all old exponentials are "wrong" — they are
//   too large relative to the new scale. We must shrink them.
//
// STEP D — Correction factor  c = exp(m_prev - m_new):
//   c = exp(0.707 - 1.414) = exp(-0.707) ≈ 0.493
//
//   WHY THIS WORKS:
//     Old: O_prev stores sum_j [ exp(score_j - 0.707) × V[j] ]
//     We want: sum_j [ exp(score_j - 1.414) × V[j] ]
//     Relationship: exp(score_j - 1.414) = exp(score_j - 0.707) × exp(0.707 - 1.414)
//                                                                   ^^^^^^^^^^^^^^^^
//                                                                    = c = 0.493
//     So: O_prev × c  already equals what we want.
//     Same logic applies to s_prev.
//
// STEP E — Compute exp(score - m_new) for this block:
//   p[0] = exp(0.707 - 1.414) = exp(-0.707) = 0.493
//   p[1] = exp(1.414 - 1.414) = exp( 0.000) = 1.000
//   new_sum = 0.493 + 1.000 = 1.493
//
// STEP F — Update statistics:
//   s_new = s_prev × c + new_sum = 1.493 × 0.493 + 1.493 = 0.736 + 1.493 = 2.229
//   m_new = 1.414
//
// STEP G — Rescale old output and add new P@V:
//   O_new = O_prev × c          +  p[0]×V[2]      +  p[1]×V[3]
//         = [1.000, 0.493]×0.493 +  0.493×[1,1]    +  1.000×[2,0]
//         = [0.493, 0.243]       +  [0.493, 0.493]  +  [2.000, 0]
//         = [2.986, 0.736]
//
// STATE AFTER BLOCK 1:
//   m_prev = 1.414
//   s_prev = 2.229
//   O_prev = [2.986, 0.736]
//
// ============ KV BLOCK 2: uses K-rows {4,5} and V-rows {4,5} ===============
//
// WHAT WE NEED FROM THE MATRICES:
//   From Q: Q[0] = [1, 0]                   (still in shared memory)
//   From K: K[4]=[0,2], K[5]=[1,1]          (new load)
//   From V: V[4]=[0,2], V[5]=[1,1]          (new load)
//
// STEP A — Compute raw scores:
//   score[0,0] = dot([1,0], [0,2]) = 0
//   score[0,1] = dot([1,0], [1,1]) = 1
//
// STEP B — Scale by 0.707:
//   scaled[0,0] = 0.000
//   scaled[0,1] = 0.707
//
// STEP C — Find max in this block:
//   m_local = max(0.000, 0.707) = 0.707
//   m_new   = max(m_prev=1.414, m_local=0.707) = 1.414
//
//   **m_new == m_prev  ← NO CHANGE in max!**
//   Correction c = exp(1.414 - 1.414) = exp(0) = 1.0   (no rescaling needed)
//
// STEP D — Correction factor:
//   c = 1.000   ← identity, old output is already on the right scale
//
// STEP E — Compute exp(score - m_new) for this block:
//   p[0] = exp(0.000 - 1.414) = exp(-1.414) = 0.243
//   p[1] = exp(0.707 - 1.414) = exp(-0.707) = 0.493
//   new_sum = 0.243 + 0.493 = 0.736
//
// STEP F — Update statistics:
//   s_new = s_prev × c + new_sum = 2.229 × 1.0 + 0.736 = 2.965
//   m_new = 1.414
//
// STEP G — Rescale old output and add new P@V:
//   O_new = O_prev × c          + p[0]×V[4]     + p[1]×V[5]
//         = [2.986, 0.736]×1.0  + 0.243×[0,2]   + 0.493×[1,1]
//         = [2.986, 0.736]       + [0,    0.486]  + [0.493, 0.493]
//         = [3.479, 1.715]
//
// STATE AFTER BLOCK 2:
//   m_prev = 1.414
//   s_prev = 2.965
//   O_prev = [3.479, 1.715]
//
// ============ EPILOGUE — FINAL NORMALISATION ================================
//
//   O_final = O_prev / s_prev = [3.479/2.965,  1.715/2.965]
//                              = [1.173,        0.578]
//
//   Reference:                 = [1.173,        0.578]  ✓  MATCHES!
//
// ---------------------------------------------------------------------------
// RUNNING STATE SUMMARY TABLE  (query row 0 across all 3 KV blocks)
//
//              m_prev    s_prev    O_prev              c (correction)
//  ─────────────────────────────────────────────────────────────────────────
//  Initial   │  -INF  │  0.000  │ [0.000, 0.000]  │    -
//  ─────────────────────────────────────────────────────────────────────────
//  Block 0   │ 0.707  │  1.493  │ [1.000, 0.493]  │  0.000  (m grew from -INF)
//  (K/V 0,1) │        │         │                 │  old state wiped out ✓
//  ─────────────────────────────────────────────────────────────────────────
//  Block 1   │ 1.414  │  2.229  │ [2.986, 0.736]  │  0.493  (m grew 0.707→1.414)
//  (K/V 2,3) │        │         │                 │  old values shrunk ✓
//  ─────────────────────────────────────────────────────────────────────────
//  Block 2   │ 1.414  │  2.965  │ [3.479, 1.715]  │  1.000  (m unchanged)
//  (K/V 4,5) │        │         │                 │  no rescaling needed ✓
//  ─────────────────────────────────────────────────────────────────────────
//  Epilogue  │    -   │    -    │ [1.173, 0.578]  │  ÷ 2.965  (normalise)
//  ─────────────────────────────────────────────────────────────────────────
//
// PHYSICAL MEANING OF EACH COLUMN:
//   m_prev  = the largest attention score seen so far (for numerical safety)
//   s_prev  = sum of exp(all scores seen - m_prev)  = denominator of softmax
//   O_prev  = sum of exp(score_j - m_prev) × V[j]  = unnormalised numerator
//   c       = exp(m_old - m_new), the factor that "re-bases" old values
//
// =============================================================================
// SECTION 5 — WHAT EACH VARIABLE STORES AT EVERY MOMENT
// =============================================================================
//
//  Variable       | Stores (physically)
//  ---------------|-----------------------------------------------------------------
//  q_block        | Q rows for this CUDA block. Loaded ONCE. Never changes.
//                 | Used at every KV iteration to compute new scores.
//  ---------------|-----------------------------------------------------------------
//  kv_block       | DUAL PURPOSE buffer, reused each KV iteration:
//                 |   Phase 1 — holds K[block]^T  (so Q @ kv_block = scores)
//                 |   Phase 2 — holds V[block]    (so probs @ kv_block = new O)
//  ---------------|-----------------------------------------------------------------
//  scores         | After matmul:        raw dot products Q@K^T for this block
//                 | After scaling:       scores × inv_sqrt_d
//                 | After online softmax: exp(score - m_new)  = probabilities
//                 |   (these are NOT normalised yet — we haven't divided by s yet)
//  ---------------|-----------------------------------------------------------------
//  m_prev / m_curr| Running maximum of ALL scaled scores seen for each query row.
//                 | max_prev = max after all previous KV blocks
//                 | max_curr = max being computed for the current KV block
//                 | After step F:  max_curr becomes max_prev for next iteration
//  ---------------|-----------------------------------------------------------------
//  sum_exp        | Running sum of exp(score - m_global) for each query row.
//                 | = denominator of softmax over all KV blocks seen so far.
//                 | Always relative to the CURRENT m (rescaled whenever m grows).
//  ---------------|-----------------------------------------------------------------
//  output         | Running unnormalised output O.
//                 | = sum_{j seen so far} exp(score_j - m) × V[j]
//                 | = numerator of the softmax-weighted sum.
//                 | Rescaled by correction c each time m grows.
//  ---------------|-----------------------------------------------------------------
//
// =============================================================================
// SECTION 6 — THE P@V STEP  (why it uses add_to_output=true)
// =============================================================================
//
// After online_softmax_and_accum_output runs:
//   scores[Br × Bc] = probabilities (exp(score - m_new), NOT normalised yet)
//   output[Br × d]  = already rescaled by correction factor
//
// We need:  output += scores @ values
//
//   scores  [Br × Bc]:  probability weights for this block
//   values  [Bc × d]:   V rows for this block
//   output  [Br × d]:   accumulator (add_to_output = true, so += not =)
//
// For query row 0, KV block 1 in our example:
//   scores row 0 = [0.493, 1.000]   (p[0], p[1])
//   values       = [[1,1],           (V[2])
//                   [2,0]]           (V[3])
//
//   scores[0] @ values = 0.493×[1,1] + 1.000×[2,0]
//                      = [0.493, 0.493] + [2.000, 0]
//                      = [2.493, 0.493]
//
//   output was [0.493, 0.243] (after rescaling O_prev by c=0.493)
//   output += [2.493, 0.493]
//   output = [2.986, 0.736]   ✓  matches Step G result above
//
// =============================================================================
// SECTION 7 — WARP PARALLELISM
// =============================================================================
//
// Template parameters:
//   Br   = block rows for Q  (e.g. 32)     one CUDA block handles Br Q-rows
//   Bc   = block cols for KV (e.g. 32)     one KV iteration handles Bc KV-rows
//   THREADS = Br threads per block (e.g. 8 warps × 32 = 256)
//   Wr   = Q rows each warp owns   (e.g. 4)
//   Lc   = columns each lane owns  (e.g. 4)
//
// static_assert: Br == WARPS_PER_BLOCK × Wr  (every Q row owned by exactly one warp)
//
// GRID / BLOCK MAP  (N=8, Br=4, Bc=4, Wr=2, 2 warps shown for clarity):
//
//   GPU GRID (one block per Br-chunk of Q rows):
//
//   ┌─────────────────────────┐   ┌─────────────────────────┐
//   │     CUDA Block 0        │   │     CUDA Block 1        │
//   │   handles Q rows 0-3    │   │   handles Q rows 4-7    │
//   │   blockIdx.x = 0        │   │   blockIdx.x = 1        │
//   └─────────────────────────┘   └─────────────────────────┘
//
//   INSIDE BLOCK 0  (256 threads, 8 warps, Wr=2 rows per warp):
//
//   Q rows in this block:     d=4 columns
//                          col0 col1 col2 col3
//   row0 ┐                [  .    .    .    .  ]  ← owned by WARP 0
//   row1 ┘ Wr=2 rows      [  .    .    .    .  ]    (threads 0-31)
//   row2 ┐                [  .    .    .    .  ]  ← owned by WARP 1
//   row3 ┘                [  .    .    .    .  ]    (threads 32-63)
//
//   INSIDE WARP 0  (32 lanes, Lc=2 columns per lane):
//
//   Lane assignments across d=4 columns:
//   lane_id:   0    1    2    3    4    5  ...  31
//   col owned: 0,1  2,3  0,1  2,3  0,1  2,3 ...   (stride = 32*Lc = 64)
//              ↑Lc=2 cols each
//
//   For d=4:  lane0 → cols {0,1},  lane1 → cols {2,3}
//             lanes 2-31 → no more columns (d=4 is small here)
//
//   REGISTER TILE per lane  (Wr=2, Lc=2):
//
//         col_start  col_start+1
//   row0 [ acc[0]      acc[1]   ]   ← these 4 floats live in REGISTERS
//   row1 [ acc[2]      acc[3]   ]     never touch shared memory during matmul
//
// WARP ASSIGNMENT:
//   warp_id = threadIdx.x / 32
//   lane_id = threadIdx.x % 32
//
//   Warp 0 owns Q rows:  0, 1, ..., Wr-1         (and the same rows of output)
//   Warp 1 owns Q rows:  Wr, ..., 2*Wr-1
//   ...
//   Warp 7 owns Q rows:  7*Wr, ..., Br-1
//
// LANE ASSIGNMENT (within a warp):
//   Lane 0  owns columns: 0, Lc, 2*Lc, ...
//   Lane 1  owns columns: 1, Lc+1, ...
//   ...
//   Lane 31 owns columns: 31, Lc+31, ...
//
// WHY THIS MAPPING?
//   - Each warp handles a CONTIGUOUS row range of Q → good L1 reuse of K,V
//   - Each lane handles STRIDED columns → all 32 lanes hit consecutive addresses
//     simultaneously → coalesced global memory access
//
// REGISTER TILE (per lane):
//   acc[Wr × Lc] lives in registers — no shared memory traffic during multiply.
//   This is the "register tiling" trick that makes the matmul fast.
//
// =============================================================================
// SECTION 8 — WARP-LEVEL REDUCTION WITH __shfl_xor_sync
// =============================================================================
//
// After each lane computes a PARTIAL max/sum over its own column slice,
// all 32 lanes in the warp must agree on the TRUE row-wide max/sum.
//
// __shfl_xor_sync(mask, val, shift):
//   Lane X exchanges its value with lane (X XOR shift).
//   Both lanes take the max (or sum) of the two values.
//
// BUTTERFLY REDUCTION (8 lanes shown, same pattern repeats for 32):
//
//   Initial partial values per lane:
//   Lane:  0     1     2     3     4     5     6     7
//        [3.0] [1.0] [4.0] [2.0] [2.5] [3.5] [1.5] [4.5]
//
//   shift=4 (XOR with 4, so lane X talks to lane X^4):
//   0↔4   1↔5   2↔6   3↔7
//   max(3.0,2.5) max(1.0,3.5) max(4.0,1.5) max(2.0,4.5)
//        │            │            │            │
//       3.0          3.5          4.0          4.5          (same at partner)
//   Lane:  0     1     2     3     4     5     6     7
//        [3.0] [3.5] [4.0] [4.5] [3.0] [3.5] [4.0] [4.5]
//
//   shift=2 (lane X talks to lane X^2):
//   0↔2   1↔3   4↔6   5↔7
//   max(3.0,4.0) max(3.5,4.5) max(3.0,4.0) max(3.5,4.5)
//        │            │            │            │
//       4.0          4.5          4.0          4.5
//   Lane:  0     1     2     3     4     5     6     7
//        [4.0] [4.5] [4.0] [4.5] [4.0] [4.5] [4.0] [4.5]
//
//   shift=1 (lane X talks to lane X^1):
//   0↔1   2↔3   4↔5   6↔7
//   max(4.0,4.5) on all
//        │
//       4.5
//   Lane:  0     1     2     3     4     5     6     7
//        [4.5] [4.5] [4.5] [4.5] [4.5] [4.5] [4.5] [4.5]
//                                                      ↑ TRUE MAX, all lanes agree!
//
// KEY: all values flow through REGISTERS only — no shared memory writes.
//      For 32 lanes: 5 rounds (shifts 16,8,4,2,1). ~5 cycles vs ~64 for shmem tree.
//
// No shared memory needed. This is faster than a tree reduction in shmem.
//
// =============================================================================
// SECTION 9 — THE FULL KERNEL ANNOTATED
// =============================================================================

#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>

#define PAD              1
#define THREADS_PER_WARP 32
#define WARPS_PER_BLOCK  8
#define FULL_MASK        0xffffffff

// ---------------------------------------------------------------------------
// matmul_warp_tiled: C = A @ B  (or C += A @ B when add_to_output=true)
//
// Key facts:
//   • Each warp computes Wr rows of C, each lane computes Lc columns of C.
//   • acc[Wr*Lc] is a register array — never touches shared memory.
//   • Outer loop: K dimension. Inner: outer-product update of acc tile.
//
// TILE OWNERSHIP MAP  (Wr=2, Lc=2, 2 warps, 4 lanes shown):
//
//   C [M × N]:
//            col0  col1  col2  col3  col4  col5  col6  col7
//   row0  ┐ [w0L0  w0L0][w0L1  w0L1][w0L2  w0L2][w0L3  w0L3]   Warp 0
//   row1  ┘ [w0L0  w0L0][w0L1  w0L1][w0L2  w0L2][w0L3  w0L3]   (Wr=2 rows)
//   row2  ┐ [w1L0  w1L0][w1L1  w1L1][w1L2  w1L2][w1L3  w1L3]   Warp 1
//   row3  ┘ [w1L0  w1L0][w1L1  w1L1][w1L2  w1L2][w1L3  w1L3]
//            └─────────┘             Lc=2 cols per lane
//
//   Lane 0 of Warp 0 owns the top-left 2×2 block and accumulates:
//   acc[0] = C[row0, col0],  acc[1] = C[row0, col1]
//   acc[2] = C[row1, col0],  acc[3] = C[row1, col1]
//
// OUTER PRODUCT UPDATE per k_idx step:
//
//   a_vals = [ A[row0, k_idx] ]   (Wr=2 values from A)
//            [ A[row1, k_idx] ]
//
//   b_vals = [ B[k_idx, col0]  B[k_idx, col1] ]   (Lc=2 values from B)
//
//   outer product:
//   acc[0] += a_vals[0] * b_vals[0]   acc[1] += a_vals[0] * b_vals[1]
//   acc[2] += a_vals[1] * b_vals[0]   acc[3] += a_vals[1] * b_vals[1]
//
//   After K iterations of this, acc holds the exact Wr×Lc sub-tile of C.
// ---------------------------------------------------------------------------
template <bool add_to_output = false, int THREADS, int Wr, int Lc>
__device__ __forceinline__ void matmul_warp_tiled(
    const float* A, const float* B, float* C,
    int M, int N, int K,
    int stride_A = 0, int stride_B = 0, int stride_C = 0
) {
    if (stride_A == 0) stride_A = K;
    if (stride_B == 0) stride_B = N;
    if (stride_C == 0) stride_C = N;

    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;

    // Each warp processes Wr rows, striding by WARPS_PER_BLOCK*Wr
    for (int row_start = Wr * warp_id; row_start < M; row_start += WARPS_PER_BLOCK * Wr) {
        // Each lane processes Lc columns, striding by 32*Lc
        for (int col_start = Lc * lane_id; col_start < N; col_start += THREADS_PER_WARP * Lc) {

            // Register accumulator for this Wr×Lc tile
            // This is the "output tile" being computed by this lane
            float acc[Wr * Lc];
            #pragma unroll
            for (int i = 0; i < Wr * Lc; i++) acc[i] = 0.0f;

            // Walk the K dimension one step at a time
            // Each step: load Wr values from A, Lc values from B, outer-product
            for (int k_idx = 0; k_idx < K; k_idx++) {

                // Load a_vals: one element per A-row that this warp owns
                // a_vals[i] = A[row_start+i, k_idx]
                float a_vals[Wr];
                #pragma unroll
                for (int i = 0; i < Wr; i++)
                    a_vals[i] = A[(row_start + i) * stride_A + k_idx];

                // Load b_vals: one element per B-column that this lane owns
                // b_vals[j] = B[k_idx, col_start+j]
                // NOTE: declared as [Wr] but used as [Lc]; requires Wr == Lc.
                float b_vals[Wr];
                #pragma unroll
                for (int j = 0; j < Wr; j++)
                    b_vals[j] = B[k_idx * stride_B + col_start + j];

                // Outer product: acc[i,j] += a_vals[i] * b_vals[j]
                // This accumulates partial dot products across K
                #pragma unroll
                for (int i = 0; i < Wr; i++)
                    #pragma unroll
                    for (int j = 0; j < Lc; j++)
                        acc[i * Lc + j] += a_vals[i] * b_vals[j];
            }

            // Write register tile to shared/global C
            #pragma unroll
            for (int i = 0; i < Wr; i++)
                #pragma unroll
                for (int j = 0; j < Lc; j++) {
                    if constexpr (add_to_output)
                        C[(row_start + i) * stride_C + (col_start + j)] += acc[i * Lc + j];
                    else
                        C[(row_start + i) * stride_C + (col_start + j)]  = acc[i * Lc + j];
                }
        }
    }
}

// ---------------------------------------------------------------------------
// online_softmax_and_accum_output
//
// What enters:
//   scores[Br × (Bc+PAD)] = raw Q@K^T values (will be overwritten with probs)
//   max_prev[Br]           = m from ALL previous KV blocks
//   sum_exp[Br]            = s from ALL previous KV blocks
//   output[Br × d]         = unnormalised O from all previous KV blocks
//   values[Bc × d]         = V for THIS KV block
//
// What exits:
//   scores    = probabilities (exp(score - m_new))
//   max_cur   = updated max (max of m_prev and this block's scores)
//   sum_exp   = s_prev * correction + this block's sum
//   output    = output * correction + scores @ values
// ---------------------------------------------------------------------------
template<int Br, int Bc, int THREADS, int Wr, int Lc, int d>
__device__ __forceinline__ void online_softmax_and_accum_output(
    float* max_cur, const float* max_prev, float* sum_exp,
    float* scores, float* output, const float* values, float inv_sqrt_d
) {
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;

    for (int row_start = Wr * warp_id; row_start < Br; row_start += WARPS_PER_BLOCK * Wr) {

        // Each register holds one value per Q-row owned by this warp
        float max_new[Wr];      // new global max (max_prev combined with this block)
        float sum_new[Wr];      // sum of exp for THIS block only
        float exp_max_diff[Wr]; // correction factor c = exp(m_prev - m_new)

        #pragma unroll
        for (int i = 0; i < Wr; i++) {
            max_new[i] = max_prev[row_start + i]; // start with inherited max
            sum_new[i] = 0.0f;
        }

        // SCORES BUFFER TRANSFORMATION (what's in scores[] at each sub-step):
        //
        //  After matmul (entering this function):
        //  scores[] = [ raw_dot(q0,k0)  raw_dot(q0,k1) ... ]   (one row shown)
        //           = [ 1.000           2.000           ... ]
        //
        //  After STEP B (scale by inv_sqrt_d):
        //  scores[] = [ 0.707           1.414           ... ]
        //
        //  After STEP E (exp(score - m_new)):
        //  scores[] = [ exp(-0.707)=0.493  exp(0)=1.000  ... ]   ← "probs" (unnorm)
        //
        //  These probs feed directly into the P@V matmul as the "A" matrix.
        //  They are NOT normalised — division by sum_exp happens in the epilogue.
        //
        // ---- STEPS A+B+C : scale scores and find block max ----
        // Each lane scans its column stripe, scales in-place, tracks max.
        for (int col_start = Lc * lane_id; col_start < Bc; col_start += THREADS_PER_WARP * Lc) {
            #pragma unroll
            for (int i = 0; i < Wr; i++) {
                #pragma unroll
                for (int j = 0; j < Lc; j++) {
                    int col_idx = col_start + j;
                    if (col_idx < Bc) {
                        // Scale and store back (overwrite raw score with scaled score)
                        float s = scores[(row_start + i) * (Bc + PAD) + col_idx] * inv_sqrt_d;
                        scores[(row_start + i) * (Bc + PAD) + col_idx] = s;
                        max_new[i] = fmaxf(max_new[i], s);
                    }
                }
            }
        }

        // Warp-wide reduce: every lane has partial max → combine to global max.
        // After this loop ALL 32 lanes hold the same max_new[i].
        #pragma unroll
        for (int i = 0; i < Wr; i++)
            #pragma unroll
            for (int shift = THREADS_PER_WARP / 2; shift >= 1; shift >>= 1)
                max_new[i] = fmaxf(max_new[i], __shfl_xor_sync(FULL_MASK, max_new[i], shift));

        // ---- STEPS D+E : compute probabilities and sum ----
        // Now that we know m_new, compute exp(score - m_new) for every score.
        // Also accumulate partial sum into sum_new.
        for (int col_start = Lc * lane_id; col_start < Bc; col_start += THREADS_PER_WARP * Lc) {
            #pragma unroll
            for (int i = 0; i < Wr; i++) {
                #pragma unroll
                for (int j = 0; j < Lc; j++) {
                    int col_idx = col_start + j;
                    if (col_idx < Bc) {
                        // scores currently holds scaled raw score; overwrite with exp
                        float prob = expf(scores[(row_start + i) * (Bc + PAD) + col_idx] - max_new[i]);
                        scores[(row_start + i) * (Bc + PAD) + col_idx] = prob;
                        sum_new[i] += prob; // accumulate partial sum
                    }
                }
            }
        }

        // Warp-wide reduce: sum across all lanes
        #pragma unroll
        for (int i = 0; i < Wr; i++)
            #pragma unroll
            for (int shift = THREADS_PER_WARP / 2; shift >= 1; shift >>= 1)
                sum_new[i] += __shfl_xor_sync(FULL_MASK, sum_new[i], shift);

        // ---- STEP F : compute correction factor and update statistics ----
        // c = exp(m_old - m_new)
        //   If m_new == m_old: c=1 (no rescaling)
        //   If m_new >  m_old: c<1 (shrink old accumulated values)
        #pragma unroll
        for (int i = 0; i < Wr; i++) {
            exp_max_diff[i] = expf(max_prev[row_start + i] - max_new[i]);
            if (lane_id == 0) {
                // Update running max
                max_cur[row_start + i]  = max_new[i];
                // New sum = old_sum × c  +  this block's sum
                //           ^^^^^^^^^^^        ^^^^^^^^^^^^
                //           rescales old exps  new exps already at m_new scale
                sum_exp[row_start + i]  = exp_max_diff[i] * sum_exp[row_start + i]
                                         + sum_new[i];
            }
        }
        __syncthreads(); // make sum_exp/max_cur visible to all threads

        // ---- STEP G (part 1) : rescale output by correction factor ----
        // output currently holds: sum_{prev j} exp(score_j - m_old) × V[j]
        // After ×c:               sum_{prev j} exp(score_j - m_new) × V[j]
        // (Same identity: exp(score_j-m_old) × exp(m_old-m_new) = exp(score_j-m_new))
        for (int d_idx = lane_id; d_idx < d; d_idx += THREADS_PER_WARP)
            #pragma unroll
            for (int i = 0; i < Wr; i++)
                output[(row_start + i) * d + d_idx] *= exp_max_diff[i];
    }

    __syncthreads(); // ensure all warps finished rescaling before matmul

    // ---- STEP G (part 2) : output += probs @ V ----
    // add_to_output=true means output += (not output =)
    // This adds the new block's weighted-value contribution.
    matmul_warp_tiled<true, THREADS, Wr, Lc>(
        scores, values, output,
        Br, d, Bc,
        Bc + PAD,  // stride for scores rows (padded)
        d,         // stride for values rows
        d          // stride for output rows
    );
}

// ---------------------------------------------------------------------------
// fa_kernel — the main kernel
//
// Grid:  one block per Br-chunk of Q rows
//        blockIdx.x = 0 → Q rows [0, Br)
//        blockIdx.x = 1 → Q rows [Br, 2Br)
//        ...
// Block: THREADS threads (flat 1D)
// ---------------------------------------------------------------------------
template<int Br, int Bc, int THREADS, int d, int Wr, int Lc>
__global__ void fa_kernel(
    const float* Q, const float* K, const float* V, float* O,
    int N, float scale, float inv_sqrt_d
) {
    static_assert(Br == WARPS_PER_BLOCK * Wr, "Br must equal WARPS*Wr");

    const int q_block_idx = blockIdx.x;
    const int warp_id     = threadIdx.x / 32;
    const int lane_id     = threadIdx.x % 32;

    // Shared memory layout (all pointers into one extern buffer):
    //
    // ┌──────────┬──────────┬────────────────┬──────────────┬─────────┬──────────┬──────────┐
    // │ output   │ q_block  │ kv_block        │ scores       │ sum_exp │ max_prev │ max_curr │
    // │ [Br × d] │ [Br × d] │ [(Bc+1) × d]   │ [Br×(Bc+1)] │  [Br]   │  [Br]    │  [Br]    │
    // ├──────────┼──────────┼────────────────┼──────────────┼─────────┼──────────┼──────────┤
    // │ partial  │ loaded   │ K^T then V      │ raw scores   │ running │ max from │ max for  │
    // │ O accum  │ once,    │ reused each     │ → probs      │ softmax │ previous │ current  │
    // │ += P@V   │ reused   │ KV iteration    │ in-place     │ denom   │ KV block │ KV block │
    // └──────────┴──────────┴────────────────┴──────────────┴─────────┴──────────┴──────────┘
    //  ^offset 0  ^Br*d      ^2*Br*d           ^2Br*d+(Bc+1)*d  ...
    //
    // kv_block DUAL USE per KV iteration:
    //
    //  PHASE 1 — holds K^T  (used for scores = q_block @ kv_block)
    //  kv_block[d_col * (Bc+PAD) + bc_row] = K[global_k_row, d_col]
    //
    //    d=4, Bc=4 layout:
    //        bc_row→  0    1    2    3
    //    d_col=0    [ k00  k10  k20  k30 ]  ← feature-0 of all 4 K-rows
    //    d_col=1    [ k01  k11  k21  k31 ]
    //    d_col=2    [ k02  k12  k22  k32 ]
    //    d_col=3    [ k03  k13  k23  k33 ]
    //
    //  PHASE 2 — holds V  (used for output += scores @ kv_block)
    //  kv_block[bc_row * d + d_col] = V[global_v_row, d_col]
    //
    //        d_col→   0    1    2    3
    //    bc_row=0   [ v00  v01  v02  v03 ]  ← V row 0 (normal row-major)
    //    bc_row=1   [ v10  v11  v12  v13 ]
    //    bc_row=2   [ v20  v21  v22  v23 ]
    //    bc_row=3   [ v30  v31  v32  v33 ]
    //
    //  K^T is no longer needed once scores are computed, so we reuse
    //  the same buffer for V without any extra memory cost.
    extern __shared__ float shared_mem[];
    float *output   = shared_mem;
    float *q_block  = shared_mem + Br * d;
    float *kv_block = shared_mem + 2 * Br * d;
    float *scores   = shared_mem + 2 * Br * d + (Bc + 1) * d;
    float *sum_exp  = shared_mem + 2 * Br * d + (Bc + 1) * d + Br * (Bc + 1);
    float *max_prev = sum_exp + Br;
    float *max_curr = max_prev + Br;

    // ---- Load Q block (once, stays in shared memory for all KV iterations) ----
    // q_block[row, col] = Q[q_block_idx*Br + row, col]
    // This is the "query tile" that all KV iterations reuse.
    for (int row_start = Wr * warp_id; row_start < Br; row_start += WARPS_PER_BLOCK * Wr)
        for (int col_start = Lc * lane_id; col_start < d; col_start += THREADS_PER_WARP * Lc)
            #pragma unroll
            for (int i = 0; i < Wr; i++)
                #pragma unroll
                for (int j = 0; j < Lc; j++) {
                    int global_row = q_block_idx * Br + row_start + i;
                    int col = col_start + j;
                    q_block[(row_start + i) * d + col] =
                        (global_row < N && col < d) ? Q[global_row * d + col] : 0.0f;
                }

    // ---- Initialise output=0, sum_exp=0, max_prev=-INF ----
    for (int row_start = Wr * warp_id; row_start < Br; row_start += WARPS_PER_BLOCK * Wr)
        for (int col_start = Lc * lane_id; col_start < d; col_start += THREADS_PER_WARP * Lc)
            #pragma unroll
            for (int i = 0; i < Wr; i++)
                #pragma unroll
                for (int j = 0; j < Lc; j++)
                    output[(row_start + i) * d + (col_start + j)] = 0.0f;

    for (int idx = lane_id; idx < Br; idx += THREADS_PER_WARP) {
        sum_exp[idx]  = 0.0f;
        max_prev[idx] = -__int_as_float(0x7f800000); // -INF
    }
    __syncthreads();

    // ---- MAIN LOOP over KV blocks ----
    //
    // Each iteration of this loop (one KV block):
    //
    //  Global memory         Shared memory              What happens
    //  ─────────────         ──────────────             ────────────────────────────
    //  K[kv*Bc : (kv+1)*Bc]  → kv_block (as K^T)       load & transpose K block
    //                           q_block @ kv_block      compute scores [Br×Bc]
    //  V[kv*Bc : (kv+1)*Bc]  → kv_block (overwrite)    load V block (K^T discarded)
    //                           online_softmax(...)     scale scores, exp, update m/s
    //                           output += probs @ V     accumulate new P@V
    //
    //  Running state after this iteration:
    //    m_prev ← max(old m, this block's max)
    //    s      ← s * correction + this block's sum
    //    output ← output * correction + this block's P@V
    //
    for (int kv_block_idx = 0; kv_block_idx < (N + Bc - 1) / Bc; kv_block_idx++) {

        // Reset max_curr for this iteration
        if (lane_id == 0)
            for (int i = 0; i < Wr; i++) {
                int r = warp_id * Wr + i;
                if (r < Br) max_curr[r] = -__int_as_float(0x7f800000);
            }
        __syncthreads();

        // ---- Load K TRANSPOSED into kv_block ----
        // We store K^T so that: q_block @ kv_block = scores  (Q[Br×d] @ K^T[d×Bc])
        // Layout: kv_block[d_col, bc_row] = K[kv_block_idx*Bc + bc_row, d_col]
        // This transpose means each K-column (a "key feature") is contiguous in memory.
        for (int row_start = Wr * warp_id; row_start < Bc; row_start += WARPS_PER_BLOCK * Wr)
            for (int col_start = Lc * lane_id; col_start < d; col_start += THREADS_PER_WARP * Lc)
                #pragma unroll
                for (int i = 0; i < Wr; i++)
                    #pragma unroll
                    for (int j = 0; j < Lc; j++) {
                        int bc_row = row_start + i;
                        int d_col  = col_start + j;
                        int k_global = kv_block_idx * Bc + bc_row;
                        // K^T layout: row = d_col, col = bc_row
                        kv_block[d_col * (Bc + PAD) + bc_row] =
                            (k_global < N && d_col < d) ? K[k_global * d + d_col] : 0.0f;
                    }
        __syncthreads();

        // ---- Compute scores = q_block @ K^T ----
        // scores[q_row, k_col] = dot(Q[q_row], K[k_col])
        // After this: scores[Br × Bc] = raw (unscaled) attention scores
        //
        // Strides:
        //   q_block  [Br × d]        → stride = d          (row-major, d cols)
        //   kv_block [d × (Bc+PAD)]  → stride = Bc+PAD     (K^T, padded)
        //   scores   [Br × (Bc+PAD)] → stride = Bc+PAD     (padded)
        matmul_warp_tiled<false, THREADS, Wr, Lc>(
            q_block, kv_block, scores,
            Br, Bc, d,
            d,          // stride_A: q_block rows are d wide
            Bc + PAD,   // stride_B: kv_block (K^T) rows are Bc+PAD wide
            Bc + PAD    // stride_C: scores rows are Bc+PAD wide
        );
        __syncthreads();

        // ---- Load V into kv_block (reuse the same buffer, K^T no longer needed) ----
        // V is stored in normal row-major order: kv_block[bc_row, d_col]
        for (int row_start = Wr * warp_id; row_start < Bc; row_start += WARPS_PER_BLOCK * Wr)
            for (int col_start = Lc * lane_id; col_start < d; col_start += THREADS_PER_WARP * Lc)
                #pragma unroll
                for (int i = 0; i < Wr; i++)
                    #pragma unroll
                    for (int j = 0; j < Lc; j++) {
                        int bc_row = row_start + i;
                        int d_col  = col_start + j;
                        int v_global = kv_block_idx * Bc + bc_row;
                        kv_block[bc_row * d + d_col] =
                            (v_global < N && d_col < d) ? V[v_global * d + d_col] : 0.0f;
                    }
        __syncthreads();

        // ---- ONLINE SOFTMAX + output accumulation ----
        // Inside this function (mapped to our Section 4 steps):
        //   Steps A+B+C: scale scores, find m_new
        //   Steps D+E:   compute exp(score - m_new), find sum_new
        //   Step  F:     update max_curr, sum_exp using correction factor
        //   Step  G p1:  output *= correction
        //   Step  G p2:  output += probs @ V
        online_softmax_and_accum_output<Br, Bc, THREADS, Wr, Lc, d>(
            max_curr, max_prev, sum_exp, scores, output, kv_block, inv_sqrt_d
        );
        __syncthreads();

        // Carry max_curr → max_prev for next KV block iteration
        if (lane_id == 0)
            for (int i = 0; i < Wr; i++) {
                int r = warp_id * Wr + i;
                if (r < Br) max_prev[r] = max_curr[r];
            }
        __syncthreads();
    }

    // ---- EPILOGUE: divide by sum_exp (final normalisation) ----
    // output now holds:  sum_{all j} exp(score_j - m_final) × V[j]
    // sum_exp holds:     sum_{all j} exp(score_j - m_final)
    // Dividing gives the true attention output.
    for (int row_start = Wr * warp_id; row_start < Br; row_start += WARPS_PER_BLOCK * Wr)
        for (int col_start = Lc * lane_id; col_start < d; col_start += THREADS_PER_WARP * Lc)
            #pragma unroll
            for (int i = 0; i < Wr; i++)
                #pragma unroll
                for (int j = 0; j < Lc; j++) {
                    int row = row_start + i, col = col_start + j;
                    if (row < Br && col < d) {
                        float s = sum_exp[row];
                        output[row * d + col] = (s > 1e-10f)
                            ? output[row * d + col] / s
                            : 0.0f;
                    }
                }
    __syncthreads();

    // Write normalised output to global memory
    for (int row_start = Wr * warp_id; row_start < Br; row_start += WARPS_PER_BLOCK * Wr)
        for (int col_start = Lc * lane_id; col_start < d; col_start += THREADS_PER_WARP * Lc)
            #pragma unroll
            for (int i = 0; i < Wr; i++)
                #pragma unroll
                for (int j = 0; j < Lc; j++) {
                    int row = row_start + i, col = col_start + j;
                    int global_row = q_block_idx * Br + row;
                    if (global_row < N && col < d)
                        O[global_row * d + col] = output[row * d + col];
                }
}

// ---------------------------------------------------------------------------
// Launch wrapper
// ---------------------------------------------------------------------------
template<int Br, int Bc, int THREADS, int Wr, int Lc, int d>
void launch_fa(const float *Q, const float *K, const float *V, float *O,
               int N, cudaStream_t stream = 0)
{
    int BLOCKS = (N + Br - 1) / Br;
    size_t smem = (2*Br*d + (Bc+1)*d + Br*(Bc+1) + 3*Br) * sizeof(float);
    float inv_sqrt_d = 1.0f / sqrtf((float)d);
    fa_kernel<Br, Bc, THREADS, d, Wr, Lc>
        <<<BLOCKS, THREADS, smem, stream>>>(Q, K, V, O, N, inv_sqrt_d, inv_sqrt_d);
}

// ---------------------------------------------------------------------------
// CPU reference attention (naive, for correctness verification)
// ---------------------------------------------------------------------------
void attention_cpu(const float *Q, const float *K, const float *V,
                   float *O, int N, int d)
{
    float inv_sqrt_d = 1.0f / sqrtf((float)d);
    for (int i = 0; i < N; i++) {
        // compute scores row i
        float scores[64]; // assumes N <= 64 for this demo
        float max_s = -1e30f;
        for (int j = 0; j < N; j++) {
            float dot = 0.f;
            for (int k = 0; k < d; k++) dot += Q[i*d+k] * K[j*d+k];
            scores[j] = dot * inv_sqrt_d;
            if (scores[j] > max_s) max_s = scores[j];
        }
        float sum = 0.f;
        for (int j = 0; j < N; j++) { scores[j] = expf(scores[j]-max_s); sum += scores[j]; }
        for (int j = 0; j < N; j++) scores[j] /= sum;
        // O[i] = scores @ V
        for (int k = 0; k < d; k++) {
            float val = 0.f;
            for (int j = 0; j < N; j++) val += scores[j] * V[j*d+k];
            O[i*d+k] = val;
        }
    }
}

// ---------------------------------------------------------------------------
// main — runs the 6-token example from the comments and a larger random test
// ---------------------------------------------------------------------------
#define CUDA_CHECK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){ \
    fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); \
    exit(1); } } while(0)

int main()
{
    // -----------------------------------------------------------------------
    // Test 1: exact 6-token example from the comment block
    // Br=2, Bc=2, Wr=1, Lc=1, d=2, THREADS=8*32=256
    // (THREADS >> N is fine; extra threads do nothing)
    // -----------------------------------------------------------------------
    // Br must equal WARPS_PER_BLOCK*Wr = 8*1 = 8.
    // Pad the 6-token example to 8 tokens (extra rows are zeros → don't affect row 0).
    printf("=== Test 1: 6-token example from comments (d=2, padded to N=8) ===\n");
    {
        constexpr int N=8, d=2, Br=8, Bc=4, Wr=1, Lc=1;
        constexpr int THREADS = WARPS_PER_BLOCK * THREADS_PER_WARP; // 256

        // First 6 rows are the example; rows 6,7 are zeros (padding)
        float h_Q[] = {1,0,  0,1,  1,1,  2,0,  0,2,  1,1,  0,0, 0,0};
        float h_K[] = {1,0,  0,1,  1,1,  2,0,  0,2,  1,1,  0,0, 0,0};
        float h_V[] = {1,0,  0,1,  1,1,  2,0,  0,2,  1,1,  0,0, 0,0};
        float h_O[N*d]={}, h_ref[N*d]={};

        float *d_Q,*d_K,*d_V,*d_O;
        CUDA_CHECK(cudaMalloc(&d_Q,N*d*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_K,N*d*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_V,N*d*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_O,N*d*sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_Q,h_Q,N*d*sizeof(float),cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_K,h_K,N*d*sizeof(float),cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_V,h_V,N*d*sizeof(float),cudaMemcpyHostToDevice));

        launch_fa<Br,Bc,THREADS,Wr,Lc,d>(d_Q,d_K,d_V,d_O,N);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(h_O,d_O,N*d*sizeof(float),cudaMemcpyDeviceToHost));

        attention_cpu(h_Q,h_K,h_V,h_ref,N,d);

        float max_err=0.f;
        for(int i=0;i<N*d;i++) max_err=fmaxf(max_err,fabsf(h_O[i]-h_ref[i]));
        printf("  Row 0 GPU  : [%.3f, %.3f]  (expected [1.173, 0.578])\n",h_O[0],h_O[1]);
        printf("  Row 0 CPU  : [%.3f, %.3f]\n",h_ref[0],h_ref[1]);
        printf("  Max error  : %.2e  (%s)\n\n",max_err,max_err<1e-4f?"PASS":"FAIL");

        cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_O);
    }

    // -----------------------------------------------------------------------
    // Test 2: larger random test
    // -----------------------------------------------------------------------
    printf("=== Test 2: 32-token random test (d=2) ===\n");
    {
        constexpr int N=32, d=2, Br=8, Bc=8, Wr=1, Lc=1;
        constexpr int THREADS = WARPS_PER_BLOCK * THREADS_PER_WARP;

        float h_Q[N*d], h_K[N*d], h_V[N*d], h_O[N*d], h_ref[N*d];
        for(int i=0;i<N*d;i++){
            h_Q[i]=sinf((float)i*0.3f);
            h_K[i]=cosf((float)i*0.5f);
            h_V[i]=sinf((float)i*0.7f);
        }

        float *d_Q,*d_K,*d_V,*d_O;
        CUDA_CHECK(cudaMalloc(&d_Q,N*d*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_K,N*d*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_V,N*d*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_O,N*d*sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_Q,h_Q,N*d*sizeof(float),cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_K,h_K,N*d*sizeof(float),cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_V,h_V,N*d*sizeof(float),cudaMemcpyHostToDevice));

        launch_fa<Br,Bc,THREADS,Wr,Lc,d>(d_Q,d_K,d_V,d_O,N);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(h_O,d_O,N*d*sizeof(float),cudaMemcpyDeviceToHost));

        attention_cpu(h_Q,h_K,h_V,h_ref,N,d);

        float max_err=0.f;
        for(int i=0;i<N*d;i++) max_err=fmaxf(max_err,fabsf(h_O[i]-h_ref[i]));
        printf("  Max error vs CPU: %.2e  (%s)\n\n",max_err,max_err<1e-4f?"PASS":"FAIL");

        cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_O);
    }

    return 0;
}
