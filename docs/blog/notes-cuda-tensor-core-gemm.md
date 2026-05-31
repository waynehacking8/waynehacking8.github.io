---
description: "Notes on CUDA matrix multiply from naive to tiled to Tensor Core (WMMA) GEMM — shared-memory tiling, fragments, and measuring against the cuBLAS ceiling on Hopper and Blackwell."
---

# Notes on CUDA Tensor Core GEMM (WMMA)

*2026-05-31 · CUDA / GPU kernels*

Working notes on writing a matrix-multiply (GEMM) kernel in CUDA and climbing from a naive
implementation to **Tensor Cores** via the **WMMA** API — and, just as important, how to know
how good your kernel actually is by measuring it against the **cuBLAS** ceiling. GEMM is the
right thing to understand deeply: it is the operation underneath every linear layer and
attention projection in an LLM.

## 1. Why GEMM is the kernel that matters

A transformer forward pass is, to a first approximation, a stack of GEMMs. If you understand
what makes a GEMM kernel fast on a GPU, you understand where inference latency comes from and
why Tensor Cores exist. The progression below is the standard pedagogy — each step removes the
bottleneck the previous one exposed.

## 2. Naive GEMM: memory-bound by construction

The textbook kernel assigns one thread per output element `C[i][j]` and loops over `k`:

```
C[i][j] = Σ_k A[i][k] * B[k][j]
```

It's correct and it's slow. Every thread re-reads entire rows of `A` and columns of `B` from
**global memory**, so the same values are fetched from DRAM hundreds of times. The kernel is
**memory-bandwidth-bound** — the arithmetic units sit idle waiting on loads. The arithmetic
intensity (FLOPs per byte) is far too low to approach peak.

## 3. Tiled GEMM: shared memory turns the problem compute-bound

The fix is **shared-memory tiling**. Threads in a block cooperatively load a tile of `A` and a
tile of `B` into fast on-chip **shared memory**, then every thread in the block reuses those
tiles for its partial sums before loading the next tile:

1. Load a `TILE × TILE` block of `A` and of `B` into shared memory (coalesced).
2. `__syncthreads()`.
3. Each thread accumulates its `C` partial product from the shared tiles.
4. `__syncthreads()`, advance to the next tile along `k`.

This raises arithmetic intensity by a factor of `TILE`: each value loaded from global memory is
now reused `TILE` times. The kernel crosses from memory-bound toward **compute-bound** — now
the FP32 ALUs are the limit. This is the single biggest jump, and it's pure data-movement
strategy, not math.

## 4. Tensor Cores via WMMA: a different compute unit

Tiling saturates the *CUDA cores*. Tensor Cores are a **separate** unit that does a small
matrix-multiply-accumulate (MMA) in one instruction — e.g. a 16×16×16 `D = A·B + C` per warp,
on half-precision inputs with FP32 accumulation. The **WMMA** (Warp Matrix Multiply-Accumulate)
API exposes them in CUDA C++:

```cpp
using namespace nvcuda::wmma;
fragment<matrix_a, 16,16,16, half, row_major> a_frag;
fragment<matrix_b, 16,16,16, half, col_major> b_frag;
fragment<accumulator, 16,16,16, float>        c_frag;

fill_fragment(c_frag, 0.0f);
for (int k = 0; k < K; k += 16) {
    load_matrix_sync(a_frag, A + ..., lda);
    load_matrix_sync(b_frag, B + ..., ldb);
    mma_sync(c_frag, a_frag, b_frag, c_frag);   // Tensor Core MMA
}
store_matrix_sync(C + ..., c_frag, ldc, mem_row_major);
```

The mental model shifts from "threads computing elements" to "**warps cooperating on
fragments**." A `fragment` is an opaque, register-resident tile; you don't index its elements,
you feed whole fragments to `mma_sync`. Inputs are FP16/BF16 (or FP8/FP4 on newer
architectures), accumulation is FP32 — which is why mixed precision is the native language of
Tensor Cores.

## 5. The number that tells the truth: % of cuBLAS

A hand-written WMMA kernel will beat your tiled kernel, but it will **not** beat cuBLAS — and
that's the point. cuBLAS is the practical ceiling (it does register-tiling, double-buffering,
swizzled layouts, and architecture-specific tuning you won't replicate in an afternoon). So the
honest metric isn't raw TFLOP/s, it's **percent of the cuBLAS ceiling on the same GPU**:

| Kernel | Typical regime |
| --- | --- |
| Naive | a few % of peak — memory-bound |
| Tiled (shared memory) | much better, still CUDA-core bound |
| WMMA (Tensor Core) | a meaningful fraction of cuBLAS |
| cuBLAS | the ceiling (100%) |

Reporting "X % of cuBLAS on sm_90 and sm_120" is a self-describing result: it's reproducible,
it normalises across GPUs, and it's honest about the gap to a production library. Profiling the
WMMA kernel with **Nsight Compute** then tells you *which* wall you're against — memory
throughput, Tensor Core utilisation, or occupancy.

## 6. Why this matters going to Blackwell

Each GPU generation widens what the MMA unit accepts: Hopper added FP8, **Blackwell** adds
**FP4** and a new generation of Tensor Core instructions (`tcgen05`). The WMMA mental model —
fragments fed to an MMA, FP32 accumulation — carries forward; what changes is the input
precision and the tile shapes. Understanding the FP16 WMMA path is the on-ramp to reasoning
about NVFP4 inference on Blackwell.

## Takeaway

GEMM performance is a ladder: naive is memory-bound, **shared-memory tiling** makes it
compute-bound, and **WMMA** moves the compute onto Tensor Cores with mixed precision. Measure
every rung as **% of cuBLAS** on the same GPU — that's the metric that's honest about how close
you are to the ceiling and portable across Hopper and Blackwell.

→ More field notes on the NVIDIA stack: [waynehacking8.github.io](https://waynehacking8.github.io/)
