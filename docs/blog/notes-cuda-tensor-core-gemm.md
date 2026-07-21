---
description: "From naive GEMM to shared-memory tiling and WMMA, with paragraph-by-paragraph Traditional Chinese and comparisons against cuBLAS on the same GPU."
date: "2026-05-31"
updated: "2026-07-21"
language:
  - en
  - zh-Hant
image: "https://wayne.is-a.dev/assets/blog/wmma-tile.webp"
tags:
  - CUDA
  - Tensor Cores
  - Performance
---

# From Naive GEMM to WMMA: Where Each CUDA Kernel Stalls｜從 Naive GEMM 到 WMMA：每個 CUDA kernel 卡在哪裡

*2026-05-31 · updated 2026-07-21 · CUDA / GPU kernels*

<figure class="pb-article-hero pb-article-contain">
  <img src="/assets/blog/wmma-tile.webp" alt="NVIDIA WMMA warp tile 結構圖" loading="eager" decoding="async">
  <figcaption>WMMA warp-level tile structure · WMMA warp-level tile 結構 · <a href="https://developer.nvidia.com/blog/cutlass-linear-algebra-cuda/">Source: NVIDIA Developer</a></figcaption>
</figure>

This article follows one GEMM through three implementations: one output element per thread, shared-memory tiling, and warp-level WMMA. Each version is compared with cuBLAS on the same GPU. GEMM sits underneath the linear layers and attention projections in an LLM.

這篇文章用三種實作走過同一個 GEMM：每個 thread 計算一個輸出元素、shared-memory tiling，以及 warp-level WMMA。每一版都和同一張 GPU 上的 cuBLAS 比較。LLM 的 linear layers 與 attention projections 底下都會用到 GEMM。
{ .pb-translation lang=zh-Hant }

## 1. GEMM in a transformer forward pass｜Transformer forward pass 裡的 GEMM

A transformer forward pass contains a stack of matrix multiplications. GEMM performance determines much of the cost of linear projections, while the kernel’s memory traffic, tile shape, and precision decide how closely it approaches the GPU’s compute throughput. The three versions below expose those constraints one at a time.

Transformer forward pass 會執行一連串矩陣乘法。Linear projections 的大部分成本取決於 GEMM，而 kernel 的記憶體流量、tile shape 與 precision，會決定它能多接近 GPU 的 compute throughput。下面三個版本依序把這些限制攤開。
{ .pb-translation lang=zh-Hant }

## 2. Naive GEMM: memory-bound by construction｜Naive GEMM：結構上就受限於記憶體

The textbook kernel assigns one thread per output element `C[i][j]` and loops over `k`:

教科書版本會讓每個 thread 負責一個輸出元素 `C[i][j]`，再沿著 `k` 迴圈累加：
{ .pb-translation lang=zh-Hant }

```
C[i][j] = Σ_k A[i][k] * B[k][j]
```

It is correct and slow. Every thread re-reads entire rows of `A` and columns of `B` from global memory, so the same values are fetched from DRAM hundreds of times. The arithmetic units wait on loads because arithmetic intensity—FLOPs per byte—is too low to approach peak compute.

這個版本正確，但很慢。每個 thread 都會從 global memory 重讀 `A` 的整列與 `B` 的整欄，同一批值可能從 DRAM 取回數百次。Arithmetic intensity，也就是每 byte 對應的 FLOPs，低到無法接近 peak compute，運算單元大部分時間都在等資料。
{ .pb-translation lang=zh-Hant }

## 3. Tiled GEMM: reuse through shared memory｜Tiled GEMM：用 shared memory 重複利用資料

Shared-memory tiling changes the data path. Threads in a block cooperatively load a tile of `A` and `B` into on-chip shared memory, reuse those values for partial sums, and then advance along `k`:

Shared-memory tiling 改變資料搬移方式。Block 內的 threads 合作把 `A` 與 `B` 的 tile 搬進晶片上的 shared memory，重複使用這些值計算 partial sums，再沿著 `k` 前進：
{ .pb-translation lang=zh-Hant }

1. Load a `TILE × TILE` block of `A` and `B` into shared memory with coalesced accesses.<br><span class="pb-inline-translation" lang="zh-Hant">用 coalesced accesses 把 `A` 與 `B` 的 `TILE × TILE` 區塊載入 shared memory。</span>
2. Call `__syncthreads()`.<br><span class="pb-inline-translation" lang="zh-Hant">呼叫 `__syncthreads()`。</span>
3. Accumulate each thread’s partial product from the shared tiles.<br><span class="pb-inline-translation" lang="zh-Hant">每個 thread 從 shared tiles 累加自己的 partial product。</span>
4. Synchronize again and load the next tile along `k`.<br><span class="pb-inline-translation" lang="zh-Hant">再次同步，再載入 `k` 方向的下一個 tile。</span>

Tiling increases reuse before another global-memory load. Whether the kernel becomes compute-bound depends on tile size, occupancy, layout, and matrix shape; tiling alone does not guarantee that transition.

Tiling 讓資料在下一次 global-memory load 前被重複使用。Kernel 是否會轉成 compute-bound，仍取決於 tile size、occupancy、layout 與 matrix shape；光是加入 tiling 並不保證完成這個轉換。
{ .pb-translation lang=zh-Hant }

## 4. Tensor Cores through WMMA｜透過 WMMA 使用 Tensor Cores

Tiling can saturate CUDA cores. Tensor Cores are a separate execution unit for matrix-multiply-accumulate instructions—for example, a 16×16×16 `D = A·B + C` operation per warp with half-precision inputs and FP32 accumulation. CUDA C++ exposes this path through the WMMA (Warp Matrix Multiply-Accumulate) API:

Tiling 可以讓 CUDA cores 飽和。Tensor Cores 是另一組執行單元，負責 matrix-multiply-accumulate 指令；例如每個 warp 執行一次 16×16×16 的 `D = A·B + C`，輸入使用 half precision，累加則使用 FP32。CUDA C++ 透過 WMMA（Warp Matrix Multiply-Accumulate）API 暴露這條路徑：
{ .pb-translation lang=zh-Hant }

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

Supported fragment shapes, layouts, and precision combinations depend on the GPU architecture. The CUDA Programming Guide, rather than this shortened kernel, defines the exact requirements.[^cuda-guide]

支援的 fragment shapes、layouts 與 precision combinations 會隨 GPU 架構改變。上面的 kernel 刻意省略細節，精確限制仍以 CUDA Programming Guide 為準。[^cuda-guide]
{ .pb-translation lang=zh-Hant }

WMMA assigns each fragment to a warp. The API exposes fragment load, MMA, and store operations rather than a stable per-thread element mapping. Inputs may use FP16 or BF16—and FP8 or FP4 on newer architectures—while accumulation commonly uses FP32.

WMMA 以 warp 為單位分配 fragment。API 暴露的是 fragment load、MMA 與 store，不提供穩定的 per-thread element mapping。輸入可用 FP16 或 BF16；較新的架構也支援 FP8 或 FP4，而 accumulation 通常使用 FP32。
{ .pb-translation lang=zh-Hant }

## 5. Compare against cuBLAS on the same shape｜用相同 shape 與 cuBLAS 比較

cuBLAS is the production baseline, not a universal upper bound for every shape. It uses register tiling, double buffering, swizzled layouts, and architecture-specific tuning. A specialized hand-written kernel can exceed a selected cuBLAS path on some shapes, so report both absolute throughput and the ratio to cuBLAS under matched conditions.

cuBLAS 是 production baseline，不是所有 shape 的絕對上限。它使用 register tiling、double buffering、swizzled layouts 與架構專屬調校；針對特定 shape 的手寫 kernel 仍可能超過某條 cuBLAS path。因此，應在條件一致時同時回報絕對吞吐量與相對 cuBLAS 比值。
{ .pb-translation lang=zh-Hant }

| Kernel／核心 | Typical regime／典型狀態 |
| --- | --- |
| Naive | A few percent of peak; memory-bound／僅達 peak 的少數百分比，受限於記憶體 |
| Tiled (shared memory)／Tiled（shared memory） | Higher reuse; often still CUDA-core bound／重複利用率較高，通常仍受 CUDA core 限制 |
| WMMA (Tensor Core)／WMMA（Tensor Core） | A meaningful fraction of the matched cuBLAS result／可達相同條件 cuBLAS 結果的一定比例 |
| cuBLAS | Production baseline for the selected shape and algorithm (100%)／指定 shape 與 algorithm 的 production baseline（100%） |

A reproducible result needs the GPU, matrix shape, precision, warm-up, timing method, cuBLAS algorithm, and measured TFLOP/s. The cuBLAS ratio belongs beside those values; results reported for `sm_90` and `sm_120` must remain separate. Nsight Compute can then identify whether the remaining limit is memory throughput, Tensor Core utilization, or occupancy.

可重現的結果要附上 GPU、matrix shape、precision、warm-up、timing method、cuBLAS algorithm 與實測 TFLOP/s，再把 cuBLAS ratio 放在這些數值旁；`sm_90` 與 `sm_120` 的結果也必須分開回報。接著可用 Nsight Compute 判斷剩餘限制來自 memory throughput、Tensor Core utilization 或 occupancy。
{ .pb-translation lang=zh-Hant }

## 6. From WMMA to Blackwell `tcgen05`｜從 WMMA 走到 Blackwell `tcgen05`

Each GPU generation changes the precision and tile shapes accepted by the MMA unit. Hopper added FP8; Blackwell adds FP4 and a new Tensor Core instruction generation, `tcgen05`. The warp-level WMMA path remains useful background, but Blackwell kernels use different data movement and instruction interfaces.[^blackwell]

每一代 GPU 都會改變 MMA unit 可接受的 precision 與 tile shapes。Hopper 加入 FP8；Blackwell 則加入 FP4 與新一代 Tensor Core 指令 `tcgen05`。Warp-level WMMA 仍是有用的背景知識，但 Blackwell kernel 使用的資料搬移與指令介面已經不同。[^blackwell]
{ .pb-translation lang=zh-Hant }

→ More field notes on the NVIDIA stack:<br><span class="pb-inline-translation" lang="zh-Hant">更多 NVIDIA stack 實作筆記：</span>
[wayne.is-a.dev](https://wayne.is-a.dev/)

[^cuda-guide]: [CUDA C++ Programming Guide — Warp Matrix Functions](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#warp-matrix-functions), NVIDIA’s specification for WMMA fragments, layouts, and synchronization requirements.／NVIDIA 對 WMMA fragments、layouts 與同步需求的規格。
[^blackwell]: [NVIDIA Blackwell Architecture Technical Brief](https://resources.nvidia.com/en-us-blackwell-architecture), used for the generation-level Tensor Core and low-precision capability summary.／本文 Tensor Core 世代差異與低精度能力的來源。
