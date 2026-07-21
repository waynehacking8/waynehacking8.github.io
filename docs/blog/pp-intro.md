---
description: "CUDA C/C++ guide covering the execution model, memory hierarchy, occupancy, profiling, streams, synchronization, and kernel optimization."
date: "2025-01-11"
updated: "2026-07-21"
language: "zh-Hant"
image: "https://wayne.is-a.dev/assets/blog/cuda-platform.webp"
tags:
  - CUDA
  - Parallel Programming
  - Performance
---

# CUDA Programming 入門

*2025-01-11 · GPU / parallel programming*

<figure class="pb-article-hero">
  <img src="/assets/blog/cuda-platform.webp" alt="NVIDIA CUDA accelerated computing 官方視覺" loading="eager" decoding="async">
  <figcaption>CUDA accelerated computing platform · <a href="https://developer.nvidia.com/cuda-zone">Source: NVIDIA Developer</a></figcaption>
</figure>

這份指南從 execution model、memory hierarchy 和 synchronization 說明 CUDA 的硬體
成本，再用 profiling 判斷該改資料存取、同步還是 kernel 組態。本文的 execution model、memory hierarchy
與 synchronization 語意以 NVIDIA CUDA C++ Programming Guide 為準；範例刻意縮短，
production kernel 仍需補齊 bounds check 與 error handling。[^cuda-guide]

## 1. CUDA 基礎架構

### 1.1 CUDA 程式設計模型

#### 1.1.1 運算架構
CUDA（Compute Unified Device Architecture）是 NVIDIA 開發的平行運算平台與程式設計模型。其架構包含：

- **主機（Host）**：CPU 及其記憶體
- **裝置（Device）**：GPU 及其記憶體
- **執行單元**：包含多個串流多處理器（Streaming Multiprocessors，SMs）

#### 1.1.2 程式執行流程
CUDA 程式的典型執行流程：

1. 分配主機和裝置記憶體
2. 將資料從主機複製到裝置
3. 呼叫CUDA核心進行運算
4. 將結果從裝置複製回主機
5. 釋放記憶體

### 1.2 核心函數開發

#### 1.2.1 函數類型限定詞
CUDA 提供三種函式類型限定詞：

1. **__global__**
   - 在裝置端執行
   - 可從主機端或裝置端呼叫
   - 必須回傳 `void`
   ```cpp
   __global__ void kernelFunction(float* data)
   {
       // 裝置端程式碼
   }
   ```

2. **__device__**
   - 在裝置端執行
   - 只能從裝置端呼叫
   ```cpp
   __device__ float deviceFunction(float x)
   {
       return x * x;
   }
   ```

3. **__host__**
   - 在主機端執行
   - 只能從主機端呼叫
   ```cpp
   __host__ void hostFunction(float* data)
   {
       // 主機端程式碼
   }
   ```

#### 1.2.2 核心函數呼叫語法

Kernel 的呼叫語法如下：
```cpp
kernelFunction<<<gridDim, blockDim, sharedMemBytes, stream>>>(parameters);
```

參數說明：
- `gridDim`：網格維度，指定區塊數量
- `blockDim`：區塊維度，指定每個區塊的執行緒數量
- `sharedMemBytes`：動態共享記憶體大小（可選）
- `stream`：CUDA 串流（可選）

### 1.3 NVCC 編譯器

#### 1.3.1 編譯流程
NVCC 編譯器的工作流程：

1. 分離主機和裝置程式碼
2. 編譯裝置程式碼，產生 PTX 或 cubin
3. 編譯主機程式碼
4. 連結所有目標檔案

#### 1.3.2 常用編譯選項

```bash
nvcc -arch=sm_70        # 指定目標架構
     -O3                # 優化等級
     -G                # 加入除錯資訊
     -lineinfo         # 加入行號資訊
     -o output.exe     # 指定輸出檔案
     source.cu         # 來源檔案
```

重要編譯選項說明：
- `-arch`：指定 GPU 架構版本
- `-code`：指定實際產生的 GPU 程式碼版本
- `-Xptxas`：傳遞選項給 PTX 組譯器
- `-maxrregcount`：限制每個執行緒使用的暫存器數量

## 2. 記憶體管理與最佳化

### 2.1 記憶體類型

#### 2.1.1 全域記憶體
- 最大容量的記憶體空間
- 所有執行緒都可存取
- 存取延遲較高

配置與使用：
```cpp
// 使用傳統方式
float *d_data;
cudaMalloc(&d_data, size);
cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);

// 使用統一記憶體
float *u_data;
cudaMallocManaged(&u_data, size);
```

#### 2.1.2 共享記憶體
- 區塊內的執行緒共享
- 存取速度接近暫存器
- 容量有限

宣告與使用：
```cpp
__global__ void sharedMemKernel()
{
    __shared__ float sharedData[256];
    sharedData[threadIdx.x] = someData;
    __syncthreads();
    // 使用共享資料
}
```

#### 2.1.3 暫存器記憶體
- 每個執行緒私有
- 最快的記憶體類型
- 數量有限

使用範例：
```cpp
__global__ void registerKernel()
{
    float localData = 0;  // 自動分配到暫存器
    // 執行運算
}
```

### 2.2 統一記憶體管理

#### 2.2.1 基本概念
Unified Memory 提供 CPU 與 GPU 共用的虛擬位址空間。Runtime 會按需遷移頁面，也允許
分配超過單張 GPU 實體記憶體的資料，但 page fault 可能成為效能瓶頸。

#### 2.2.2 記憶體預取
先把頁面預取到目前的 GPU：

```cpp
// 基本預取
cudaMemPrefetchAsync(data, size, deviceId);

// 進階預取策略
void optimizedPrefetch(float* data, size_t size)
{
    int deviceId;
    cudaGetDevice(&deviceId);
    
    // 預取資料到GPU
    cudaMemPrefetchAsync(data, size, deviceId);
    
    // 執行核心
    computeKernel<<<blocks, threads>>>(data);
    
    // 預取結果回CPU
    cudaMemPrefetchAsync(data, size, cudaCpuDeviceId);
}
```

#### 2.2.3 記憶體存取模式
相鄰執行緒應讀取相鄰位址：

1. **合併存取**
```cpp
__global__ void coalescedAccess(float* data)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float value = data[idx];  // 合併存取
}
```

2. **避免分歧存取**
```cpp
__global__ void strideAccess(float* data, int stride)
{
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * stride;
    float value = data[idx];  // 可能造成非合併存取
}
```

### 2.3 記憶體最佳化技巧

#### 2.3.1 記憶體對齊
對齊資料，減少拆分的記憶體交易：

```cpp
// 結構體對齊
struct __align__(16) AlignedStruct {
    float x, y, z, w;
};

// 動態配置對齊記憶體
void* alignedMalloc(size_t size, size_t alignment)
{
    size_t mask = alignment - 1;
    void* p;
    cudaMalloc(&p, size + mask);
    void* aligned = (void*)(((uintptr_t)p + mask) & ~mask);
    return aligned;
}
```

#### 2.3.2 記憶體覆蓋
載入 stencil 所需的 halo 區域：

```cpp
__global__ void overlappedKernel(float* in, float* out)
{
    __shared__ float tile[TILE_SIZE + 2];  // 包含重疊區域
    
    // 載入資料包含邊界
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    tile[threadIdx.x + 1] = in[idx];
    
    // 載入邊界元素
    if (threadIdx.x == 0)
        tile[0] = in[idx - 1];
    if (threadIdx.x == blockDim.x - 1)
        tile[threadIdx.x + 2] = in[idx + 1];
        
    __syncthreads();
    
    // 使用重疊區域進行計算
    out[idx] = tile[threadIdx.x + 1] + tile[threadIdx.x] + tile[threadIdx.x + 2];
}
```

## 3. 執行緒組織與管理

### 3.1 執行緒層次結構

#### 3.1.1 網格、區塊與執行緒
`dim3` 可建立一維、二維或三維的 grid 與 block：

```cpp
// 1D 配置
dim3 grid(numBlocks);
dim3 block(numThreads);

// 2D 配置
dim3 grid(numBlocksX, numBlocksY);
dim3 block(numThreadsX, numThreadsY);

// 3D 配置
dim3 grid(numBlocksX, numBlocksY, numBlocksZ);
dim3 block(numThreadsX, numThreadsY, numThreadsZ);

// 啟動核心
kernel<<<grid, block>>>(parameters);
```

#### 3.1.2 索引計算

多維度索引計算：

```cpp
__global__ void multiDimKernel(float* data)
{
    // 1D 索引
    int idx1D = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 2D 索引
    int idx2D = (blockIdx.y * gridDim.x + blockIdx.x) * (blockDim.x * blockDim.y) +
                 threadIdx.y * blockDim.x + threadIdx.x;
                 
    // 3D 索引
    int idx3D = ((blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x) *
                (blockDim.x * blockDim.y * blockDim.z) +
                (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;
}
```

### 3.2 執行緒同步

#### 3.2.1 區塊內同步
使用`__syncthreads()`確保區塊內執行緒同步：

```cpp
__global__ void syncKernel(float* data)
{
    __shared__ float sharedData[256];
    
    // 載入資料
    sharedData[threadIdx.x] = data[threadIdx.x];
    __syncthreads();  // 確保所有資料都已載入
    
    // 使用資料
    float result = sharedData[255 - threadIdx.x];
    __syncthreads();  // 確保所有執行緒完成存取
    
    data[threadIdx.x] = result;
}
```

#### 3.2.2 合作群組
Cooperative Groups 可把同步範圍明確寫進程式：

```cpp
#include <cooperative_groups.h>
using namespace cooperative_groups;

__global__ void coopKernel(float* data)
{
    // 取得當前執行緒群組
    thread_block block = this_thread_block();
    
    // 群組同步
    block.sync();
    
    // 群組大小
    int size = block.size();
    
    // 群組內執行緒索引
    int idx = block.thread_rank();
}
```

### 3.3 執行組態優化

#### 3.3.1 估算 block size 與 occupancy

CUDA occupancy API 可先給一組候選 block size；最終仍要用實測決定：

```cpp
__global__ void computeKernel(float* data, int N);

void calculateCandidateConfig(float* data, int N)
{
    int minGridSize;
    int blockSize;
    cudaOccupancyMaxPotentialBlockSize(
        &minGridSize, &blockSize, computeKernel, 0, 0);

    int gridSize = (N + blockSize - 1) / blockSize;
    computeKernel<<<gridSize, blockSize>>>(data, N);
}
```

#### 3.3.2 動態平行處理
實作動態平行處理：

```cpp
__global__ void dynamicParallelKernel(float* data, int depth)
{
    if (depth > 0)
    {
        // 遞迴啟動新的核心
        dynamicParallelKernel<<<gridDim, blockDim>>>(data, depth - 1);
        cudaDeviceSynchronize();
    }
    
    // 處理資料
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    data[idx] *= 2.0f;
}
```

## 4. 效能分析與優化

### 4.1 使用 nsys 進行效能分析

#### 4.1.1 基本分析
用 `nsys` 收集 timeline 與 CUDA API 統計：

```bash
# 基本效能分析
nsys profile --stats=true ./myapp

# 詳細記憶體分析
nsys profile --trace=cuda,nvtx,osrt,cublas ./myapp

# 產生時序報告
nsys profile --trace=cuda --force-overwrite true \
             --delay=0 --duration=0 \
             --output=timeline ./myapp
```

#### 4.1.2 分析報告解讀

nsys 產生的報告包含以下重要資訊：

1. **CUDA API 統計**
   - 核心啟動時間
   - 記憶體操作時間
   - API 呼叫頻率

2. **CUDA 核心統計**
   - 核心執行時間
   - 每個核心的效能指標
   - 硬體資源使用率

3. **記憶體操作統計**
   - 記憶體傳輸大小
   - 頁面錯誤次數
   - 資料遷移時間

範例報告解析：
```plaintext
 Time(%)  Total Time (ns)  Num Calls    Avg (ns)   Med (ns)    Min (ns)    Max (ns)   StdDev (ns)
 -------  ---------------  ---------  -----------  ----------  ----------  -----------  -----------
    45.3       2,345,678        100     23,456.8    22,345.6     19,876     28,987.6      1,234.5
```

### 4.2 效能瓶頸分析

#### 4.2.1 記憶體瓶頸分析
先量測 host-to-device 複製時間：

```cpp
// 使用事件來測量記憶體操作時間
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

cudaEventRecord(start);
cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);
cudaEventRecord(stop);
cudaEventSynchronize(stop);

float milliseconds = 0;
cudaEventElapsedTime(&milliseconds, start, stop);
printf("Memory transfer took %f ms\n", milliseconds);
```

#### 4.2.2 計算瓶頸分析
用 FLOPs／byte 粗估算術強度：

```cpp
// 計算算術密度
float computeArithmeticIntensity(int operations, int bytes_accessed)
{
    return (float)operations / bytes_accessed;
}

// 使用事件測量核心執行時間
__global__ void computeKernel(float* data, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
    {
        // 記錄計算操作數和記憶體存取量
        int compute_ops = 0;
        int memory_accesses = 0;
        
        float temp = data[idx];  // 記憶體讀取
        memory_accesses += sizeof(float);
        
        for(int i = 0; i < 100; i++)
        {
            temp = temp * temp + temp;  // 計算操作
            compute_ops += 2;
        }
        
        data[idx] = temp;  // 記憶體寫入
        memory_accesses += sizeof(float);
    }
}
```

### 4.3 效能最佳化策略

#### 4.3.1 記憶體最佳化
把會重複使用的值放進 shared memory：

```cpp
__global__ void optimizedMemoryKernel(float* data, int N)
{
    // 使用共享記憶體減少全域記憶體存取
    __shared__ float sharedData[BLOCK_SIZE];
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 合併讀取全域記憶體
    if (idx < N)
        sharedData[threadIdx.x] = data[idx];
    
    __syncthreads();
    
    // 使用共享記憶體進行計算
    if (threadIdx.x < blockDim.x - 1)
    {
        float result = sharedData[threadIdx.x] + sharedData[threadIdx.x + 1];
        data[idx] = result;
    }
}
```

#### 4.3.2 計算最佳化
固定迴圈長度時，可讓編譯器展開迴圈：

```cpp
__global__ void optimizedComputeKernel(float* data, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
    {
        // 迴圈展開
        float sum = 0;
        #pragma unroll 4
        for(int i = 0; i < 16; i += 4)
        {
            sum += data[idx + i * N];
            sum += data[idx + (i+1) * N];
            sum += data[idx + (i+2) * N];
            sum += data[idx + (i+3) * N];
        }
        
        data[idx] = sum;
    }
}
```

## 5. 串流與非同步執行

### 5.1 CUDA 串流基礎

#### 5.1.1 串流建立與管理
```cpp
// 建立多個串流
cudaStream_t streams[4];
for(int i = 0; i < 4; i++)
{
    cudaStreamCreate(&streams[i]);
}

// 在不同串流中執行核心
for(int i = 0; i < 4; i++)
{
    int offset = i * N/4;
    processKernel<<<blocks, threads, 0, streams[i]>>>(
        d_data + offset, N/4
    );
}

// 同步與清理
for(int i = 0; i < 4; i++)
{
    cudaStreamSynchronize(streams[i]);
    cudaStreamDestroy(streams[i]);
}
```

#### 5.1.2 串流回呼函式

串流完成後執行 host 端回呼：

```cpp
void CUDART_CB callbackFunc(cudaStream_t stream, cudaError_t status, void* userData)
{
    printf("Stream %p completed with status %d\n", stream, status);
}

// 設定回呼
cudaStreamAddCallback(stream, callbackFunc, nullptr, 0);
```

### 5.2 串流同步機制

#### 5.2.1 事件同步
事件可建立跨 stream 的相依關係：

```cpp
// 建立事件
cudaEvent_t event;
cudaEventCreate(&event);

// 在串流中記錄事件
cudaEventRecord(event, stream1);

// 使其他串流等待事件
cudaStreamWaitEvent(stream2, event);
cudaStreamWaitEvent(stream3, event);

// 清理
cudaEventDestroy(event);
```

#### 5.2.2 串流優先權
設定串流優先權：

```cpp
// 建立高優先權串流
cudaStream_t highPriorityStream;
int priority;
cudaDeviceGetStreamPriorityRange(nullptr, &priority);
cudaStreamCreateWithPriority(&highPriorityStream, 
                            cudaStreamNonBlocking, 
                            priority);
```

## 6. 進階最佳化技巧

### 6.1 動態平行處理

由 device 端 kernel 啟動子 kernel：

```cpp
__global__ void recursiveKernel(int* data, int depth)
{
    if (depth > 0)
    {
        // 動態啟動子核心
        recursiveKernel<<<gridDim.x, blockDim.x>>>(
            data + blockIdx.x * blockDim.x, 
            depth - 1
        );
    }
    
    // 處理當前層級的資料
    int idx = threadIdx.x;
    data[idx] = blockIdx.x * blockDim.x + idx;
}
```

### 6.2 原子操作最佳化

多個執行緒更新同一位置時，可使用原子操作：

```cpp
__global__ void atomicKernel(int* counter, int* data, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
    {
        // 使用原子操作進行計數
        int old = atomicAdd(counter, 1);
        
        // 使用原子操作進行條件更新
        atomicMax(&data[idx], old);
    }
}
```

## 7. 實戰案例分析

### 7.1 N體模擬最佳化

以下範例計算 N 體問題的作用力：

```cpp
#define SOFTENING 1e-9f

typedef struct { 
    float4 pos;
    float4 vel;
} Body;

__global__ 
void bodyForce(Body* bodies, float dt, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        float4 myPos = bodies[idx].pos;
        float4 acc = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        
        // 計算引力
        for(int j = 0; j < n; j++)
        {
            float4 otherPos = bodies[j].pos;
            float3 r;
            r.x = otherPos.x - myPos.x;
            r.y = otherPos.y - myPos.y;
            r.z = otherPos.z - myPos.z;
            
            float distSqr = r.x * r.x + r.y * r.y + r.z * r.z + SOFTENING;
            float invDist = rsqrtf(distSqr);
            float invDist3 = invDist * invDist * invDist;
            
            acc.x += r.x * invDist3;
            acc.y += r.y * invDist3;
            acc.z += r.z * invDist3;
        }
        
        // 更新速度
        bodies[idx].vel.x += acc.x * dt;
        bodies[idx].vel.y += acc.y * dt;
        bodies[idx].vel.z += acc.z * dt;
    }
}

// 主程式
int main()
{
    const int nBodies = 30000;
    const int nSteps = 100;
    const float dt = 0.01f;
    
    // 配置記憶體
    Body* d_bodies;
    cudaMalloc(&d_bodies, nBodies * sizeof(Body));
    
    // 設定執行組態
    int threadsPerBlock = 256;
    int blocks = (nBodies + threadsPerBlock - 1) / threadsPerBlock;
    
    // 時間步進
    for(int step = 0; step < nSteps; step++)
    {
        bodyForce<<<blocks, threadsPerBlock>>>(d_bodies, dt, nBodies);
        cudaDeviceSynchronize();
    }
    
    cudaFree(d_bodies);
    return 0;
}
```

### 7.2 矩陣乘法最佳化

以下矩陣乘法先把 tile 搬進 shared memory：

```cpp
#define TILE_SIZE 16

__global__ 
void matrixMul(float* A, float* B, float* C, int N)
{
    __shared__ float ds_A[TILE_SIZE][TILE_SIZE];
    __shared__ float ds_B[TILE_SIZE][TILE_SIZE];
    
    int bx = blockIdx.x;  int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;
    
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;
    
    float sum = 0.0f;
    
    for(int phase = 0; phase < N/TILE_SIZE; phase++)
    {
        // 協作載入資料到共享記憶體
        ds_A[ty][tx] = A[row * N + phase * TILE_SIZE + tx];
        ds_B[ty][tx] = B[(phase * TILE_SIZE + ty) * N + col];
        __syncthreads();
        
        // 計算部分乘積
        for(int k = 0; k < TILE_SIZE; k++)
        {
            sum += ds_A[ty][k] * ds_B[k][tx];
        }
        __syncthreads();
    }
    
    C[row * N + col] = sum;
}
```

[^cuda-guide]: [NVIDIA CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/), 本文執行模型、記憶體階層、同步與 kernel launch 語意的主要規格來源。
