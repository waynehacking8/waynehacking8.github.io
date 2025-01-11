---
style: |
  .language-cpp {
    background-color: #f6f8fa;
    border-radius: 3px;
    padding: 16px;
  }
  .language-bash {
    background-color: #282c34;
    color: #abb2bf;
    border-radius: 3px;
    padding: 16px;
  }
  .language-plaintext {
    background-color: #f5f5f5;
    border-radius: 3px;
    padding: 16px;
  }
---

# CUDA C/C++ 完整技術指南

## 1. CUDA 基礎架構

### 1.1 CUDA程式設計模型

#### 1.1.1 運算架構
CUDA (Compute Unified Device Architecture) 是NVIDIA開發的平行運算平台與程式設計模型。其架構包含：

- **主機 (Host)**：CPU及其記憶體
- **設備 (Device)**：GPU及其記憶體
- **執行單元**：包含多個串流多處理器 (Streaming Multiprocessors, SMs)

#### 1.1.2 程式執行流程
CUDA程式的典型執行流程：

1. 配置主機和設備記憶體
2. 將資料從主機複製到設備
3. 呼叫CUDA核心進行運算
4. 將結果從設備複製回主機
5. 釋放記憶體

### 1.2 核心函數開發

#### 1.2.1 函數類型限定詞
CUDA提供三種函數類型限定詞：

1. **__global__**
   - 在設備端執行
   - 可從主機端或設備端呼叫
   - 必須返回void
   ```cpp
   __global__ void kernelFunction(float* data)
   {
       // 設備端程式碼
   }
   ```

2. **__device__**
   - 在設備端執行
   - 只能從設備端呼叫
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

完整的核心函數呼叫語法：
```cpp
kernelFunction<<<gridDim, blockDim, sharedMemBytes, stream>>>(parameters);
```

參數說明：
- `gridDim`：網格維度，指定區塊數量
- `blockDim`：區塊維度，指定每個區塊的執行緒數量
- `sharedMemBytes`：動態共享記憶體大小（可選）
- `stream`：CUDA串流（可選）

### 1.3 NVCC編譯器

#### 1.3.1 編譯流程
NVCC編譯器的工作流程：

1. 分離主機和設備程式碼
2. 編譯設備程式碼生成PTX或cubin
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
- `-arch`：指定GPU架構版本
- `-code`：指定實際產生的GPU程式碼版本
- `-Xptxas`：傳遞選項給PTX彙編器
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
統一記憶體提供單一記憶體空間視圖：
- 自動管理資料遷移
- 支援過量配置
- 頁面錯誤機制

#### 2.2.2 記憶體預取
使用預取優化效能：

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
最佳化記憶體存取模式：

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
確保資料對齊以提高存取效率：

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
使用記憶體覆蓋減少記憶體使用：

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
完整的執行緒層次配置：

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
使用合作群組進行更靈活的同步：

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

#### 3.3.1 佔用率計算
計算最佳執行組態：

```cpp
void calculateOptimalConfig()
{
    int deviceId;
    cudaGetDevice(&deviceId);
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, deviceId);
    
    int maxThreadsPerSM = prop.maxThreadsPerMultiProcessor;
    int maxBlocksPerSM = prop.maxBlocksPerMultiProcessor;
    int warpSize = prop.warpSize;
    
    // 計算最佳區塊大小
    int blockSize = warpSize * 8;  // 通常是 warp size 的倍數
    if (blockSize > prop.maxThreadsPerBlock)
        blockSize = prop.maxThreadsPerBlock;
        
    // 計算網格大小
    int numSMs = prop.multiProcessorCount;
    int numBlocks = numSMs * maxBlocksPerSM;
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
使用 nsys 收集效能資料：

```bash
# 基本效能分析
nsys profile --stats=true ./myapp

# 詳細記憶體分析
nsys profile --trace=cuda,nvtx,osrt,cublas ./myapp

# 產生時序報告
nsys profile --trace=cuda --force-overwrite true \
             --delay=0 --duration=0 \
             --output=timeline ./myapp

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
識別和解決記憶體瓶頸：

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
分析計算密集度：

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
實作記憶體最佳化策略：

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
實作計算最佳化策略：

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

#### 5.1.2 串流回調函數
實作串流回調：

```cpp
void CUDART_CB callbackFunc(cudaStream_t stream, cudaError_t status, void* userData)
{
    printf("Stream %p completed with status %d\n", stream, status);
}

// 設定回調
cudaStreamAddCallback(stream, callbackFunc, nullptr, 0);
```

### 5.2 串流同步機制

#### 5.2.1 事件同步
使用事件進行串流同步：

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

實作動態平行處理：

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

實作高效率的原子操作：

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

完整的 N體模擬實作：

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

使用共享記憶體的矩陣乘法實作：

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

