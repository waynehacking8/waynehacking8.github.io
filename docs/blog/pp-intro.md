---
description: "CUDA C/C++ guide covering the execution model, memory hierarchy, occupancy, profiling, streams, synchronization, and kernel optimization."
date: "2025-01-11"
updated: "2026-07-21"
language: "en"
image: "https://wayne.is-a.dev/assets/blog/cuda-platform.webp"
tags:
  - CUDA
  - Parallel Programming
  - Performance
---

# Intro to CUDA Programming

*2025-01-11 · GPU / parallel programming*

<figure class="pb-article-hero">
  <img src="/assets/blog/cuda-platform.webp" alt="NVIDIA CUDA accelerated computing official visual" loading="eager" decoding="async">
  <figcaption>CUDA accelerated computing platform · <a href="https://developer.nvidia.com/cuda-zone">Source: NVIDIA Developer</a></figcaption>
</figure>

This guide starts from the execution model, memory hierarchy and synchronization to explain what CUDA costs in hardware, then uses profiling to decide whether to change data access, synchronization or the kernel configuration. The semantics of the execution model, memory hierarchy and synchronization follow the NVIDIA CUDA C++ Programming Guide. The examples are deliberately shortened, and a production kernel still needs bounds checks and error handling.[^cuda-guide]

## 1. CUDA Basics

### 1.1 The CUDA Programming Model

#### 1.1.1 Compute Architecture
CUDA (Compute Unified Device Architecture) is a parallel computing platform and programming model developed by NVIDIA. Its architecture has these parts:

- **Host**: the CPU and its memory
- **Device**: the GPU and its memory
- **Execution units**: multiple Streaming Multiprocessors (SMs)

#### 1.1.2 Program Flow
A typical CUDA program runs like this:

1. Allocate host and device memory
2. Copy data from host to device
3. Launch the CUDA kernel to compute
4. Copy the results from device back to host
5. Free memory

### 1.2 Writing Kernel Functions

#### 1.2.1 Function Type Qualifiers
CUDA provides three function type qualifiers:

1. **__global__**
   - Runs on the device
   - Callable from the host or the device
   - Must return `void`
   ```cpp
   __global__ void kernelFunction(float* data)
   {
       // device code
   }
   ```

2. **__device__**
   - Runs on the device
   - Callable from the device only
   ```cpp
   __device__ float deviceFunction(float x)
   {
       return x * x;
   }
   ```

3. **__host__**
   - Runs on the host
   - Callable from the host only
   ```cpp
   __host__ void hostFunction(float* data)
   {
       // host code
   }
   ```

#### 1.2.2 Kernel Launch Syntax

A kernel is launched like this:
```cpp
kernelFunction<<<gridDim, blockDim, sharedMemBytes, stream>>>(parameters);
```

Parameters:
- `gridDim`: grid dimensions, the number of blocks
- `blockDim`: block dimensions, the number of threads per block
- `sharedMemBytes`: dynamic shared memory size (optional)
- `stream`: the CUDA stream (optional)

### 1.3 The NVCC Compiler

#### 1.3.1 Compilation Flow
What NVCC does:

1. Separate host and device code
2. Compile the device code into PTX or cubin
3. Compile the host code
4. Link all object files

#### 1.3.2 Common Compiler Options

```bash
nvcc -arch=sm_70        # target architecture
     -O3                # optimization level
     -G                # add debug info
     -lineinfo         # add line number info
     -o output.exe     # output file
     source.cu         # source file
```

Notes on the important options:
- `-arch`: the GPU architecture version
- `-code`: the GPU code version actually generated
- `-Xptxas`: pass options to the PTX assembler
- `-maxrregcount`: limit the number of registers per thread

## 2. Memory Management and Optimization

### 2.1 Memory Types

#### 2.1.1 Global Memory
- The largest memory space
- Accessible by all threads
- Higher access latency

Allocation and use:
```cpp
// the traditional way
float *d_data;
cudaMalloc(&d_data, size);
cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);

// with unified memory
float *u_data;
cudaMallocManaged(&u_data, size);
```

#### 2.1.2 Shared Memory
- Shared by the threads in a block
- Access speed close to registers
- Limited capacity

Declaration and use:
```cpp
__global__ void sharedMemKernel()
{
    __shared__ float sharedData[256];
    sharedData[threadIdx.x] = someData;
    __syncthreads();
    // use the shared data
}
```

#### 2.1.3 Register Memory
- Private to each thread
- The fastest memory type
- Limited in number

Example:
```cpp
__global__ void registerKernel()
{
    float localData = 0;  // allocated to a register automatically
    // do the computation
}
```

### 2.2 Unified Memory

#### 2.2.1 Basics
Unified Memory gives the CPU and GPU one shared virtual address space. The runtime migrates pages on demand and allows allocating more data than a single GPU's physical memory, but page faults can become a performance bottleneck.

#### 2.2.2 Prefetching
Prefetch the pages to the current GPU first:

```cpp
// basic prefetch
cudaMemPrefetchAsync(data, size, deviceId);

// a more complete prefetch strategy
void optimizedPrefetch(float* data, size_t size)
{
    int deviceId;
    cudaGetDevice(&deviceId);
    
    // prefetch data to the GPU
    cudaMemPrefetchAsync(data, size, deviceId);
    
    // run the kernel
    computeKernel<<<blocks, threads>>>(data);
    
    // prefetch the result back to the CPU
    cudaMemPrefetchAsync(data, size, cudaCpuDeviceId);
}
```

#### 2.2.3 Access Patterns
Adjacent threads should read adjacent addresses:

1. **Coalesced access**
```cpp
__global__ void coalescedAccess(float* data)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float value = data[idx];  // coalesced access
}
```

2. **Avoid divergent access**
```cpp
__global__ void strideAccess(float* data, int stride)
{
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * stride;
    float value = data[idx];  // may cause uncoalesced access
}
```

### 2.3 Memory Optimization Tips

#### 2.3.1 Alignment
Align the data to reduce split memory transactions:

```cpp
// struct alignment
struct __align__(16) AlignedStruct {
    float x, y, z, w;
};

// allocate aligned memory dynamically
void* alignedMalloc(size_t size, size_t alignment)
{
    size_t mask = alignment - 1;
    void* p;
    cudaMalloc(&p, size + mask);
    void* aligned = (void*)(((uintptr_t)p + mask) & ~mask);
    return aligned;
}
```

#### 2.3.2 Overlapping Tiles
Load the halo region a stencil needs:

```cpp
__global__ void overlappedKernel(float* in, float* out)
{
    __shared__ float tile[TILE_SIZE + 2];  // includes the overlap region
    
    // load the data, boundaries included
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    tile[threadIdx.x + 1] = in[idx];
    
    // load the boundary elements
    if (threadIdx.x == 0)
        tile[0] = in[idx - 1];
    if (threadIdx.x == blockDim.x - 1)
        tile[threadIdx.x + 2] = in[idx + 1];
        
    __syncthreads();
    
    // compute with the overlap region
    out[idx] = tile[threadIdx.x + 1] + tile[threadIdx.x] + tile[threadIdx.x + 2];
}
```

## 3. Thread Organization and Management

### 3.1 Thread Hierarchy

#### 3.1.1 Grids, Blocks and Threads
`dim3` builds 1D, 2D or 3D grids and blocks:

```cpp
// 1D configuration
dim3 grid(numBlocks);
dim3 block(numThreads);

// 2D configuration
dim3 grid(numBlocksX, numBlocksY);
dim3 block(numThreadsX, numThreadsY);

// 3D configuration
dim3 grid(numBlocksX, numBlocksY, numBlocksZ);
dim3 block(numThreadsX, numThreadsY, numThreadsZ);

// launch the kernel
kernel<<<grid, block>>>(parameters);
```

#### 3.1.2 Index Calculation

Multi-dimensional index calculation:

```cpp
__global__ void multiDimKernel(float* data)
{
    // 1D index
    int idx1D = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 2D index
    int idx2D = (blockIdx.y * gridDim.x + blockIdx.x) * (blockDim.x * blockDim.y) +
                 threadIdx.y * blockDim.x + threadIdx.x;
                 
    // 3D index
    int idx3D = ((blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x) *
                (blockDim.x * blockDim.y * blockDim.z) +
                (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;
}
```

### 3.2 Thread Synchronization

#### 3.2.1 Synchronization Within a Block
Use `__syncthreads()` to synchronize the threads in a block:

```cpp
__global__ void syncKernel(float* data)
{
    __shared__ float sharedData[256];
    
    // load the data
    sharedData[threadIdx.x] = data[threadIdx.x];
    __syncthreads();  // make sure all data is loaded
    
    // use the data
    float result = sharedData[255 - threadIdx.x];
    __syncthreads();  // make sure all threads have finished reading
    
    data[threadIdx.x] = result;
}
```

#### 3.2.2 Cooperative Groups
Cooperative Groups let you write the synchronization scope explicitly in the code:

```cpp
#include <cooperative_groups.h>
using namespace cooperative_groups;

__global__ void coopKernel(float* data)
{
    // get the current thread group
    thread_block block = this_thread_block();
    
    // group synchronization
    block.sync();
    
    // group size
    int size = block.size();
    
    // thread index within the group
    int idx = block.thread_rank();
}
```

### 3.3 Tuning the Launch Configuration

#### 3.3.1 Estimating Block Size and Occupancy

The CUDA occupancy API can give a candidate block size first, but the final choice still comes from measurement:

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

#### 3.3.2 Dynamic Parallelism
A dynamic parallelism example:

```cpp
__global__ void dynamicParallelKernel(float* data, int depth)
{
    if (depth > 0)
    {
        // launch a new kernel recursively
        dynamicParallelKernel<<<gridDim, blockDim>>>(data, depth - 1);
        cudaDeviceSynchronize();
    }
    
    // process the data
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    data[idx] *= 2.0f;
}
```

## 4. Profiling and Optimization

### 4.1 Profiling with nsys

#### 4.1.1 Basic Profiling
Use `nsys` to collect the timeline and CUDA API statistics:

```bash
# basic profiling
nsys profile --stats=true ./myapp

# detailed memory analysis
nsys profile --trace=cuda,nvtx,osrt,cublas ./myapp

# generate a timeline report
nsys profile --trace=cuda --force-overwrite true \
             --delay=0 --duration=0 \
             --output=timeline ./myapp
```

#### 4.1.2 Reading the Report

The nsys report contains this key information:

1. **CUDA API statistics**
   - Kernel launch time
   - Memory operation time
   - API call frequency

2. **CUDA kernel statistics**
   - Kernel execution time
   - Performance metrics per kernel
   - Hardware resource utilization

3. **Memory operation statistics**
   - Memory transfer size
   - Number of page faults
   - Data migration time

Example report:
```plaintext
 Time(%)  Total Time (ns)  Num Calls    Avg (ns)   Med (ns)    Min (ns)    Max (ns)   StdDev (ns)
 -------  ---------------  ---------  -----------  ----------  ----------  -----------  -----------
    45.3       2,345,678        100     23,456.8    22,345.6     19,876     28,987.6      1,234.5
```

### 4.2 Finding Bottlenecks

#### 4.2.1 Memory Bottlenecks
Measure the host-to-device copy time first:

```cpp
// use events to time the memory operation
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

#### 4.2.2 Compute Bottlenecks
Estimate arithmetic intensity roughly as FLOPs/byte:

```cpp
// compute arithmetic intensity
float computeArithmeticIntensity(int operations, int bytes_accessed)
{
    return (float)operations / bytes_accessed;
}

// time the kernel with events
__global__ void computeKernel(float* data, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
    {
        // record the compute operations and memory traffic
        int compute_ops = 0;
        int memory_accesses = 0;
        
        float temp = data[idx];  // memory read
        memory_accesses += sizeof(float);
        
        for(int i = 0; i < 100; i++)
        {
            temp = temp * temp + temp;  // compute operation
            compute_ops += 2;
        }
        
        data[idx] = temp;  // memory write
        memory_accesses += sizeof(float);
    }
}
```

### 4.3 Optimization Strategies

#### 4.3.1 Memory Optimization
Put values that get reused into shared memory:

```cpp
__global__ void optimizedMemoryKernel(float* data, int N)
{
    // use shared memory to cut global memory accesses
    __shared__ float sharedData[BLOCK_SIZE];
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // coalesced reads from global memory
    if (idx < N)
        sharedData[threadIdx.x] = data[idx];
    
    __syncthreads();
    
    // compute from shared memory
    if (threadIdx.x < blockDim.x - 1)
    {
        float result = sharedData[threadIdx.x] + sharedData[threadIdx.x + 1];
        data[idx] = result;
    }
}
```

#### 4.3.2 Compute Optimization
When the loop length is fixed, let the compiler unroll the loop:

```cpp
__global__ void optimizedComputeKernel(float* data, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
    {
        // loop unrolling
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

## 5. Streams and Asynchronous Execution

### 5.1 CUDA Stream Basics

#### 5.1.1 Creating and Managing Streams
```cpp
// create several streams
cudaStream_t streams[4];
for(int i = 0; i < 4; i++)
{
    cudaStreamCreate(&streams[i]);
}

// run kernels in different streams
for(int i = 0; i < 4; i++)
{
    int offset = i * N/4;
    processKernel<<<blocks, threads, 0, streams[i]>>>(
        d_data + offset, N/4
    );
}

// synchronize and clean up
for(int i = 0; i < 4; i++)
{
    cudaStreamSynchronize(streams[i]);
    cudaStreamDestroy(streams[i]);
}
```

#### 5.1.2 Stream Callbacks

Run a host callback after the stream completes:

```cpp
void CUDART_CB callbackFunc(cudaStream_t stream, cudaError_t status, void* userData)
{
    printf("Stream %p completed with status %d\n", stream, status);
}

// register the callback
cudaStreamAddCallback(stream, callbackFunc, nullptr, 0);
```

### 5.2 Stream Synchronization

#### 5.2.1 Event Synchronization
Events create dependencies across streams:

```cpp
// create the event
cudaEvent_t event;
cudaEventCreate(&event);

// record the event in a stream
cudaEventRecord(event, stream1);

// make other streams wait for the event
cudaStreamWaitEvent(stream2, event);
cudaStreamWaitEvent(stream3, event);

// clean up
cudaEventDestroy(event);
```

#### 5.2.2 Stream Priorities
Setting a stream priority:

```cpp
// create a high priority stream
cudaStream_t highPriorityStream;
int priority;
cudaDeviceGetStreamPriorityRange(nullptr, &priority);
cudaStreamCreateWithPriority(&highPriorityStream, 
                            cudaStreamNonBlocking, 
                            priority);
```

## 6. Advanced Optimization

### 6.1 Dynamic Parallelism

A device-side kernel launches child kernels:

```cpp
__global__ void recursiveKernel(int* data, int depth)
{
    if (depth > 0)
    {
        // launch a child kernel dynamically
        recursiveKernel<<<gridDim.x, blockDim.x>>>(
            data + blockIdx.x * blockDim.x, 
            depth - 1
        );
    }
    
    // process the data at the current level
    int idx = threadIdx.x;
    data[idx] = blockIdx.x * blockDim.x + idx;
}
```

### 6.2 Atomic Operations

Use atomic operations when several threads update the same location:

```cpp
__global__ void atomicKernel(int* counter, int* data, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
    {
        // count with an atomic operation
        int old = atomicAdd(counter, 1);
        
        // conditional update with an atomic operation
        atomicMax(&data[idx], old);
    }
}
```

## 7. Worked Examples

### 7.1 N-Body Simulation

The example below computes the forces in an N-body problem:

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
        
        // compute the gravitational force
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
        
        // update the velocity
        bodies[idx].vel.x += acc.x * dt;
        bodies[idx].vel.y += acc.y * dt;
        bodies[idx].vel.z += acc.z * dt;
    }
}

// main program
int main()
{
    const int nBodies = 30000;
    const int nSteps = 100;
    const float dt = 0.01f;
    
    // allocate memory
    Body* d_bodies;
    cudaMalloc(&d_bodies, nBodies * sizeof(Body));
    
    // set the launch configuration
    int threadsPerBlock = 256;
    int blocks = (nBodies + threadsPerBlock - 1) / threadsPerBlock;
    
    // time stepping
    for(int step = 0; step < nSteps; step++)
    {
        bodyForce<<<blocks, threadsPerBlock>>>(d_bodies, dt, nBodies);
        cudaDeviceSynchronize();
    }
    
    cudaFree(d_bodies);
    return 0;
}
```

### 7.2 Matrix Multiplication

The matrix multiplication below first moves tiles into shared memory:

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
        // cooperatively load data into shared memory
        ds_A[ty][tx] = A[row * N + phase * TILE_SIZE + tx];
        ds_B[ty][tx] = B[(phase * TILE_SIZE + ty) * N + col];
        __syncthreads();
        
        // compute the partial product
        for(int k = 0; k < TILE_SIZE; k++)
        {
            sum += ds_A[ty][k] * ds_B[k][tx];
        }
        __syncthreads();
    }
    
    C[row * N + col] = sum;
}
```

[^cuda-guide]: [NVIDIA CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/), the main specification source for the execution model, memory hierarchy, synchronization and kernel launch semantics in this post.
