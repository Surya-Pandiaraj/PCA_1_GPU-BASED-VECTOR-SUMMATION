### NAME: SURYA P <br>
### REG NO: 212224230280 <br> 
### Date: 27/10/2026

## EX. No. 1 : GPU BASED VECTOR SUMMATION

<h1> <align=center> SUM ARRAY ON HOST AND DEVICE </h3>
PCA-GPU-based-vector-summation.-Explore-the-differences.
i) Using the program sumArraysOnGPU-timer.cu, set the block.x = 1023. Recompile and run it. Compare the result with the execution configuration of block.x = 1024. Try to explain the difference and the reason.

ii) Refer to sumArraysOnGPU-timer.cu, and let block.x = 256. Make a new kernel to let each thread handle two elements. Compare the results with other execution confi gurations.

## AIM:

To perform vector addition on host and device.

## EQUIPMENTS REQUIRED:
Hardware – PCs with NVIDIA GPU & CUDA NVCC
Google Colab with NVCC Compiler

## PROCEDURE:

1. Initialize the device and set the device properties.
2. Allocate memory on the host for input and output arrays.
3. Initialize input arrays with random values on the host.
4. Allocate memory on the device for input and output arrays, and copy input data from host to device.
5. Launch a CUDA kernel to perform vector addition on the device.
6. Copy output data from the device to the host and verify the results against the host's sequential vector addition. Free memory on the host and the device.

## PROGRAM:
```c
%%writefile vector_add.cu
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>

// CUDA error checking macro
#define CHECK(call)                                                   \
{                                                                     \
    const cudaError_t error = call;                                   \
    if (error != cudaSuccess)                                         \
    {                                                                 \
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);        \
        fprintf(stderr, "code: %d, reason: %s\n",                    \
                error, cudaGetErrorString(error));                    \
        exit(1);                                                      \
    }                                                                 \
}

// Timer function
double seconds()
{
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1.0e-6);
}

// Compare CPU and GPU results
void checkResult(float *hostRef, float *gpuRef, const int N)
{
    double epsilon = 1.0E-5;
    bool match = true;

    for (int i = 0; i < N; i++)
    {
        if (fabs(hostRef[i] - gpuRef[i]) > epsilon)
        {
            match = false;
            printf("Arrays do not match!\n");
            printf("host %f gpu %f at index %d\n",
                   hostRef[i], gpuRef[i], i);
            break;
        }
    }

    if (match)
        printf("Arrays match.\n\n");
}

// Initialize array with random values
void initialData(float *ip, int size)
{
    srand((unsigned)time(NULL));

    for (int i = 0; i < size; i++)
    {
        ip[i] = (float)(rand() & 0xFF) / 10.0f;
    }
}

// CPU vector addition
void sumArraysOnHost(float *A, float *B, float *C, const int N)
{
    for (int i = 0; i < N; i++)
    {
        C[i] = A[i] + B[i];
    }
}

// GPU kernel
__global__ void sumArraysOnGPU(float *A, float *B, float *C, const int N)
{
    int i = (blockIdx.x * blockDim.x + threadIdx.x)*2;

    if (i < N)
        C[i] = A[i] + B[i];
    if (i + 1 < N)
        C[i + 1] = A[i + 1] + B[i + 1];
}

int main()
{
    printf("CUDA Vector Addition Starting...\n");

    // Select GPU
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("Using Device %d: %s\n", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));

    // Vector size
    int nElem = 1 << 24;   // 16,777,216 elements
    printf("Vector size: %d\n", nElem);

    size_t nBytes = nElem * sizeof(float);

    // Allocate host memory
    float *h_A     = (float *)malloc(nBytes);
    float *h_B     = (float *)malloc(nBytes);
    float *hostRef = (float *)malloc(nBytes);
    float *gpuRef  = (float *)malloc(nBytes);

    double iStart, iElaps;

    // Initialize input data
    iStart = seconds();
    initialData(h_A, nElem);
    initialData(h_B, nElem);
    iElaps = seconds() - iStart;
    printf("initialData elapsed: %f sec\n", iElaps);

    memset(hostRef, 0, nBytes);
    memset(gpuRef, 0, nBytes);

    // CPU computation
    iStart = seconds();
    sumArraysOnHost(h_A, h_B, hostRef, nElem);
    iElaps = seconds() - iStart;
    printf("CPU elapsed: %f sec\n", iElaps);

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    CHECK(cudaMalloc((void **)&d_A, nBytes));
    CHECK(cudaMalloc((void **)&d_B, nBytes));
    CHECK(cudaMalloc((void **)&d_C, nBytes));

    // Copy data to device
    CHECK(cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice));

    // Kernel launch configuration
    int threads = 256;
    dim3 block(threads);
    dim3 grid((nElem + (block.x * 2) - 1) / (block.x * 2));

    // GPU computation
    iStart = seconds();
    sumArraysOnGPU<<<grid, block>>>(d_A, d_B, d_C, nElem);
    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;
    printf("GPU elapsed: %f sec\n", iElaps);

    // Check for kernel errors
    CHECK(cudaGetLastError());

    // Copy result back
    CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));

    // Verify results
    checkResult(hostRef, gpuRef, nElem);

    // Free device memory
    CHECK(cudaFree(d_A));
    CHECK(cudaFree(d_B));
    CHECK(cudaFree(d_C));

    // Free host memory
    free(h_A);
    free(h_B);
    free(hostRef);
    free(gpuRef);

    return 0;
}
```

## OUTPUT:
### ilen=256
<img width="336" height="157" alt="image" src="https://github.com/user-attachments/assets/c791e307-fac3-41e8-bf44-44039fbe03fe" />

### ilen=512
<img width="321" height="146" alt="image" src="https://github.com/user-attachments/assets/0983c457-ab5e-4d7f-9f8f-1c44456d7bbe" />

### ilen=1023
<img width="336" height="157" alt="image" src="https://github.com/user-attachments/assets/ef49a575-fc7e-46ca-a685-ded93c578566" />

### ilen=1024
<img width="327" height="152" alt="image" src="https://github.com/user-attachments/assets/dd834744-16bd-4e49-928a-c463fbe3a025" />

## Comparison of Block Sizes:

| Block Size | Number of Blocks | GPU Time (sec) | Result |
|---:|---:|---:|---|
| 256 | 65536 | 0.001160 | Arrays match |
| 512 | 32768 | 0.001707 | Arrays match |
| 1023 | 16401 | 0.000964 | Arrays match |
| 1024 | 16384 | 0.000936 | Arrays match |

## Explanation:

- **256 threads:** Requires more blocks to process the array.
- **512 threads:** Uses fewer blocks than 256, but takes slightly more time in this run.
- **1023 threads:** Is not divisible by 32, so the last warp has one inactive thread.
- **1024 threads:** Is divisible by 32 and requires the fewest blocks among these configurations.

## Conclusion:

All four configurations produce correct results. The 1024-thread block is the fastest in the given output. Small timing differences may occur due to GPU scheduling and measurement variations.
Thus, Implementation of sum arrays on host and device is done in nvcc cuda using random number.
