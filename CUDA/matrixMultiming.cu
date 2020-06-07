#include <stdio.h>
#include <assert.h>
#include <cuda.h>

#ifndef N
#define N 10
#endif

#define b 10

#define PRINTC
#define CHECKC


__global__ void hello(int A[N][N], int B[N][N], int C[N][N])
{
  int k, sum = 0;
  for (k = 0; k < N; k++)
    sum += A[threadIdx.x][k] * B[k][threadIdx.y];

  C[threadIdx.x][threadIdx.y] = sum;
}

int main(void)
{
  int *device_A, *device_B, *device_C;
  int *host_A, *host_B, *host_C;
  int size = sizeof(int) * N * N;
  int *aptr, *bptr;

  cudaMalloc((void **)&device_A, size);
  cudaMalloc((void **)&device_B, size);
  cudaMalloc((void **)&device_C, size);
  host_A = (int *)malloc(size);
  host_B = (int *)malloc(size);
  host_C = (int *)malloc(size);

  aptr = host_A;
  bptr = host_B;
  for (int i = 0; i < N; i++)
    for (int j = 0; j < N; j++) {
      *aptr++ = *bptr++ = ((i == j)? 1 : 0);
    }

  cudaMemcpy(device_A, host_A, size, cudaMemcpyHostToDevice);
  cudaMemcpy(device_B, host_B, size, cudaMemcpyHostToDevice);

  float time;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

  dim3 block(b, b);
  dim3 grid(N / b, N / b);
  hello <<< grid, block >>> ((int (*)[N])device_A, (int (*)[N])device_B,
			   (int (*)[N])device_C);
  cudaMemcpy(host_C, device_C, size, cudaMemcpyDeviceToHost);
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time, start, stop);
  printf("the multiplcaition takes %f seconds\n", time);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  int k;
#ifdef PRINTC
  k = 0;
  for (int i = 0; i < N; i++)
    for (int j = 0; j < N; j++)
      printf("host_C[%d][%d] = %d\n", i, j, host_C[k++]);
#endif
#ifdef CHECKC
  k = 0;
  for (int i = 0; i < N; i++)
    for (int j = 0; j < N; j++) 
      assert(host_C[k++] == ((i == j)? 1 : 0));
#endif
  cudaFree(device_A);
  cudaFree(device_B);
  cudaFree(device_C);
  free(host_A);
  free(host_B);
  free(host_C);
}
			
