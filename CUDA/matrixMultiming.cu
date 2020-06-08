#include <stdio.h>
#include <cuda.h>
#include <assert.h>

#ifndef N
#define N 256
#endif

#define b 16

#define CHECKC

__global__ void matrixMul(int A[N][N], int B[N][N], int C[N][N])
{
  int row = blockIdx.x * b + threadIdx.x;
  int column = blockIdx.y * b + threadIdx.y;
  int sum = 0;
  for (int i = 0; i < N; i++) {
      sum += A[row][i] * B[column][i];
  }
  C[row][column] = sum;
}

int host_A[N][N], host_B[N][N], host_C[N][N];

int main(void)
{
  int *device_A, *device_B, *device_C;
  int size = sizeof(int) * N * N;
  cudaMalloc((void **)&device_A, size);
  cudaMalloc((void **)&device_B, size);
  cudaMalloc((void **)&device_C, size);

  for (int i = 0; i < N; i++)
    for (int j = 0; j < N; j++) {
      host_A[i][j] = host_B[i][j] = ((i == j)? 1 : 0);
    }

  cudaMemcpy(device_A, host_A, size, cudaMemcpyHostToDevice);
  cudaMemcpy(device_B, host_B, size, cudaMemcpyHostToDevice);

  dim3 block(b, b);
  dim3 grid(N / b, N / b);

  float time;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
  matrixMul <<< grid, block >>> ((int (*)[N])device_A, (int (*)[N])device_B,
			   (int (*)[N])device_C);
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time, start, stop);
  printf("the multiplcaition takes %f seconds\n", time);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  cudaMemcpy(host_C, device_C, size, cudaMemcpyDeviceToHost);
#ifdef CHECKC
  for (int i = 0; i < N; i++)
    for (int j = 0; j < N; j++) 
      assert(host_C[i][j] == (i == j)? 1 : 0);
#endif

  cudaFree(device_A);
  cudaFree(device_B);
  cudaFree(device_C);
}
			
