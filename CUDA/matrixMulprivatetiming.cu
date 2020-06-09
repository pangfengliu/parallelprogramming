#include <stdio.h>
#include <cuda.h>
#include <assert.h>

#ifndef N
#define N 1024
#endif

#define b 8

#define CHECKC

__global__ void matrixMul(int A[N][N], int B[N][N], int C[N][N])
{
  int k, sum = 0;
  int row = blockIdx.x * b + threadIdx.x;
  int column = blockIdx.y * b + threadIdx.y;
  int r;
  __shared__ int sA[b][b];
  __shared__ int sB[b][b];
  for (r = 0; r < N / b; r++) {
    sA[threadIdx.x][threadIdx.y] = A[row][r * b + threadIdx.y];
    sB[threadIdx.x][threadIdx.y] = B[r * b + threadIdx.x][column];
    __syncthreads();
    for (k = 0; k < b; k++)
      sum += sA[threadIdx.x][k] * sB[k][threadIdx.y];
    __syncthreads();
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
  printf("the multiplcaition takes %f seconds\n", time / 1000);
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
			
