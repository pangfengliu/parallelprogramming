#include <stdio.h>
#include <cuda.h>

#ifndef N
#define N 32
#endif

#define b 8

__global__ void matrixMul(int A[N][N], int B[N][N], int C[N][N])
{
  int row = blockIdx.x * b + threadIdx.x;
  int column = blockIdx.y * b + threadIdx.y;
  __shared__ int sA[b][b];
  __shared__ int sB[b][b];

  int sum = 0;
  for (int r = 0; r < b; r++) {
    sA[threadIdx.x][threadIdx.y] = A[row][r * b + threadIdx.y];
    sB[threadIdx.x][threadIdx.y] = B[r * b + threadIdx.x][column];
    __syncthreads();
    for (int k = 0; k < b; k++)
      sum += sA[threadIdx.x][k] * sB[k][threadIdx.y];
    __syncthreads();
  }
  C[row][column] = sum;
}


int main(void)
{
  int *device_A, *device_B, *device_C;
  int *host_A, *host_B, *host_C;
  int i, j, k;
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
  for (i = 0; i < N; i++)
    for (j = 0; j < N; j++) {
      *aptr++ = *bptr++ = ((i == j)? 1 : 0);
    }

  cudaMemcpy(device_A, host_A, size, cudaMemcpyHostToDevice);
  cudaMemcpy(device_B, host_B, size, cudaMemcpyHostToDevice);

  dim3 block(b, b);
  dim3 grid(N / b, N / b);
  matrixMul <<< grid, block >>> ((int (*)[N])device_A, (int (*)[N])device_B,
			   (int (*)[N])device_C);
  cudaMemcpy(host_C, device_C, size, cudaMemcpyDeviceToHost);

  k = 0;
  for (i = 0; i < N; i++)
    for (j = 0; j < N; j++)
      printf("host_C[%d][%d] = %d\n", i, j, host_C[k++]);

  cudaFree(device_A);
  cudaFree(device_B);
  cudaFree(device_C);
  free(host_A);
  free(host_B);
  free(host_C);
}
			
