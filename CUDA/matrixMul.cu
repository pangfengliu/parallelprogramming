#include <stdio.h>
#include <cuda.h>

#ifndef Size
#define Size 10
#endif

__global__ void hello(int A[Size][Size], int B[Size][Size], int C[Size][Size])
{
  int k, sum = 0;
  for (k = 0; k < Size; k++)
    sum += A[threadIdx.x][k] * B[k][threadIdx.y];

  C[threadIdx.x][threadIdx.y] = sum;
}

int main(void)
{
  int *device_A, *device_B, *device_C;
  int *host_A, *host_B, *host_C;
  int i, j, k;
  int size = sizeof(int) * Size * Size;
  int *aptr, *bptr;

  dim3 blocks(Size, Size);

  cudaMalloc((void **)&device_A, size);
  cudaMalloc((void **)&device_B, size);
  cudaMalloc((void **)&device_C, size);
  host_A = (int *)malloc(size);
  host_B = (int *)malloc(size);
  host_C = (int *)malloc(size);

  aptr = host_A;
  bptr = host_B;
  for (i = 0; i < Size; i++)
    for (j = 0; j < Size; j++) {
      *aptr++ = *bptr++ = (i == j);
    }

  cudaMemcpy(device_A, host_A, size, cudaMemcpyHostToDevice);
  cudaMemcpy(device_B, host_B, size, cudaMemcpyHostToDevice);

  hello <<< 1, blocks >>> ((int (*)[Size])device_A, (int (*)[Size])device_B,
			   (int (*)[Size])device_C);
  cudaMemcpy(host_C, device_C, size, cudaMemcpyDeviceToHost);

  k = 0;
  for (i = 0; i < Size; i++)
    for (j = 0; j < Size; j++)
      printf("host_C[%d][%d] = %d\n", i, j, host_C[k++]);

  cudaFree(device_A);
  cudaFree(device_B);
  cudaFree(device_C);
  free(host_A);
  free(host_B);
  free(host_C);
}
			
