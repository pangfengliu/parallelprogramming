#include <stdio.h>
#include <cuda.h>

#define N 4
__global__ void hello(int int_array[4][N][N])
{
  int_array[blockIdx.x][threadIdx.x][threadIdx.y]
    *= ((threadIdx.x + threadIdx.y) * blockIdx.x);
}

int main(void)
{
  int *device_int_array;
  int size = sizeof(int) * N * N * 4;
  int host_int_array[4][N][N];
  dim3 blocks (N, N);
  cudaMalloc((void **)&device_int_array, size);
  for (int i = 0; i < N; i++)
    for (int j = 0; j < N; j++)
      for (int k = 0; k < 4; k++)
	host_int_array[k][i][j] = i + j;
  cudaMemcpy(device_int_array, host_int_array, size,
	     cudaMemcpyHostToDevice);
  hello <<< 4, blocks >>> ((int (*)[N][N])device_int_array);
  cudaMemcpy(host_int_array, device_int_array, size,
	     cudaMemcpyDeviceToHost);
  for (int k = 0; k < 4; k++)
    for (int i = 0; i < N; i++)
      for (int j = 0; j < N; j++)
	printf("host_int_array[%d][%d][%d] = %d\n", k, i, j,
	       host_int_array[k][i][j]);
  cudaFree(device_int_array);
}
