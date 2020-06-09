#include <stdio.h>
#include <cuda.h>

#define N 4
__global__ void hello(int int_array[N][N])
{
  int_array[threadIdx.x][threadIdx.y]
    *= (threadIdx.x + threadIdx.y);
}

int main(void)
{
  int *device_int_array;
  int size = sizeof(int) * N * N;
  int host_int_array[N][N];
  dim3 blocks (N,N);
  cudaMalloc((void **)&device_int_array, size);
  for (int i = 0; i < N; i++)
    for (int j = 0; j < N; j++)
      host_int_array[i][j] = i + j;
  cudaMemcpy(device_int_array, host_int_array, size,
	     cudaMemcpyHostToDevice);
  hello <<< 1, blocks >>> ((int (*)[N])device_int_array);
  cudaMemcpy(host_int_array, device_int_array, size,
	     cudaMemcpyDeviceToHost);
  for (int i = 0; i < N; i++)
    for (int j = 0; j < N; j++)
      printf("host_int_array[%d][%d] = %d\n", i, j,
	     host_int_array[i][j]);
  cudaFree(device_int_array);
}
