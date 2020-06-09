#include <stdio.h>
#include <cuda.h>
#define N 8
__global__ void hello(int int_array[N])
{
  int_array[threadIdx.x] *= threadIdx.x;
}

int main(void)
{
  int *device_int_array;
  int *host_int_array;
  int size = sizeof(int) * N;
  cudaMalloc((void **)&device_int_array, size);
  host_int_array = (int *)malloc(size);
  for (int i = 0; i < N; i++)
    host_int_array[i] = i;
  cudaMemcpy(device_int_array, host_int_array, size,
	     cudaMemcpyHostToDevice);
  hello <<< 1, N >>> (device_int_array);
  cudaMemcpy(host_int_array, device_int_array, size,
	     cudaMemcpyDeviceToHost);
  for (int i = 0; i < N; i++)
    printf("host_int_array[%d] = %d\n", i, host_int_array[i]);
  cudaFree(device_int_array);
  free(host_int_array);
}
