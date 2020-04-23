#define N 1024

__kernel void mul(__global int matrixA[N][N], 
		  __global int matrixB[N][N], 
		  __global int matrixC[N][N]) 
{
  int row = get_global_id(0);
  int col = get_global_id(1);
  int sum = 0;
  for (int i = 0; i < N; i++) 
    sum += matrixA[row][i] * matrixB[i][col];
  matrixC[row][col] = sum;
}
