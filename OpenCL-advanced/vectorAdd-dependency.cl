__kernel void add(__global int* matrixA, 
		  __global int* matrixB, 
		  __global int* matrixC) 
{
  int idx = get_global_id(0);
  // for (int i = 0; i < 10000; i++)
  matrixC[idx] = matrixA[idx] + matrixB[idx];
}

