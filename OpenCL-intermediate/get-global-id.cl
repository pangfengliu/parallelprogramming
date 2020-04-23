#define N 16

__kernel void getGlobalId(__global int globalId[2][N][N])
{
  int id0 = get_global_id(0);
  int id1 = get_global_id(1);
  globalId[0][id0][id1] = id0;
  globalId[1][id0][id1] = id1;
}

