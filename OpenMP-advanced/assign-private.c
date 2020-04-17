#include <omp.h>
#include <stdio.h>
#include <math.h>
#define N 999999999
int main()
{
  double v;
#pragma omp parallel for private(v)
  for (int i = 1; i <= N; i++) 
    v = sqrt(i);
  return 0;
}
