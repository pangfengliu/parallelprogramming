#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define N 999999999
int main()
{
  double *v = malloc(sizeof(double));
#pragma omp parallel for
  for (int i = 1; i <= N; i++)
    *v = sqrt(i);
  return 0;
}
