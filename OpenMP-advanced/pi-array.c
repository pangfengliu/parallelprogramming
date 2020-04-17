#include <omp.h>
#include <stdio.h>
#define N 100000000
#define MAXT 4
int main()
{
  /* main */
  double x;
  double area[MAXT] = {0.0};
  double t = omp_get_wtime();
#pragma omp parallel for private(x)
  for (int i = 0; i < N; i++) {
    x = (i + 0.5) / N;
    area[omp_get_thread_num()] += 
      4.0 / (1.0 + x * x);
  }
  t = omp_get_wtime() - t;
  double areaSum = 0.0;
  for (int i = 0; i < omp_get_num_procs(); i++)
    areaSum += area[i];
  double pi = areaSum / N;
  /* end */
  printf("execution time is %f\n", t);
  printf("pi = %f\n", pi);
  return 0;
}
