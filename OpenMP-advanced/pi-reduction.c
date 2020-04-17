#include <omp.h>
#include <stdio.h>
#define N 100000000
int main()
{
  /* loop */
  double x;
  double area = 0.0;
  double t = omp_get_wtime();
#pragma omp parallel for private(x) \
  reduction(+ : area)
  for (int i = 0; i < N; i++) {
    x = (i + 0.5) / N;
    area += 4.0 / (1.0 + x * x);
  }
  double pi = area / N;
  t = omp_get_wtime() - t;
  printf("execution time is %f\n", t);
  printf("pi = %f\n", pi);
  /* end */
  return 0;
}
