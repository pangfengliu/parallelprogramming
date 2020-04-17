#include <omp.h>
#include <stdio.h>

#define N 100000000

int main()
{
  double area, pi, x;
  int i, p;
  double t;

  t = omp_get_wtime();
  area = 0.0;

#pragma omp parallel for private(x) reduction(+:area) 
  for (i = 0; i < N; i++) 
#ifdef BIGCRITICAL
#pragma omp critical
#endif
    {
      x = (i + 0.5) / N;
#ifdef SMALLCRITICAL
#pragma omp critical
#endif
      area += 4.0/(1.0 + x*x);
    }
  pi = area / N;
  t = omp_get_wtime() - t;
  printf("time is %lf\n", t);

  printf("pi = %.16lf\n", pi);
  return 0;
}
