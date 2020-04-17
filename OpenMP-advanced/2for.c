#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <unistd.h>		/* for sleep */
/* main */
int main(int argc, char *argv[])
{
  assert(argc == 3);
  omp_set_num_threads(atoi(argv[1]));
  int n = atoi(argv[2]);
  printf("# of proc = %d\n", omp_get_num_procs());
  printf("# of loop iterations = %d\n", n);
  /* twoloop */
  double t = omp_get_wtime();
#pragma omp parallel
  {
#pragma omp for
    for (int i = 0; i < n; i++) 
      sleep(i);
#pragma omp for
    for (int i = n - 1; i >= 0; i--) 
      sleep(i);
  }
  printf("time = %f\n", omp_get_wtime() - t);
  /* end */
  return 0;
}

