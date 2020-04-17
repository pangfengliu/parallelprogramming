#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

int main(int argc, char *argv[])
{
  /* main */
  assert(argc == 3);
  omp_set_num_threads(atoi(argv[1]));
  int n = atoi(argv[2]);
  printf("# of proc = %d\n", omp_get_num_procs());
  printf("# of loop iterations = %d\n", n);
  int v = 101;
  printf("before the loop thread %d with v = %d.\n",
	 omp_get_thread_num(), v);
#pragma omp parallel for firstprivate(v)
  for (int i = 0; i < n; i++) {
    v += i;
    printf("thread %d runs index %d with v = %d.\n",
	   omp_get_thread_num(), i, v);
  }
  printf("after the loop thread %d with v = %d.\n",
	 omp_get_thread_num(), v);
  /* end */
  return 0;
}

