#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <unistd.h>		/* for sleep */

int main(int argc, char *argv[])
{
  assert(argc == 3);
  omp_set_num_threads(atoi(argv[1]));
  int n = atoi(argv[2]);
  printf("# of proc = %d\n", omp_get_num_procs());
  printf("# of loop iterations = %d\n", n);
  int elapsedTime = 0;
  /* loop */
#pragma omp parallel for firstprivate(elapsedTime) \
schedule(guided, 1)
  for (int i = 0; i < n; i++) {
    sleep(i);
    elapsedTime += i;
    printf("thread %d i %d elapsed time %d.\n",
	   omp_get_thread_num(), i, elapsedTime);
  }
  /* loopend */
  return 0;
}

