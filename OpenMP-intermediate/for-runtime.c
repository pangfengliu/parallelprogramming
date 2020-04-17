/* header */
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <unistd.h>		/* for sleep */
/* main */
int main(int argc, char *argv[])
{
  assert(argc == 5);
  omp_set_num_threads(atoi(argv[1]));
  int n = atoi(argv[2]);
  printf("# of proc = %d\n", omp_get_num_procs());
  printf("# of loop iterations = %d\n", n);

  int policy = atoi(argv[3]);
  int chunk = atoi(argv[4]);
  omp_set_schedule(policy, chunk);
  /* loop */
  int elapsedTime = 0;
#pragma omp parallel for firstprivate(elapsedTime) \
schedule(runtime)
  for (int i = 0; i < n; i++) {
    sleep(i);
    elapsedTime += i;
    printf("thread %d i %d elapsed time %d.\n",
	   omp_get_thread_num(), i, elapsedTime);
  }
  return 0;
}
/* end */
