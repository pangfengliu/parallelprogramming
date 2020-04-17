#include <omp.h>
#include <stdio.h>

int main(void)
{
#pragma omp parallel
  {
    printf("Hello, world.\n");
    printf("# of proc = %d\n", omp_get_num_procs());
    printf("# of threads = %d\n", omp_get_num_threads());
  }
  return 0;
}
