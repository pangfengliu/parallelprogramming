/* header */
#include <stdio.h>
#include <assert.h>
#include "omp.h"

#define N 8192
int a[N][N], b[N][N];

int main()
{
  double t;
  printf("number of processor = %d\n", 
	 omp_get_num_procs());

  t = omp_get_wtime();
  /* sections */
#pragma omp parallel sections
  {				
#pragma omp section
    {
      printf("thread %d for a\n", omp_get_thread_num());
      for (int i = 0; i < N; i++)
	for (int j = 0; j < N; j++)
	  a[i][j] = i + j;
    } /* section */
#pragma omp section
    {
      printf("thread %d for b\n", omp_get_thread_num());
      for (int i = 0; i < N; i++)
	for (int j = 0; j < N; j++)
	  b[i][j] = i - j;
    } /* section */
  } /* parallel sectsions */
  t = omp_get_wtime() - t;
  /* sectionsend */
  printf("time is %lf\n", t);
  {
    for (int i = 0; i < N; i++)
      for (int j = 0; j < N; j++)
	assert(a[i][j] == i + j);
    
    for (int i = 0; i < N; i++)
      for (int j = 0; j < N; j++)
	assert(b[i][j] == i - j);
  }
  return 0;
}
/* end */
