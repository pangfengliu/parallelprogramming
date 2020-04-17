/* header */
#include <stdio.h>
#include <assert.h>
#include "omp.h"

#define N 4
int a[N][N], b[N][N];

int main()
{
  printf("number of processor = %d\n", 
	 omp_get_num_procs());
  /* parallelsections */
#pragma omp parallel 
#pragma omp sections
  {				
#pragma omp section
    {
      printf("thread %d init a\n", 
	     omp_get_thread_num());
      for (int i = 0; i < N; i++)
	for (int j = 0; j < N; j++)
	  a[i][j] = i + j;
    }
#pragma omp section
    {
      printf("thread %d init b\n", 
	     omp_get_thread_num());
      for (int i = 0; i < N; i++)
	for (int j = 0; j < N; j++)
	  b[i][j] = i - j;
    }
  } 
  /* parallelfor */
#pragma omp parallel
#pragma omp for
  for (int i = 0; i < N; i++) {
    printf("thread %d check a\n", 
	   omp_get_thread_num());
    for (int j = 0; j < N; j++)
      assert(a[i][j] == i + j);
  }
#pragma omp for
  for (int i = 0; i < N; i++) {
    printf("thread %d check b\n", 
	   omp_get_thread_num());
    for (int j = 0; j < N; j++)
      assert(b[i][j] == i - j);
  }
  /* end */
  return 0;
}

