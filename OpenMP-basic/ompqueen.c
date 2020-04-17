#include "omp.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 12

int position[N];

int ok(int position[], int next, int test)
{
  int i;

  for (i = 0; i < next; i++)
    if (position[i] == test || (abs(test - position[i]) == next - i))
      return 0;

  return 1;
}

void queen(int position[], int next)
{
  int test;

  if (next >= N) {
    print_solution(position);
    return;
  }

  for (test = 0; test < N; test++) 
    if (ok(position, next, test)) {
      position[next] = test;
      queen(position, next + 1);
    }
}


int main(int argc, char *argv[])
{
  int i;
  printf("# of processors = %d\n", omp_get_num_procs());

#pragma omp parallel for private(position)
  for (i = 0; i < N; i++) {
    printf("i = %d thread %d\n", i, omp_get_thread_num());
    position[0] = i;
    queen(position, 1);
  }
}

