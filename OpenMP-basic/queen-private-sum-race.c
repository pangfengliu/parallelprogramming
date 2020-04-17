/* begin */
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#define MAXN 20
int n;	/* a global n */
int numSolution;
/* ok */
int ok(int position[], int next, int test)
{
  for (int i = 0; i < next; i++)
    if (position[i] == test || 
	(abs(test - position[i]) == next - i))
      return 0;
  return 1;
}
/* queen */
int queen(int position[], int next)
{
  if (next >= n)
    numSolution++;
  for (int test = 0; test < n; test++) 
    if (ok(position, next, test)) {
      position[next] = test;
      queen(position, next + 1);
    }
}
/* main */
int main (int argc, char *argv[])
{
  assert(argc == 2);
  n = atoi(argv[1]);
  assert(n <= MAXN);
  /* loop */
  int position[MAXN];
#pragma omp parallel for private (position)
  for (int i = 0; i < n; i++) {
    position[0] = i;
    queen(position, 1);
  }
  printf("total # of solutions = %d\n", 
	 numSolution);
  return 0;
}
/* end */
