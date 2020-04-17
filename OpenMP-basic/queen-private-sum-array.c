/* begin */
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#define MAXN 20
int n;	/* a global n */
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
    return 1;
  int sum = 0;
  for (int test = 0; test < n; test++) 
    if (ok(position, next, test)) {
      position[next] = test;
      sum += queen(position, next + 1);
    }
  return sum;
}
/* main */
int main (int argc, char *argv[])
{
  assert(argc == 2);
  n = atoi(argv[1]);
  assert(n <= MAXN);
  /* loop */
  int position[MAXN];
  int num[MAXN] = {0};
#pragma omp parallel for private (position)
  for (int i = 0; i < n; i++) {
    position[0] = i;
    num[i] = queen(position, 1);
    printf("iteration %d # of solution = %d\n", 
	   i, num[i]);
  }
  int numSolution = 0;
  for (int i = 0; i < n; i++)
    numSolution += num[i];
  printf("total # of solutions = %d\n", numSolution);
  return 0;
}
/* end */
