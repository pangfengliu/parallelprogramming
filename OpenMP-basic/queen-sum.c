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
/* go */
int goQueen(int i)
{
  int position[MAXN];
  position[0] = i;
  int num = queen(position, 1);
  printf("goQueen: i = %d, # of solution = %d\n", 
	 position[0], num);
  return num;
}
/* main */
int main (int argc, char *argv[])
{
  assert(argc == 2);
  n = atoi(argv[1]);
  assert(n <= MAXN);
  int numSolution = 0;
#pragma omp parallel for
  for (int i = 0; i < n; i++)
    numSolution += goQueen(i);
  printf("total # of solution = %d\n", numSolution);
  return 0;
}
/* end */
