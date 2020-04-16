/* declaration */
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#define MAXN 20
int n;				/* a global n */
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
void queen(int position[], int next)
{
  if (next >= n)
    numSolution++;
  else
    for (int test = 0; test < n; test++) 
      if (ok(position, next, test)) {
	position[next] = test;
	queen(position, next + 1);
      }
}
/* go */
void *goQueen(void *pos)
{
  int *position = (int *)pos;
  queen(position, 1);
  pthread_exit(NULL);
}
/* main */
int main (int argc, char *argv[])
{
  assert(argc == 2);
  n = atoi(argv[1]);
  assert(n <= MAXN);
  int *position;
  pthread_t threads[MAXN];
  pthread_attr_t attr;
  pthread_attr_init(&attr);
  pthread_attr_setdetachstate(&attr, 
    PTHREAD_CREATE_JOINABLE);
  for(int i = 0; i < n; i++) {
    position = (int *)calloc(n, sizeof(int));
    assert(position != NULL);
    position[0] = i;
    int error = pthread_create(&threads[i], &attr, goQueen, 
			       (void *)position);
    assert(error == 0);
  }
  pthread_attr_destroy(&attr);
  /* join */
  for (int i = 0; i < n; i++) {
    pthread_join(threads[i], NULL);
#ifdef VERBOSE
    printf("main: thread %d done\n", i);
#endif
  }
  printf("total # of solution %d\n", numSolution);
  pthread_exit(NULL);
  return 0;
}
/* end */
