/* header */
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <assert.h>
#include "omp.h"

#define N 400000000

char notPrime[N];
int nPrime = 0;
char primes[N];
/* main */
int main(int argc, char *argv[])
{
  assert(argc == 2);
  int n = atoi(argv[1]);
  int bound = round(sqrt(n));

  for (int i = 2; i <= bound; i++) 
    if (!notPrime[i]) {
#pragma omp parallel for 
      for (int j = 2 * i; j < n; j += i)
	notPrime[j] = 1;
    }
  /* count */
  int nPrime = 0;
#pragma omp parallel for reduction(+ : nPrime)
  for (int i = 2; i < n; i++) 
    if (notPrime[i] == 0)
      nPrime++;

  printf("number of prime is %d\n", nPrime);
  return 0;
}
/* end */
