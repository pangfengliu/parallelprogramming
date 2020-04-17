#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#define N 100000

int main(int argc, char *argv[])
{
  int uni[2][N];
  int omp[4][N];

  srand(0);
  for (int i = 0; i < N; i++)
    uni[0][i] = rand();

  srand(0);
  for (int i = 0; i < N; i++)
    uni[1][i] = rand();

  for (int i = 0; i < N; i++) 
    if (uni[0][i] != uni[1][i]) {
      fprintf(stderr, "i = %d uni %d != uni %d\n", i, uni[0][i], uni[1][i]);
      exit(-1);
    }

#pragma omp parallel for
  for (unsigned int t = 0; t < 4; t++) {
    srand(t);
    for (int i = 0; i < N; i++)
      omp[t][i] = rand_r(&t);
  }

  for (int i = 0; i < N; i++) 
    if (uni[0][i] != omp[0][i]) {
      fprintf(stderr, "i = %d %d != %d\n", i, uni[0][i], omp[0][i]);
      exit(-1);
    }

  return 0;
}
