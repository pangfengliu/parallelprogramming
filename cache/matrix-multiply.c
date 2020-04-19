#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#define N 2048
#define A 47
#define B 97

int a[N][N], b[N][N], c[N][N];

void initMatrix(int m[N][N], int n)
{
  for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++) 
      m[i][j] = A * i + B * j; 
}

int main(int argc, char *argv[])
{
  assert(argc == 2);
  int n = atoi(argv[1]);
  assert (n <= N);
  initMatrix(a, n);
  initMatrix(b, n);
  for (int i = 0; i < N; i++)
    for (int j = 0; j < N; j++) {
      c[i][j] = 0;
      for (int k = 0; k < N; k++)
	c[i][j] += a[i][k] * b[k][j];
    }
  return 0;
}
