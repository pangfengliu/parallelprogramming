#include <stdio.h>

#define N 2048
#define A 47
#define B 97

int a[N][N], b[N][N], c[N][N];

void initMatrix(int m[N][N])
{
  for (int i = 0; i < N; i++)
    for (int j = 0; j < N; j++) 
      m[i][j] = A * i + B * j; 
}

int main()
{
  initMatrix(a);
  initMatrix(b);
  for (int j = 0; j < N; j++) 
    for (int i = 0; i < N; i++) {
      c[i][j] = 0;
      for (int k = 0; k < N; k++)
	c[i][j] += a[i][k] * b[k][j];
    }
  return 0;
}
