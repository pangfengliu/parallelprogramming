/* header */
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define MAXN 4096
#define SIDE (MAXN + 2)

#define nLiveNeighbor(A, i, j) \
  A[i + 1][j] + A[i - 1][j] + A[i][j + 1] + \
  A[i][j - 1] + A[i + 1][j + 1] + A[i + 1][j - 1] + \
  A[i - 1][j + 1] + A[i - 1][j - 1]

char A[SIDE][SIDE];
char B[SIDE][SIDE];
/* print */
void print(char A[SIDE][SIDE], int n)
{
  for (int i = 1; i <= n; i++) {
    for (int j = 1; j <= n; j++) 
      printf("%2d ", A[i][j]);
    printf("\n");
  }
}
/* main */
int main()
{
  int n, generation, cell;
#ifdef READINPUT
  scanf("%d%d", &n, &generation);
  for (int i = 1; i <= n; i++)
    for (int j = 1; j <= n; j++) {
      scanf("%d", &cell);
      A[i][j] = cell;
    }
#else
  n = 4096;
  generation = 20;
  for (int i = 1; i <= n; i++)
    for (int j = 1; j <= n; j++)
      A[i][j] = rand() % 2;
#endif
  /* generation */
  int nln;
  for (int g = 0; g < generation; g++) 
    if (g % 2 == 0)
#pragma omp parallel for       /*  from A to B */
      for (int i = 1; i <= n; i++)
	for (int j = 1; j <= n; j++) {
	  nln = nLiveNeighbor(A, i, j);
	  B[i][j] = ((A[i][j] == 0 && nln == 3) ||
	    (A[i][j] == 1 && (nln == 2 || nln == 3)));
	}
    else
#pragma omp parallel for       /*  from B to A */
      for (int i = 1; i <= n; i++)
	for (int j = 1; j <= n; j++) {
	  nln = nLiveNeighbor(B, i, j);
	  A[i][j] = ((B[i][j] == 0 && nln == 3) ||
	    (B[i][j] == 1 && (nln == 2 || nln == 3)));
	}
  /* printcell */
#ifdef PRINT
  if (generation % 2 == 0)
    print(lifeA, n);
  else
    print(lifeB, n);
#endif
  return 0;
}
/* end */
