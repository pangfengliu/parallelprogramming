#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <limits.h>

#define MAXM 5000

int count[MAXM][MAXM];

void initRowCol(int row[], int col[], int n)
{
  for (int r = 0; r < n; r++)
    row[r] = rand() % 100;

  col[1] = rand() % 100;
  for (int c = 1; c < n; c++)
    col[c] = row[c - 1];
}

int findMinOp(int row[], int col[], int count[MAXM][MAXM], int n)
{
  for (int i = 0; i < n - 1; i++)
    count[i][i + 1] = row[i] * col[i] * col[i + 1];

  for (int d = 3; d <= n; d++)
    for (int i = 0; i + d <= n; i++) {
      count[i][i + d - 1] = INT_MAX;
      for (int s = i + 1; s < i + d; s++) { /* separation */
	int cost = count[i][s] + count[s + 1][i + d - 1] + row[i] * col[s] * col[i + d - 1];
	if (cost < count[i][i + d - 1])
	  count[i][i + d - 1] = cost;
      }
#ifdef DEBUG
      printf("count[%d][%d] = %d\n", i, i + d - 1, count[i][i + d - 1]);
#endif
    }
  return(count[0][n - 1]);
}
  
int main(int argc, char *argv[])
{
  assert(argc == 2);
  int n = atoi(argv[1]);
  assert (n <= MAXM);

  int row[MAXM], col[MAXM];
  initRowCol(row, col, n);
  printf("%d\n", findMinOp(row, col, count, n));
  return 0;
}
