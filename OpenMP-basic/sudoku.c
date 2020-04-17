#include <omp.h>
#include <stdio.h>
/* rowColConflict */
int rowColConflict(int try, int row, int col, int sudoku[9][9])
{
  int conflict = 0;
  for (int i = 0; i < 9 && !conflict; i++)
    if (((col != i) && (sudoku[row][i] == try)) || 
	((row != i) && (sudoku[i][col] == try)))
      conflict = 1;
  return conflict;
}
/* blockConflict */
int blockConflict(int try, int row, int col, int sudoku[9][9])
{
  int blockRow = row / 3;
  int blockCol = col / 3;

  int conflict = 0;
  for (int i = 0; i < 3 && !conflict; i++)
    for (int j = 0; j < 3 && !conflict; j++)
      if (sudoku[3 * blockRow + i][3 * blockCol + j] 
	  == try)
	conflict = 1;
  return conflict;
}
/* conflict */
int conflict(int try, int row, int col, int sudoku[9][9])
{
  return (rowColConflict(try, row, col, sudoku) ||
	  blockConflict(try, row, col, sudoku));
}
/* placeNumber */
int placeNumber(int n, int sudoku[9][9])
{
  if (n == 81)
    return 1;
  int row = n / 9;
  int col = n % 9;
  if (sudoku[row][col] != 0)
    return(placeNumber(n + 1, sudoku));
  /* numSolution */
  int numSolution = 0;
  for (int try = 1; try <= 9; try++) {
    if (!conflict(try, row, col, sudoku)) {
      sudoku[row][col] = try;
      numSolution += placeNumber(n + 1, sudoku);
    }
  } /* for */
  sudoku[row][col] = 0;
  return numSolution;
}
/* main */
int main(void)
{
  int sudoku[9][9];
  int firstZero = -1;
  for (int i = 0; i < 9; i++)
    for (int j = 0; j < 9; j++) {
      scanf("%d", &(sudoku[i][j]));
      if (sudoku[i][j] == 0 && firstZero == -1) 
	firstZero = i * 9 + j;
    }
  /* omp */
#ifdef _OPENMP
  omp_set_num_threads(9);
#endif
  int numSolution = 0;
#pragma omp parallel for reduction(+ : numSolution) \
  firstprivate(sudoku)
  for (int i = 1; i <= 9; i++) {
    if (!conflict(i, firstZero / 9, firstZero % 9, sudoku)) {
      sudoku[firstZero / 9][firstZero % 9] = i;
      numSolution += placeNumber(firstZero, sudoku);
    }
  }
  printf("# of solution = %d\n", numSolution);
  return 0;
}
/* end */
