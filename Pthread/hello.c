#include <stdio.h>
/* print */
void PrintHello(int id)
{
  printf("printHello: id = %d\n", id);
}
/* main */
#define NUM_ID 8
int main(int argc, char *argv[])
{
  for (int t = 0; t < NUM_ID; t++) {
    printf("main: t = %d\n", t);
    PrintHello(t);
  }
  return 0;
}
/* end */
