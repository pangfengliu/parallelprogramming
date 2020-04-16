/* declaration */
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#define NUM_THREADS 8
/* print */
void *printHello(void *thread_id)
{
  int tid = *((int *)thread_id);	
  printf("printHello: tid = %d\n", tid);
  pthread_exit(NULL);
}
/* main */
int main(int argc, char *argv[])
{
   pthread_t threads[NUM_THREADS];
   for(int t = 0; t < NUM_THREADS; t++) {
      printf("main: create thread %d\n", t);
      int rc = 
	pthread_create(&threads[t], NULL, 
		       printHello, (void *)(&t));
      if (rc) {
	printf("main: error code %d\n", rc);
	exit(-1);
      }
   }
   pthread_exit(NULL);
   return 0;
}
/* end */
