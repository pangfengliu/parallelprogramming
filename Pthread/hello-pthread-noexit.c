/* declaration */
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#define NUM_THREADS 8
/* print */
void *printHello(void *thread_id)
{
   long tid = (long)thread_id;
   printf("printHello: tid = %ld\n", tid);
   pthread_exit(NULL);
}
/* main */
int main (int argc, char *argv[])
{
   pthread_t threads[NUM_THREADS];
   int rc;
   long t;
   for(t = 0; t < NUM_THREADS; t++) {
      printf("main: create thread %ld\n", t);
      rc = pthread_create(&threads[t], NULL, 
			  printHello, (void *)t);
      if (rc) {
	printf("main: error code %d\n", rc);
	exit(-1);
      }
   }
   return 0;
}
/* end */
