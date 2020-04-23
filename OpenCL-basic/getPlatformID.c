#include <stdio.h>
#include <assert.h>
#include <CL/cl.h>
#define MAXPLATFORM 5
int main(int argc, char *argv[])
{
  printf("Hello, OpenCL\n");
  cl_platform_id platform_id[MAXPLATFORM];
  cl_uint platform_id_got;
  clGetPlatformIDs(MAXPLATFORM, platform_id, 
		   &platform_id_got);
  printf("%d platform found\n", platform_id_got);
  return 0;
}
