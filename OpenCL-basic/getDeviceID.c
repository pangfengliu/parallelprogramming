#include <stdio.h>
#include <assert.h>
#include <CL/cl.h>
/* header */
#define MAXB 256
#define MAXPLATFORM 5
#define MAXDEVICE 10
int main(int argc, char *argv[])
{
  printf("Hello, OpenCL\n");
  cl_platform_id platform_id[MAXPLATFORM];
  cl_device_id device_id[MAXDEVICE];
  cl_uint platform_id_got;
  clGetPlatformIDs(MAXPLATFORM, platform_id, 
		   &platform_id_got);
  printf("%d platform found\n", platform_id_got);
  /* getinfo */
  for (int i = 0; i < platform_id_got; i++) {
    char buffer[MAXB];
    size_t length;
    clGetPlatformInfo(platform_id[i], CL_PLATFORM_NAME, 
		      MAXB, buffer, &length);
    buffer[length] = '\0';
    printf("Platform name %s\n", buffer);
    clGetPlatformInfo(platform_id[i], CL_PLATFORM_VENDOR, 
		      MAXB, buffer, &length);
    buffer[length] = '\0';
    printf("Platform vendor %s\n", buffer);
    clGetPlatformInfo(platform_id[i], CL_PLATFORM_VERSION, 
		      MAXB, buffer, &length);
    buffer[length] = '\0';
    printf("OpenCL version %s\n", buffer);
    clGetPlatformInfo(platform_id[i], CL_PLATFORM_PROFILE, 
		      MAXB, buffer, &length);
    buffer[length] = '\0';
    printf("Platform profile %s\n", buffer);
    /* getDeviceID */
    cl_device_id devices[MAXDEVICE];
    cl_uint device_id_got;
    clGetDeviceIDs(platform_id[i], CL_DEVICE_TYPE_ALL, 
      MAXDEVICE, devices, &device_id_got);
    printf("There are %d devices\n", device_id_got); 
    clGetDeviceIDs(platform_id[i], CL_DEVICE_TYPE_CPU, 
      MAXDEVICE, devices, &device_id_got);
    printf("There are %d CPU devices\n", device_id_got); 
    clGetDeviceIDs(platform_id[i], CL_DEVICE_TYPE_GPU, 
      MAXDEVICE, devices, &device_id_got);
    printf("There are %d GPU devices\n", device_id_got); 
  }
  return 0;
}
/* end */
