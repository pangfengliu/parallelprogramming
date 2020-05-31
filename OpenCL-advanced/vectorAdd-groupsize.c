/* header */
#define COPYC
#include <stdio.h>
#include <assert.h>
#include <CL/cl.h>
#define N (65536 * 4)
#define MAXGPU 10
#define MAXK 1024
#define MAXLOG 4096
#define NANO2SECOND 1000000000.0
cl_uint A[N], B[N], C[N];
/* main */
int main(int argc, char *argv[])
{
  printf("Hello, OpenCL\n");
  cl_int status;
  cl_platform_id platform_id;
  cl_uint platform_id_got;
  status = clGetPlatformIDs(1, &platform_id, 
			    &platform_id_got);
  assert(status == CL_SUCCESS && platform_id_got == 1);
  printf("%d platform found\n", platform_id_got);
  /* getdevice */
  cl_device_id GPU[MAXGPU];
  cl_uint GPU_id_got;
  status = 
    clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 
		   MAXGPU, GPU, &GPU_id_got);
  assert(status == CL_SUCCESS && GPU_id_got >= 1);
  printf("There are %d GPU devices\n", GPU_id_got); 
  cl_uint unit;
  status = 
    clGetDeviceInfo(GPU[0], CL_DEVICE_MAX_COMPUTE_UNITS, 
		    sizeof(cl_uint), &unit, NULL);
  assert(status == CL_SUCCESS);
  printf("# of compute units is %d\n", unit);
  /* getcontext */
  cl_context context = 
    clCreateContext(NULL, 1, GPU, NULL, NULL, &status);
  assert(status == CL_SUCCESS);
  /* commandqueue */
  const cl_queue_properties properties[] =
    {CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0};
  cl_command_queue commandQueue = 
    clCreateCommandQueueWithProperties(context, GPU[0],
				       properties, &status);
  assert(status == CL_SUCCESS);
  /* kernelsource */
  FILE *kernelfp = fopen(argv[1], "r");
  assert(kernelfp != NULL);
  char kernelBuffer[MAXK];
  const char *constKernelSource = kernelBuffer;
  size_t kernelLength = 
    fread(kernelBuffer, 1, MAXK, kernelfp);
  printf("The size of kernel source is %zu\n", kernelLength);
  cl_program program =
    clCreateProgramWithSource(context, 1, &constKernelSource, 
			      &kernelLength, &status);
  assert(status == CL_SUCCESS);
  /* buildprogram */
  status = 
    clBuildProgram(program, 1, GPU, NULL, NULL, NULL);
  if (status != CL_SUCCESS) {
    char log[MAXLOG];
    size_t logLength;
    clGetProgramBuildInfo(program, GPU[0], 
			  CL_PROGRAM_BUILD_LOG,
			  MAXLOG, log, &logLength);
    puts(log);
    exit(-1);
  }
  printf("Build program completes\n");
  /* createkernel */
  cl_kernel kernel = clCreateKernel(program, "add", 
				    &status);
  assert(status == CL_SUCCESS);
  printf("Build kernel completes\n");
  /* vectors */
  for (int i = 0; i < N; i++) {
    A[i] = i;
    B[i] = N - i;
  }
  /* createbuffer */
  cl_mem bufferA = 
    clCreateBuffer(context, 
		   CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		   N * sizeof(cl_uint), A, &status);
  assert(status == CL_SUCCESS);
  cl_mem bufferB = 
    clCreateBuffer(context, 
		   CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		   N * sizeof(cl_uint), B, &status);
  assert(status == CL_SUCCESS);
  cl_mem bufferC = 
    clCreateBuffer(context, 
		   CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR,
		   N * sizeof(cl_uint), C, &status);
  assert(status == CL_SUCCESS);
  /* setarg */
  status = clSetKernelArg(kernel, 0, sizeof(cl_mem), 
			  (void*)&bufferA);
  assert(status == CL_SUCCESS);
  status = clSetKernelArg(kernel, 1, sizeof(cl_mem), 
			  (void*)&bufferB);
  assert(status == CL_SUCCESS);
  status = clSetKernelArg(kernel, 2, sizeof(cl_mem), 
			  (void*)&bufferC);
  assert(status == CL_SUCCESS);
  /* setNDRange */
  size_t workItem[] = {(size_t)N};
  FILE *timefp = fopen("vectorAdd-groupsize.dat", "w");
  assert(timefp != NULL);
  for (int groupSize = 1; groupSize <= 256; groupSize *= 2) {
    cl_event event;
    size_t localSize[1];
    localSize[0] = groupSize;
    status = 
      clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL, 
			     workItem, localSize, 
			     0, NULL, &event);
    assert(status == CL_SUCCESS);
    /* waitforevent */
    status = clWaitForEvents(1, &event); 
    assert(status == CL_SUCCESS);
    printf("The kernel with group size %d completes.\n", 
	   groupSize);
    /* getbase */
    cl_ulong base;
    status = 
      clGetEventProfilingInfo(event, 
			      CL_PROFILING_COMMAND_QUEUED, 
			      sizeof(cl_ulong), &base, NULL);
    assert(status == CL_SUCCESS);
    /* gettime */
    cl_ulong timeEnterQueue, timeSubmit, timeStart, 
      timeEnd;
    status = 
      clGetEventProfilingInfo(event, 
			      CL_PROFILING_COMMAND_QUEUED, 
			      sizeof(cl_ulong), &timeEnterQueue, NULL);
    assert(status == CL_SUCCESS);
    status = 
      clGetEventProfilingInfo(event, 
			      CL_PROFILING_COMMAND_SUBMIT, 
			      sizeof(cl_ulong), &timeSubmit, NULL);
    assert(status == CL_SUCCESS);
    /* getrest */
    status = 
      clGetEventProfilingInfo(event, 
			      CL_PROFILING_COMMAND_START, 
			      sizeof(cl_ulong), &timeStart, NULL);
    assert(status == CL_SUCCESS);
    status = 
      clGetEventProfilingInfo(event, 
			      CL_PROFILING_COMMAND_END, 
			      sizeof(cl_ulong), &timeEnd, NULL); 
    assert(status == CL_SUCCESS);
    /* printtime */
    printf("\nkernel entered queue at %f\n", 
	   (timeEnterQueue - base) / NANO2SECOND);
    printf("kernel submitted to device at %f\n", 
	   (timeSubmit - base) / NANO2SECOND);
    printf("kernel started at %f\n", 
	   (timeStart - base) / NANO2SECOND);
    printf("kernel ended  at %f\n", 
	   (timeEnd - base) / NANO2SECOND);
    printf("kernel queued time %f seconds\n", 
	   (timeSubmit - timeEnterQueue) / NANO2SECOND);
    printf("kernel submission time %f seconds\n", 
	   (timeStart - timeSubmit) / NANO2SECOND);
    printf("kernel execution time %f seconds\n", 
	   (timeEnd - timeStart) / NANO2SECOND);
    fprintf(timefp, "%d %f\n", groupSize, 
	    (timeEnd - timeStart) / NANO2SECOND);
  }
  /* checkandfree */
#ifdef COPYC
  clEnqueueReadBuffer(commandQueue, bufferC, CL_TRUE,
		      0, N * sizeof(cl_uint), C, 
		      0, NULL, NULL);
#endif
  for (int i = 0; i < N; i++) 
    assert(C[i] == A[i] + B[i]);

  clReleaseContext(context);	/* context etcmake */
  clReleaseCommandQueue(commandQueue);
  clReleaseProgram(program);
  clReleaseKernel(kernel);
  clReleaseMemObject(bufferA);	/* buffers */
  clReleaseMemObject(bufferB);
  clReleaseMemObject(bufferC);
  fclose(timefp);
  return 0;
}
/* end */
