/* header */
#define CL_USE_DEPRECATED_OPENCL_2_0_APIS 

#include <stdio.h>
#include <assert.h>
#include <CL/cl.h>

#include "matrix.h"
#define MAXGPU 10
#define MAXK 1024
#define MAXLOG 4096

cl_uint A[N][N], B[N][N], C[N][N];
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
  cl_device_id GPU[MAXGPU];
  cl_uint GPU_id_got;
  status = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 
			  MAXGPU, GPU, &GPU_id_got);
  assert(status == CL_SUCCESS);
  printf("There are %d GPU devices\n", GPU_id_got); 
  /* getcontext */
  cl_context context = 
    clCreateContext(NULL, GPU_id_got, GPU, NULL, NULL, 
		    &status);
  assert(status == CL_SUCCESS);
  /* commandqueue */
  cl_command_queue commandQueue =
    clCreateCommandQueue(context, GPU[0], 
			 CL_QUEUE_PROFILING_ENABLE, 
			 &status);
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
    clBuildProgram(program, GPU_id_got, GPU, NULL, NULL, 
		   NULL);

  char log[MAXLOG];
  size_t logLength;
  clGetProgramBuildInfo(program, GPU[0], CL_PROGRAM_BUILD_LOG,
			MAXLOG, log, &logLength);
  puts(log);
  assert(status == CL_SUCCESS);
  printf("Build program completes\n");
  /* createkernel */
  cl_kernel kernel = clCreateKernel(program, "mul", &status);
  assert(status == CL_SUCCESS);
  printf("Build kernel completes\n");
  /* vector */
  for (int i = 0; i < N; i++) 
    for (int j = 0; j < N; j++) {
      A[i][j] = i + j;
      B[i][j] = i - j;
    }
  /* createbuffer */
  cl_mem bufferA = 
    clCreateBuffer(context, 
		   CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		   N * N * sizeof(cl_uint), A, &status);
  assert(status == CL_SUCCESS);
  cl_mem bufferB = 
    clCreateBuffer(context, 
		   CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		   N * N * sizeof(cl_uint), B, &status);
  assert(status == CL_SUCCESS);
  cl_mem bufferC = 
    clCreateBuffer(context, 
		   CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR,
		   N * N * sizeof(cl_uint), C, &status);
  assert(status == CL_SUCCESS);
  printf("Build buffers completes\n");
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
  printf("Set kernel arguments completes\n");
  /* setshape */
  size_t globalThreads[] = {(size_t)N, (size_t)N};
  size_t localThreads[] = {BSIDE, BSIDE};
  cl_event event;
  status = 
    clEnqueueNDRangeKernel(commandQueue, kernel, 2, NULL, 
			     globalThreads, localThreads, 
			     0, NULL, &event);
printf("status = %d\n", status);
  assert(status == CL_SUCCESS);
  /* waitforevent */
  clWaitForEvents(1, &event); 
  printf("Kernel execution completes.\n");
  /* gettime */
  cl_ulong timeEnterQueue, timeSubmit, timeStart, timeEnd;
  status = 
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_QUEUED, 
			    sizeof(cl_ulong), &timeEnterQueue, NULL);
  assert(status == CL_SUCCESS);
  status = 
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_SUBMIT, 
			    sizeof(cl_ulong), &timeSubmit, NULL);
  assert(status == CL_SUCCESS);
  status = 
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, 
			    sizeof(cl_ulong), &timeStart, NULL);
  assert(status == CL_SUCCESS);
  status = 
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, 
			    sizeof(cl_ulong), &timeEnd, NULL); 
  assert(status == CL_SUCCESS);
  /* printtime */
  printf("kernel queued time %f seconds\n", 
	 (timeSubmit - timeEnterQueue) / 1000000000.0);
  printf("kernel submission time %f seconds\n", 
	 (timeStart - timeSubmit) / 1000000000.0);
  printf("kernel execution time %f seconds\n", 
	 (timeEnd - timeStart) / 1000000000.0);
  /* checkandfree */
  for (int i = 0; i < N; i++)
    for (int j = 0; j < N; j++) {
      int sum = 0;
      for (int k = 0; k < N; k++)
	sum += A[i][k] * B[k][j];
      assert(C[i][j] == sum);
    }

  clReleaseContext(context);	/* context etcmake */
  clReleaseCommandQueue(commandQueue);
  clReleaseProgram(program);
  clReleaseKernel(kernel);
  clReleaseMemObject(bufferA);	/* buffers */
  clReleaseMemObject(bufferB);
  clReleaseMemObject(bufferC);
  return 0;
}
/* end */
