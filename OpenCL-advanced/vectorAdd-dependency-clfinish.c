/* header */
#define CL_USE_DEPRECATED_OPENCL_2_0_APIS 
#include <stdio.h>
#include <assert.h>
#include <CL/cl.h>
#define N (65536)
#define MAXGPU 10
#define MAXK 1024
#define MAXLOG 4096
#define DEVICENUM 2
#define NANO2SECOND 1000000000.0

// #define COPYG

cl_uint A[N], B[N], C[N], D[N], E[N], F[N], G[N];
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
  assert(status == CL_SUCCESS && 
	 GPU_id_got >= DEVICENUM);
  printf("There are %d GPU devices\n", GPU_id_got); 
  /* getcontext */
  cl_context context = 
    clCreateContext(NULL, DEVICENUM, GPU, NULL, NULL, 
		    &status);
  assert(status == CL_SUCCESS);
  /* commandqueue */
  const cl_queue_properties properties[] =
    {CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0};
  cl_command_queue commandQueue[DEVICENUM];
  for (int device = 0; device < DEVICENUM; device++) {
    commandQueue[device] = 
      clCreateCommandQueueWithProperties(context, GPU[device],
			   properties, &status);
    assert(status == CL_SUCCESS);
  }
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
    clBuildProgram(program, DEVICENUM, GPU, NULL, 
		   NULL, NULL);
  if (status != CL_SUCCESS) {
    char log[MAXLOG];
    size_t logLength;
    for (int device = 0; device < DEVICENUM; device++) {
      clGetProgramBuildInfo(program, GPU[device], 
			    CL_PROGRAM_BUILD_LOG,
			    MAXLOG, log, &logLength);
      puts(log);
    }
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
    D[i] = i;
    E[i] = N - i;
  }
  /* createbuffer1 */
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
		   CL_MEM_READ_WRITE,
		   N * sizeof(cl_uint), NULL, &status);
  assert(status == CL_SUCCESS);
  /* createbuffer2 */
  cl_mem bufferD = 
    clCreateBuffer(context, 
		   CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		   N * sizeof(cl_uint), D, &status);
  assert(status == CL_SUCCESS);
  cl_mem bufferE = 
    clCreateBuffer(context, 
		   CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		   N * sizeof(cl_uint), E, &status);
  assert(status == CL_SUCCESS);
  cl_mem bufferF = 
    clCreateBuffer(context, 
		   CL_MEM_READ_WRITE,
		   N * sizeof(cl_uint), NULL, &status);
  assert(status == CL_SUCCESS);
  /* createbuffer3 */
  cl_mem bufferG = 
    clCreateBuffer(context, 
		   CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR,
		   N * sizeof(cl_uint), G, &status);
  assert(status == CL_SUCCESS);
  printf("Build buffers completes\n");
  /* shape */
  size_t globalThreads[] = {(size_t)N};
  size_t localThreads[] = {256};
  /* ABC */
#define EVENT 3
  cl_event events[EVENT];
  status = clSetKernelArg(kernel, 0, sizeof(cl_mem), 
			  (void*)&bufferA);
  assert(status == CL_SUCCESS);
  status = clSetKernelArg(kernel, 1, sizeof(cl_mem), 
			  (void*)&bufferB);
  assert(status == CL_SUCCESS);
  status = clSetKernelArg(kernel, 2, sizeof(cl_mem), 
			  (void*)&bufferC);
  assert(status == CL_SUCCESS);

  status = 
    clEnqueueNDRangeKernel(commandQueue[0], kernel, 1, NULL, 
			   globalThreads, localThreads, 
			   0, NULL, &(events[0]));
  assert(status == CL_SUCCESS);
  /* DEF */
  status = clSetKernelArg(kernel, 0, sizeof(cl_mem), 
			  (void*)&bufferD);
  assert(status == CL_SUCCESS);
  status = clSetKernelArg(kernel, 1, sizeof(cl_mem), 
			  (void*)&bufferE);
  assert(status == CL_SUCCESS);
  status = clSetKernelArg(kernel, 2, sizeof(cl_mem), 
			  (void*)&bufferF);
  assert(status == CL_SUCCESS);
  status = 
    clEnqueueNDRangeKernel(commandQueue[1], kernel, 1, NULL, 
			   globalThreads, localThreads, 
			   0, NULL, &(events[1]));
  assert(status == CL_SUCCESS);
  /* CFG */
  clFinish(commandQueue[0]);
  clFinish(commandQueue[1]);
  status = clSetKernelArg(kernel, 0, sizeof(cl_mem), 
			  (void*)&bufferC);
  assert(status == CL_SUCCESS);
  status = clSetKernelArg(kernel, 1, sizeof(cl_mem), 
			  (void*)&bufferF);
  assert(status == CL_SUCCESS);
  status = clSetKernelArg(kernel, 2, sizeof(cl_mem), 
			  (void*)&bufferG);
  assert(status == CL_SUCCESS);
  status = 
    clEnqueueNDRangeKernel(commandQueue[0], kernel, 1, NULL, 
			   globalThreads, localThreads, 
			   0, NULL, &(events[2])); 
  assert(status == CL_SUCCESS);
  /* waitforevent */
  status = clWaitForEvents(1, &(events[2])); 
  assert(status == CL_SUCCESS);
  printf("All kernels complete.\n");
  /* getbase */
  cl_ulong base;
  status = 
    clGetEventProfilingInfo(events[0], 
      CL_PROFILING_COMMAND_QUEUED, 
      sizeof(cl_ulong), &base, NULL);
  assert(status == CL_SUCCESS);
  /* gettime */
  for (int event = 0; event < EVENT; event++) {
    cl_ulong timeEnterQueue, timeSubmit, timeStart, 
      timeEnd;
    status = 
      clGetEventProfilingInfo(events[event], 
        CL_PROFILING_COMMAND_QUEUED, 
        sizeof(cl_ulong), &timeEnterQueue, NULL);
    assert(status == CL_SUCCESS);
    status = 
      clGetEventProfilingInfo(events[event], 
        CL_PROFILING_COMMAND_SUBMIT, 
        sizeof(cl_ulong), &timeSubmit, NULL);
    assert(status == CL_SUCCESS);
    /* getrest */
    status = 
      clGetEventProfilingInfo(events[event], 
        CL_PROFILING_COMMAND_START, 
        sizeof(cl_ulong), &timeStart, NULL);
    assert(status == CL_SUCCESS);
    status = 
      clGetEventProfilingInfo(events[event], 
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
  }  
  /* checkandfree */
#ifdef COPYG
  clEnqueueReadBuffer(commandQueue[0], bufferG, CL_TRUE,
		      0,  N * sizeof(cl_uint), G, 
		      0, NULL, NULL);
#endif
  for (int i = 0; i < N; i++) 
    assert(G[i] == A[i] + B[i] + D[i] + E[i]);

  clReleaseContext(context);	/* context etcmake */
  for (int device = 0; device < DEVICENUM; device++)
    clReleaseCommandQueue(commandQueue[device]);
  clReleaseProgram(program);
  clReleaseKernel(kernel);
  clReleaseMemObject(bufferA);	/* buffers */
  clReleaseMemObject(bufferB);
  clReleaseMemObject(bufferC);
  clReleaseMemObject(bufferD);
  clReleaseMemObject(bufferE);
  clReleaseMemObject(bufferF);
  clReleaseMemObject(bufferG);
  return 0;
}
/* end */
