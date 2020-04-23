/* header */
#define CL_USE_DEPRECATED_OPENCL_2_0_APIS 

#include <stdio.h>
#include <assert.h>
#include <CL/cl.h>

#define MAXGPU 10
#define MAXK 1024
#define N (1024 * 1024)
#define DEVICENUM 3

/* main */
int main(int argc, char *argv[])
{
  assert(argc == 2);
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
  assert(status == CL_SUCCESS && GPU_id_got >= DEVICENUM);
  printf("There are %d GPU devices\n", GPU_id_got); 
  /* getcontext */
  cl_context context = 
    clCreateContext(NULL, DEVICENUM, GPU, NULL, NULL, 
		    &status);
  assert(status == CL_SUCCESS);
  /* commandqueue */
  cl_command_queue commandQueue =
    clCreateCommandQueue(context, GPU[0], 0, &status);
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
  assert(status == CL_SUCCESS);
  printf("Build program completes\n");
  /* createkernel */
  cl_kernel kernel = clCreateKernel(program, "add", &status);
  assert(status == CL_SUCCESS);
  printf("Build kernel completes\n");
  /* vectors */
  cl_uint* A = (cl_uint*)malloc(N * sizeof(cl_uint));
  cl_uint* B = (cl_uint*)malloc(N * sizeof(cl_uint));
  cl_uint* C = (cl_uint*)malloc(N * sizeof(cl_uint));
  cl_uint* D = (cl_uint*)malloc(N * sizeof(cl_uint));
  cl_uint* E = (cl_uint*)malloc(N * sizeof(cl_uint));
  cl_uint* F = (cl_uint*)malloc(N * sizeof(cl_uint));
  cl_uint* G = (cl_uint*)malloc(N * sizeof(cl_uint));

  assert(A != NULL && B != NULL && C != NULL);

  for (int i = 0; i < N; i++) {
    A[i] = i;
    B[i] = N - i;
    D[i] = i;
    E[i] = N - i;
  }

  /* createbuffer1 */
  cl_mem bufferAin = 
    clCreateBuffer(context, 
		   CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		   N * sizeof(cl_uint), A, &status);
  assert(status == CL_SUCCESS);
  cl_mem bufferBin = 
    clCreateBuffer(context, 
		   CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		   N * sizeof(cl_uint), B, &status);
  assert(status == CL_SUCCESS);
  cl_mem bufferCout = 
    clCreateBuffer(context, 
		   CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR,
		   N * sizeof(cl_uint), C, &status);
  assert(status == CL_SUCCESS);

  /* createbuffer2 */
  cl_mem bufferDin = 
    clCreateBuffer(context, 
		   CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		   N * sizeof(cl_uint), D, &status);
  assert(status == CL_SUCCESS);
  cl_mem bufferEin = 
    clCreateBuffer(context, 
		   CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		   N * sizeof(cl_uint), E, &status);
  assert(status == CL_SUCCESS);
  cl_mem bufferFout = 
    clCreateBuffer(context, 
		   CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR,
		   N * sizeof(cl_uint), F, &status);
  assert(status == CL_SUCCESS);

  /* createbuffer3 */
  cl_mem bufferCin = 
    clCreateBuffer(context, 
		   CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		   N * sizeof(cl_uint), C, &status);
  assert(status == CL_SUCCESS);
  cl_mem bufferFin = 
    clCreateBuffer(context, 
		   CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		   N * sizeof(cl_uint), F, &status);
  assert(status == CL_SUCCESS);
  cl_mem bufferGout = 
    clCreateBuffer(context, 
		   CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR,
		   N * sizeof(cl_uint), G, &status);
  assert(status == CL_SUCCESS);
  printf("Build buffers completes\n");

  size_t globalThreads[] = {(size_t)N};
  size_t localThreads[] = {1};

  /* ABC */
  cl_event events[3];
  status = clSetKernelArg(kernel, 0, sizeof(cl_mem), 
			  (void*)&bufferAin);
  assert(status == CL_SUCCESS);
  status = clSetKernelArg(kernel, 1, sizeof(cl_mem), 
			  (void*)&bufferBin);
  assert(status == CL_SUCCESS);
  status = clSetKernelArg(kernel, 2, sizeof(cl_mem), 
			  (void*)&bufferCout);
  assert(status == CL_SUCCESS);
  printf("Set kernel arguments completes\n");
  status = 
    clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL, 
			   globalThreads, localThreads, 
			   0, NULL, &(events[0]));
  assert(status == CL_SUCCESS);

  status = clSetKernelArg(kernel, 0, sizeof(cl_mem), 
			  (void*)&bufferDin);
  assert(status == CL_SUCCESS);
  status = clSetKernelArg(kernel, 1, sizeof(cl_mem), 
			  (void*)&bufferEin);
  assert(status == CL_SUCCESS);
  status = clSetKernelArg(kernel, 2, sizeof(cl_mem), 
			  (void*)&bufferFout);
  assert(status == CL_SUCCESS);
  printf("Set kernel arguments completes\n");
  status = 
    clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL, 
			   globalThreads, localThreads, 
			   0, NULL, &(events[1]));
  assert(status == CL_SUCCESS);

  status = clSetKernelArg(kernel, 0, sizeof(cl_mem), 
			  (void*)&bufferCin);
  assert(status == CL_SUCCESS);
  status = clSetKernelArg(kernel, 1, sizeof(cl_mem), 
			  (void*)&bufferFin);
  assert(status == CL_SUCCESS);
  status = clSetKernelArg(kernel, 2, sizeof(cl_mem), 
			  (void*)&bufferGout);
  assert(status == CL_SUCCESS);
  printf("Set kernel arguments completes\n");
  status = 
    clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL, 
			   globalThreads, localThreads, 
			   2, events, &(events[2])); /* wait for the first two events */
  assert(status == CL_SUCCESS);

  /* checkandfree */
  for (int i = 0; i < N; i++)
    assert(A[i] + B[i] == C[i]);

  free(A);			/* host memory */
  free(B);
  free(C);
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
