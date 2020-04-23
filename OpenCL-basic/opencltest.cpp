#include <iostream>
#include <fstream>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cstring>
#include <string>

#include <CL/cl.h>

int main() {

	int size = 1024 * 1024;

	cl_uint* matrixA = (cl_uint*)malloc(size * sizeof(cl_uint));
	cl_uint* matrixB = (cl_uint*)malloc(size * sizeof(cl_uint));
	cl_uint* matrixC = (cl_uint*)malloc(size * sizeof(cl_uint));

	srand(time(NULL));
	for (int i = 0; i < size; i ++) {
		matrixA[i] = rand() % 10000;
		matrixB[i] = rand() % 10000;
	}

	cl_uint numPlatforms;
	cl_platform_id platform = NULL;
	cl_int status = clGetPlatformIDs(0, NULL, &numPlatforms);
	if (status != CL_SUCCESS || numPlatforms == 0) {
		std::cout << "Error: No Platforms\n";
		return 0;
	}

	cl_platform_id* platforms = (cl_platform_id*)malloc(numPlatforms * sizeof(cl_platform_id));
	status = clGetPlatformIDs(numPlatforms, platforms, NULL);
	if (status != CL_SUCCESS) {
		std::cout << "Error: GetPlatformIds\n";
		return 0;
	}	 
	platform = platforms[0];
	free(platforms);

	std::cout << "numPlatform: " << numPlatforms << std::endl;

	cl_uint numDevices = 0;
	cl_device_id* devices;
	status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, NULL, &numDevices);
	if (status != CL_SUCCESS) {
		std::cout << "Error: GetDeviceIds(num GPUs)\n";
		return 0;
	}	 

	std::cout << "num GPUs: " << numDevices << std::endl;

	if (numDevices == 0) { /* no GPU found */
		status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 0, NULL, &numDevices);
		if (status != CL_SUCCESS) {
			std::cout << "Error: GetDeviceIds(num CPUs)\n";
			return 0;
		}	 
		devices = (cl_device_id*)malloc(numDevices * sizeof(cl_device_id));
		status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, numDevices, devices, NULL);
		if (status != CL_SUCCESS) {
			std::cout << "Error: GetDeviceIds(CPU)\n";
			return 0;
		}	 	
	} else {
		devices = (cl_device_id*)malloc(numDevices * sizeof(cl_device_id));
		status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, numDevices, devices, NULL);
		if (status != CL_SUCCESS) {
			std::cout << "Error: GetDeviceIds(GPU)\n";
			return 0;
		}	 	
	}

	cl_context context = clCreateContext(NULL, 3, devices, NULL, NULL, NULL);

	cl_command_queue commandQueue = clCreateCommandQueue(context, devices[0], 0, NULL);

	std::ifstream stream("kernel.cl");
	if (!stream.is_open()) {
		std::cout << "Error: Kernel File\n";
		return 0;
	}
	std::string kernelInput = std::string(std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>());

	const char* source = kernelInput.c_str();
	size_t sourceLength[] = {strlen(source)};
	cl_program program = clCreateProgramWithSource(context, 1, &source, sourceLength, &status);
	if (status != CL_SUCCESS) {
		std::cout << "Error: CreateProgramWithSource\n";
		return 0;
	}	 	
	status = clBuildProgram(program, 3, devices, NULL, NULL, NULL);
	if (status != CL_SUCCESS) {
		std::cout << "Error: BuildProgram\n";
		return 0;
	}	 	

	cl_kernel kernel = clCreateKernel(program, "add", NULL);

	cl_mem bufferA = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, size * sizeof(cl_uint), matrixA, NULL);
	cl_mem bufferB = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, size * sizeof(cl_uint), matrixB, NULL);
	cl_mem bufferC = clCreateBuffer(context, CL_MEM_WRITE_ONLY, size * sizeof(cl_uint), NULL, NULL);

	clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&bufferA);
	clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&bufferB);
	clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&bufferC);

	size_t globalThreads[] = {(size_t)size};
	size_t localThreads[] = {1};

	status = clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL, globalThreads, localThreads, 0, NULL, NULL);
	if (status != CL_SUCCESS) {
		std::cout << "Error: EnqueueNDRangeKernel\n";
		return 0;
	}	 	

	clEnqueueReadBuffer(commandQueue, bufferC, CL_TRUE, 0, size * sizeof(cl_uint), matrixC, 0, NULL, NULL);

	for (int i = 0; i < size; i ++) {
		if (matrixA[i] + matrixB[i] != matrixC[i]) {
			std::cout << "Fail!!\n";
			break;
		}
	}

	free(devices);
	clReleaseKernel(kernel);
	clReleaseProgram(program);
	clReleaseMemObject(bufferA);
	clReleaseMemObject(bufferB);
	clReleaseMemObject(bufferC);
	clReleaseCommandQueue(commandQueue);
	clReleaseContext(context);

	free(matrixA);
	free(matrixB);
	free(matrixC);
}
