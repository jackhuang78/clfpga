#include <stdio.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <CL/cl.h>
#include <CL/cl_ext.h>
#include "oclutil.h"

#ifndef CL_MEM_BANK_1_ALTERA
#define CL_MEM_BANK_1_ALTERA 0
#define CL_MEM_BANK_2_ALTERA 0
#endif
#define AOCL_ALIGNMENT 64

#define REPEAT 11
#define GLOBAL_READ (0x01)
#define LOCAL_READ (0x02)
#define ADD (0x04)

void parse_arguments(int argc, char **argv);
void run_global_read(void);
void run_local_read(void);
void run_add(void);

int kernels;
int iterations;
cl_device_id device;


int main(int argc, char **argv) {
	int i;
	cl_int ret;
	printf(">>>>> BEGIN benchmark.c <<<<<\n");


	parse_arguments(argc, argv);
	if(kernels & GLOBAL_READ) {
		run_global_read();
	}
	if(kernels & LOCAL_READ) {
		run_local_read();
	}
	if(kernels & ADD) {
		run_add();
	}






	printf(">>>>> END   benchmark.c <<<<<\n");
	exit(0);
}

/**
 *	Get OpenCL devices and display device info.
 *	Parse arguments.
 */
void parse_arguments(int argc, char **argv) {
	int i;
	cl_int ret;

	cl_uint num_devices;
	cl_device_id *devices;
	ret = oclGetDevices(&num_devices, &devices);
	CHECK_RC(ret, "oclGetDevices()")
	printf("Get OpenCL devices and display device info\n");
	printf("\tDevices found: %u\n", num_devices);
	for(i = 0; i < num_devices; i++) {
		printf("\tInfo of device %d:\n", i);
		ret = oclPrintDeviceInfo(devices[i], "\t\t");
		CHECK_RC(ret, "oclPrintDeviceInfo()")
	}
	int sel = (argc < 2) ? -1 : atoi(argv[1]);
	if(sel < 0 || sel >= num_devices) {
		printf("Invalid device selection: %d\n", sel);
		exit(-1);
	}
	device = devices[sel];
	kernels = (argc < 3) ? 7 : atoi(argv[2]);
	iterations = (argc < 4) ? 10 : atoi(argv[3]);
	printf("\tDevice selected: %d\n", sel);
	printf("\tKernels to run: %d\n", kernels);
	//printf("\tIterations: %d\n", iterations);

}

void run_global_read(void) {
	int i;
	cl_int ret;
	printf("Run kernel for global_read\n");

	/*
		Build OpenCL kernel
	*/
	cl_context context;
	cl_command_queue queue;
	cl_kernel kernel;
	cl_build_status status;
	char *log;
	ret = oclKernelSetup(device, "benchmark/global_read.cl", "global_read", 
		&context, &queue, &kernel, &status, &log);
	CHECK_RC(ret, "oclKernelSetup()")
	CHECK_BS(status, "oclKernelSetup()", log)

	/*
		Initialize Memory
	*/
	size_t local_sz = 1024;
	size_t workgroups = 1024;
	size_t global_sz = local_sz * workgroups;
	size_t input_sz = 1 << 30;
	size_t output_sz = global_sz * sizeof(unsigned int);
	unsigned int *input = malloc(input_sz);
	unsigned int *output = malloc(output_sz);
	unsigned int input_elem = input_sz / sizeof(unsigned int);
	for(i = 0; i < input_elem; i++) {
		input[i] = (i + 1) % input_elem;
	}
	printf("\tworkgroups:\t%lu\n", workgroups);
	printf("\tlocal_sz:\t%lu\n", local_sz);
	printf("\tglobal_sz:\t%lu\n", global_sz);
	printf("\tinput_sz:\t%lu\n", input_sz);
	printf("\toutput_sz:\t%lu\n", output_sz);
	cl_mem input_mem = clCreateBuffer(context, CL_MEM_BANK_1_ALTERA | CL_MEM_READ_ONLY, input_sz, NULL, &ret);
	CHECK_RC(ret, "clCreateBuffer(input_mem)")
	cl_mem output_mem = clCreateBuffer(context, CL_MEM_BANK_1_ALTERA | CL_MEM_READ_ONLY, output_sz, NULL, &ret);
	CHECK_RC(ret, "clCreateBuffer(output_mem)")

	unsigned int iter = (kernels == GLOBAL_READ) ? iterations : 10;
	printf("\titerations:\t%d\n", iterations);

	double total_time = 0.0;
	for(i = 0; i < REPEAT; i++) {
		printf(".");
		fflush(stdout);

		ret = clEnqueueWriteBuffer(queue, input_mem, CL_TRUE, 0, input_sz, input, 0, NULL, NULL);
		CHECK_RC(ret, "clEnqueueWriteBuffer(input_mem)")

		ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&input_mem);
		CHECK_RC(ret, "clSetKernelArg(input)")
		ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&output_mem);
		CHECK_RC(ret, "clSetKernelArg(output)")
		ret = clSetKernelArg(kernel, 2, sizeof(unsigned int), (void *)&iter);
		CHECK_RC(ret, "clSetKernelArg(iterations)")

		cl_event event;
		cl_ulong queued, submit, start, end;
		ret = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_sz, &local_sz, 0, NULL, &event);
		CHECK_RC(ret, "clEnqueueNDRangeKernel()")
		clFinish(queue);
		ret = clWaitForEvents(1, &event);
		CHECK_RC(ret, "clWaitForEvents()");
		ret = oclGetProfilingInfo(&event, &queued, &submit, &start, &end);
		CHECK_RC(ret, "oclGetProfilingInfo()")

		if(i != 0) 
			total_time += ((double)(end - start)) * 1e-9;

	}
	printf("\n");

	double seconds = total_time / (REPEAT - 1);
	double bytes = global_sz * iter * 16 * sizeof(unsigned int);
	printf("Time:\t\t%f s\n", seconds);
	printf("Data:\t\t%f MB\n", bytes / 1024 / 1024);
	printf("Bandwidth:\t%f MB/s\n", bytes/1024/1024/seconds);




	free(input);
	free(output);
}

void run_local_read(void) {
	int i;
	cl_int ret;
	printf("Run kernel for local_read\n");

	/*
		Build OpenCL kernel
	*/
	cl_context context;
	cl_command_queue queue;
	cl_kernel kernel;
	cl_build_status status;
	char *log;
	ret = oclKernelSetup(device, "benchmark/local_read.cl", "local_read", 
		&context, &queue, &kernel, &status, &log);
	CHECK_RC(ret, "oclKernelSetup()")
	CHECK_BS(status, "oclKernelSetup()", log)

	/*
		Initialize Memory
	*/
	size_t local_sz = 1024;
	size_t workgroups = 1024;
	size_t global_sz = local_sz * workgroups;
	size_t input_sz = 1 << 14;
	size_t output_sz = global_sz * sizeof(unsigned int);
	unsigned int *input = malloc(input_sz);
	unsigned int *output = malloc(output_sz);
	unsigned int input_elem = input_sz / sizeof(unsigned int);
	for(i = 0; i < input_elem; i++) {
		input[i] = (i + 1) % input_elem;
	}
	printf("\tworkgroups:\t%lu\n", workgroups);
	printf("\tlocal_sz:\t%lu\n", local_sz);
	printf("\tglobal_sz:\t%lu\n", global_sz);
	printf("\tinput_sz:\t%lu\n", input_sz);
	printf("\toutput_sz:\t%lu\n", output_sz);
	cl_mem input_mem = clCreateBuffer(context, CL_MEM_BANK_1_ALTERA | CL_MEM_READ_ONLY, input_sz, NULL, &ret);
	CHECK_RC(ret, "clCreateBuffer(input_mem)")
	cl_mem output_mem = clCreateBuffer(context, CL_MEM_BANK_1_ALTERA | CL_MEM_READ_ONLY, output_sz, NULL, &ret);
	CHECK_RC(ret, "clCreateBuffer(output_mem)")

	unsigned int iter = (kernels == LOCAL_READ) ? iterations : 10;
	printf("\titerations:\t%d\n", iterations);

	double total_time = 0.0;
	for(i = 0; i < REPEAT; i++) {
		printf(".");
		fflush(stdout);

		ret = clEnqueueWriteBuffer(queue, input_mem, CL_TRUE, 0, input_sz, input, 0, NULL, NULL);
		CHECK_RC(ret, "clEnqueueWriteBuffer(input_mem)")

		ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&input_mem);
		CHECK_RC(ret, "clSetKernelArg(input)")
		ret = clSetKernelArg(kernel, 1, input_sz, NULL);
		CHECK_RC(ret, "clSetKernelArg(shared)")
		ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&output_mem);
		CHECK_RC(ret, "clSetKernelArg(output)")
		ret = clSetKernelArg(kernel, 3, sizeof(unsigned int), (void *)&iter);
		CHECK_RC(ret, "clSetKernelArg(iterations)")

		cl_event event;
		cl_ulong queued, submit, start, end;
		ret = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_sz, &local_sz, 0, NULL, &event);
		CHECK_RC(ret, "clEnqueueNDRangeKernel()")
		clFinish(queue);
		ret = clWaitForEvents(1, &event);
		CHECK_RC(ret, "clWaitForEvents()");
		ret = oclGetProfilingInfo(&event, &queued, &submit, &start, &end);
		CHECK_RC(ret, "oclGetProfilingInfo()")

		if(i != 0) 
			total_time += ((double)(end - start)) * 1e-9;

	}
	printf("\n");

	double seconds = total_time / (REPEAT - 1);
	double bytes = global_sz * iter * 16 * sizeof(unsigned int);
	printf("Time:\t\t%f s\n", seconds);
	printf("Data:\t\t%f MB\n", bytes / 1024 / 1024);
	printf("Bandwidth:\t%f MB/s\n", bytes/1024/1024/seconds);




	free(input);
	free(output);
}

void run_add(void) {
	printf("Run kernel for add\n");


}



