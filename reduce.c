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

#define MREDUCE6 0x01
void parse_arguments(int argc, char **argv);
void mreduce6(void);


cl_device_id device;
int kernels;
size_t size;



int main(int argc, char **argv) {
	int i;
	cl_int ret;
	printf(">>>>> BEGIN reduce.c <<<<<\n");


	parse_arguments(argc, argv);
	if(kernels & MREDUCE6) {
		mreduce6();
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

	// parse arguments
	int sel = (argc < 2) ? -1 : atoi(argv[1]);
	if(sel < 0 || sel >= num_devices) {
		printf("Invalid device selection: %d\n", sel);
		exit(-1);
	}
	device = devices[sel];
	kernels = (argc < 3) ? 0 : atoi(argv[2]);
	size = (argc < 4) ? 16 << 20 : 1 << atoi(argv[3]);
	printf("\tDevice selected: %d\n", sel);
	printf("\tKernels to run: %d\n", kernels);
	printf("\tData size: %d\n", size);

}

void mreduce6(void) {
	int i, j;
	cl_int ret;
	printf("Running mreduce6()\n");

	/*
		Build OpenCL kernel
	*/
	cl_context context;
	cl_command_queue queue;
	cl_kernel kernel;
	cl_build_status status;
	char *log;
#ifndef ALTERA
	char *kernel_file = "reduce/mreduce6.cl";
#else
	char *kernel_file = "reduce/mreduce6.aocx";
#endif
	ret = oclKernelSetup(device, kernel_file, "mreduce6", 
		&context, &queue, &kernel, &status, &log);
	CHECK_RC(ret, "oclKernelSetup()")
	CHECK_BS(status, "oclKernelSetup()", log)

	/*
		Initialize Memory
	*/
	size_t local_sz = 128;
	size_t workgroups = size / local_sz / 2 / sizeof(float);
	size_t global_sz = local_sz * workgroups;
	size_t input_sz = size;
	size_t output_sz = workgroups * sizeof(float);
	float *input, *output;
#ifndef ALTERA
	input = malloc(input_sz);
	output = malloc(output_sz);
#else
	posix_memalign ((void **)&input, AOCL_ALIGNMENT, input_sz);
	posix_memalign ((void **)&output, AOCL_ALIGNMENT, output_sz);
#endif
	unsigned int input_len = input_sz / sizeof(float);
	for(i = 0; i < input_len; i++) {
		input[i] = (float)i;
	}
	printf("\tworkgroups:\t%lu\n", workgroups);
	printf("\tlocal_sz:\t%lu\n", local_sz);
	printf("\tglobal_sz:\t%lu\n", global_sz);
	printf("\tinput_sz:\t%lu\n", input_sz);
	printf("\toutput_sz:\t%lu\n", output_sz);
	cl_mem input0_mem = clCreateBuffer(context, CL_MEM_BANK_1_ALTERA | CL_MEM_READ_ONLY, input_sz / 2, NULL, &ret);
	CHECK_RC(ret, "clCreateBuffer(input0_mem)")
	cl_mem input1_mem = clCreateBuffer(context, CL_MEM_BANK_2_ALTERA | CL_MEM_READ_ONLY, input_sz / 2, NULL, &ret);
	CHECK_RC(ret, "clCreateBuffer(input1_mem)")
	cl_mem output_mem = clCreateBuffer(context, CL_MEM_BANK_1_ALTERA | CL_MEM_READ_ONLY, output_sz, NULL, &ret);
	CHECK_RC(ret, "clCreateBuffer(output_mem)")

	
	double total_time = 0.0;
	for(i = 0; i < REPEAT; i++) {
		printf(".");
		fflush(stdout);

		ret = clEnqueueWriteBuffer(queue, input0_mem, CL_TRUE, 0, input_sz / 2, &input[0], 0, NULL, NULL);
		CHECK_RC(ret, "clEnqueueWriteBuffer(input0_mem)")
		ret = clEnqueueWriteBuffer(queue, input1_mem, CL_TRUE, 0, input_sz / 2, &input[input_len/2], 0, NULL, NULL);
		CHECK_RC(ret, "clEnqueueWriteBuffer(input1_mem)")

		ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&input0_mem);
		CHECK_RC(ret, "clSetKernelArg(input0)")
		ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&input1_mem);
		CHECK_RC(ret, "clSetKernelArg(input1)")
		ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&output_mem);
		CHECK_RC(ret, "clSetKernelArg(output)")
		ret = clSetKernelArg(kernel, 3, sizeof(unsigned int), (void *)&input_len);
		CHECK_RC(ret, "clSetKernelArg(input_len)")

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

		float out = 0.0, golden = 0.0;
		ret = clEnqueueReadBuffer(queue, output_mem, CL_TRUE, 0, output_sz, output, 0, NULL, NULL);
		CHECK_RC(ret, "clEnqueueWriteBuffer(output_mem)")
		for(j = 0; j < output_sz / sizeof(float); j++)
			out += output[j];
		for(j = 0; j < input_sz / sizeof(float); j++) 
			golden -= input[j];
		float error = (out-golden)/golden;
		if(error > 0.0001) 
			printf("\t%% error: %f\n", error);


	}
	printf("\n");

	double seconds = total_time / (REPEAT - 1);
	double operations = global_sz * 2;
	printf("Time:\t\t%f s\n", seconds);
	printf("Operations:\t%f G\n", operations / 1024 / 1024/ 1024);
	printf("Bandwidth:\t%f G/s\n", operations/1024/1024/ 1024/seconds);




	free(input);
	free(output);
}
