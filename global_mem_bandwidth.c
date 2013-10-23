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

#define T float
#define LSZ 256
#define GSZ (1024 * 256 *  LSZ)
#define EXTRA 31
#define OFFSET 4096

typedef enum {
	read_linear,
	read_linear_uncached,
	read_single,
	write_linear,
	read_random,
	read_random1,
	read_uncoalescing
} kernel_t;
kernel_t to_kernel_id(char *kernel_name);
double run_kernel(cl_device_id device, char *kernel_file, char *kernel_name, 
				T *input, T *output, T *golden, size_t input_sz, size_t output_sz,
				int iter, int verify);
int verify_kernel(T *input, T *output, T *c0, T *golden, size_t input_sz, size_t output_sz, kernel_t kid);

int main(int argc, char **argv) {
	printf("======== BEGIN global_mem_bandwidth.c ========\n");
	int i, j;
	cl_int ret, ret0, ret1, ret2, ret3;


	/**
	 *	Get OpenCL devices and display device info.
	 */
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
	int iter = (argc < 3) ? 10 : atoi(argv[2]);
	int verify = (argc < 4) ? 0 : atoi(argv[3]);
	if(sel < 0 || sel >= num_devices) {
		printf("Invalid device selection: %d", sel);
		exit(-1);
	}
	printf("\tDevice selected: %d\n", sel);
	printf("\tIterations: %d\n", iter);
	printf("\tVerification: %s\n", verify ? "on" : "off");
	
	/**
	 *	Initialize host memory buffers
	 */
	size_t input_sz = (GSZ + EXTRA) * sizeof(T);
	size_t output_sz = (GSZ + EXTRA) * sizeof(T);
	T *input, *output, *golden;
#ifdef ALTERA		
	posix_memalign ((void **)&input, AOCL_ALIGNMENT, input_sz);
	posix_memalign ((void **)&output, AOCL_ALIGNMENT, output_sz);	
#else
	input = (T *)malloc(input_sz);
	output = (T *)malloc(output_sz);
#endif
	if(verify)
		golden = (T *)malloc(output_sz);
	printf("Initialize host memory buffers\n");
	printf("\tGlobal Size: %d K\n", GSZ >> 10);
	printf("\tLocal Size: %d\n", LSZ);
	printf("\tNumber of Workgroup: %d\n", GSZ / LSZ);
	printf("\tInput Memory Buffer: %lu MB\n", input_sz >> 20);
	printf("\tOutput Memory Buffer: %lu MB\n", output_sz >> 20);
	
	
	
	//==================================================================
	printf("Run kernels\n");
	double bw;

#ifndef ALTERA
	bw = run_kernel(devices[sel], "global_mem_bandwidth/read_linear.cl", "read_linear", 
					input, output, golden, input_sz, output_sz, iter, verify);
	printf("\tread_linear:\t%7.3f GB/s\n", bw/1024/1024/1024);
	bw = run_kernel(devices[sel], "global_mem_bandwidth/read_single.cl", "read_single", 
					input, output, golden, input_sz, output_sz, iter, verify);
	printf("\tread_single:\t%7.3f GB/s\n", bw/1024/1024/1024);
	bw = run_kernel(devices[sel], "global_mem_bandwidth/write_linear.cl", "write_linear", 
					input, output, golden, input_sz, output_sz, iter, verify);
	printf("\twrite_linear:\t%7.3f GB/s\n", bw/1024/1024/1024);
#else
	bw = run_kernel(devices[sel], "global_mem_bandwidth/read_linear.aocx", "read_linear", 
					input, output, golden, input_sz, output_sz, iter, verify);
	printf("\tread_linear:\t%7.3f GB/s\n", bw/1024/1024/1024);
	bw = run_kernel(devices[sel], "global_mem_bandwidth/read_single.aocx", "read_single", 
					input, output, golden, input_sz, output_sz, iter, verify);
	printf("\tread_single:\t%7.3f GB/s\n", bw/1024/1024/1024);
	bw = run_kernel(devices[sel], "global_mem_bandwidth/write_linear.aocx", "write_linear", 
					input, output, golden, input_sz, output_sz, iter, verify);
	printf("\twrite_linear:\t%7.3f GB/s\n", bw/1024/1024/1024);
#endif	
	
	free(input);
	free(output);
}

double run_kernel(cl_device_id device, char *kernel_file, char *kernel_name, 
				T *input, T *output, T *golden, size_t input_sz, size_t output_sz,
				int iter, int verify) {
	int i, j;
	cl_int ret;
	kernel_t kid = to_kernel_id(kernel_name);

	//printf("Run kernel for %s\n", kernel_name);


	/**
	 *	Build OpenCL kernel
	 */
	cl_context context;
	cl_command_queue queue;
	cl_kernel kernel;
	cl_build_status status;
	char *log;
	//printf("\tKernel file: %s\n", kernel_file);
	//printf("\tKernel name: %s\n", kernel_name);
	ret = oclKernelSetup(device, kernel_file, kernel_name, 
		&context, &queue, &kernel, &status, &log);
	CHECK_RC(ret, "oclKernelSetup()")
	CHECK_BS(status, "oclKernelSetup()", log)


	/**
	 *	Initialize device memory buffers
	 */
	cl_mem input_mem, output_mem, const_mem;

	switch(kid) {
	case read_linear:
	case read_single:
		input_mem = clCreateBuffer(context, CL_MEM_BANK_1_ALTERA | CL_MEM_READ_ONLY, input_sz, NULL, &ret);
		CHECK_RC(ret, "clCreateBuffer(input_mem)")
		output_mem = clCreateBuffer(context, CL_MEM_BANK_1_ALTERA | CL_MEM_WRITE_ONLY, output_sz, NULL, &ret);
		CHECK_RC(ret, "clCreateBuffer(output_mem)")
		break;
	case write_linear:
		const_mem = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(T), NULL, &ret);
		CHECK_RC(ret, "clCreateBuffer(const_mem)")
		output_mem = clCreateBuffer(context, CL_MEM_BANK_1_ALTERA | CL_MEM_WRITE_ONLY, output_sz, NULL, &ret);
		CHECK_RC(ret, "clCreateBuffer(output_mem)")
		break;


	}



	/**
	 *	Loop to run kernel
	 */	
	double seconds = 0;
	int error = 0;
	srand(time(0));
	for(i = 0; i < iter; i++) {
		printf(".");
		fflush(stdout);
		
		/**
		 *	Initialize input buffer
		 */
		for(j = 0; j < GSZ + EXTRA; j++)
			input[j] = (T) rand();
		T c0[1];
		c0[0] = 123;
		
		//if(verify) {
		switch(kid) {
		case read_linear:
		case read_single:
			ret = clEnqueueWriteBuffer(queue, input_mem, CL_TRUE, 0, input_sz, input, 0, NULL, NULL);
			CHECK_RC(ret, "clEnqueueWriteBuffer(input_mem)")
			break;
		
		case write_linear:
			ret = clEnqueueWriteBuffer(queue, const_mem, CL_TRUE, 0, sizeof(T), c0, 0, NULL, NULL);
			CHECK_RC(ret, "clEnqueueWriteBuffer(const_mem)")				
			break;
		}
		//clFinish(queue);
		//}

		//printf("finsh enqueue write\n");

		/**
		 *	Set kernel arguments
		 */
		switch(kid) {
		case read_linear:	
		case read_single:	
			ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&input_mem);
			CHECK_RC(ret, "clSetKernelArg(input)")
			ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&output_mem);
			CHECK_RC(ret, "clSetKernelArg(output)")
			break;

		case write_linear:
			ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&const_mem);
			CHECK_RC(ret, "clSetKernelArg(c0)")			
			ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&output_mem);
			CHECK_RC(ret, "clSetKernelArg(output)")
			break;
		}
		//clFinish(queue);


		//sleep(1);
			
		/**
		 *	Run kernel and get profiling info
		 */
		cl_event event;
		size_t gsz = GSZ, lsz = LSZ;
		cl_ulong queued, submit, start, end;

		ret = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &gsz, &lsz, 0, NULL, &event);
		CHECK_RC(ret, "clEnqueueNDRangeKernel()")
		clFinish(queue);
		ret = clWaitForEvents(1, &event);
		CHECK_RC(ret, "clWaitForEvents()");
		ret = oclGetProfilingInfo(&event, &queued, &submit, &start, &end);
		CHECK_RC(ret, "oclGetProfilingInfo()")

		if(i != 0)
			seconds += ((double)(end - start)) * 1e-9;

		/**
		 *	Read output buffer, and verify result if specified
		 */
		//printf("enqueueReadBuffer\n");
		//if(verify) {
		ret = clEnqueueReadBuffer(queue, output_mem, CL_TRUE, 0, output_sz, output, 0, NULL, NULL);
		CHECK_RC(ret, "clEnqueueReadBuffer()")
		//clFinish(queue);
		//}
		if(verify && !verify_kernel(input, output, c0, golden, input_sz, output_sz, kid)) {
			printf("Verification failed\n");
			exit(-1);
		}
			

				
	}
	//return total_time / (iter - 1);
	//printf("Average time: %lu\n", total_time / (iter-1));

	/**
	 * Release Resource
	 */
	/*
	ret = clReleaseMemObject(input_mem);
	CHECK_RC(ret, "clReleaseMemObject(input_mem)")
	ret = clReleaseMemObject(output_mem);
	CHECK_RC(ret, "clReleaseMemObject(output_mem)")
	ret = clReleaseMemObject(const_mem);
	CHECK_RC(ret, "clReleaseMemObject(const_mem)")*/
	


	/**
	 *	Calculate bandwidth
	 */
	double bytes = sizeof(T) * (EXTRA + 1) * GSZ * (iter - 1);
	//printf("seconds: %f, bytes: %f, bw: %f\n", seconds, bytes, bytes/seconds);
	return bytes/seconds;
	


}

int verify_kernel(T *input, T *output, T *c0, T *golden, size_t input_sz, size_t output_sz, kernel_t kid) {
	int i, j;
	for(i = 0; i < GSZ; i++) {
		golden[i] = 0;
	}

	switch(kid) {
	case read_linear:
		for(i = 0; i < GSZ; i++)
			for(j = 0; j <= EXTRA; j++)
				golden[i] += input[i+j];
		break;

	case read_single:
		for(i = 0; i < GSZ; i++)
			for(j = 0; j <= EXTRA; j++)
				golden[i] += input[j];
		break;

	case write_linear:
		for(i = 0; i < GSZ; i++)
			golden[i] = *c0;
		break;

	}

	int error = 0;
	for(i = 0; i < GSZ; i++) {
		
		if(error > 10) 
			return error == 0;

		if(output[i] != golden[i]) {
			printf("%d: %f != %f\n", i, output[i], golden[i]);
			error += 1;
		}
	}

	return error == 0;
}

kernel_t to_kernel_id(char *kernel_name) {

	if(strcmp(kernel_name, "read_linear") == 0)
		return read_linear;
	else if(strcmp(kernel_name, "read_linear_uncached") == 0)
		return read_linear_uncached;
	else if(strcmp(kernel_name, "read_single") == 0)
		return read_single;
	else if(strcmp(kernel_name, "write_linear") == 0)
		return write_linear;
	else if(strcmp(kernel_name, "read_random") == 0)
		return read_random;
	else if(strcmp(kernel_name, "read_random1") == 0)
		return read_random1;
	else if(strcmp(kernel_name, "read_uncoalescing") == 0)
		return read_uncoalescing;
	else {
		printf("Unknown kernel name: %s\n", kernel_name);
		exit(-1);
	}

}
