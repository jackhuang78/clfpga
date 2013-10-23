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
#define VECT 2
#define LSZ (1024)
#define GSZ (1024*LSZ)
#define EXTRA 255


double run_kernel(cl_device_id device, char *kernel_file, char *kernel_name);

size_t local_sz, output_sz;
T *output;
int iter;


int main(int argc, char **argv) {
	printf("======== BEGIN local_mem_bandwidth.c ========\n");
	int i, j;
	cl_int ret;


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
	iter = (argc < 3) ? 10 : atoi(argv[2]);
	//int verify = (argc < 4) ? 0 : atoi(argv[3]);
	if(sel < 0 || sel >= num_devices) {
		printf("Invalid device selection: %d", sel);
		exit(-1);
	}
	printf("\tDevice selected: %d\n", sel);
	printf("\tIterations: %d\n", iter);
	//printf("\tVerification: %s\n", verify ? "on" : "off");


	/**
	 *	Initialize host memory buffers
	 */
	output_sz = (GSZ) * sizeof(T) * VECT;
	local_sz = (LSZ + EXTRA) * sizeof(T) * VECT;
#ifdef ALTERA		
	posix_memalign ((void **)&output, AOCL_ALIGNMENT, output_sz);	
#else
	output = (T *)malloc(output_sz);
#endif
	printf("Initialize host memory buffers\n");
	printf("\tGlobal Size: %d K\n", GSZ >> 10);
	printf("\tLocal Size: %d\n", LSZ);
	printf("\tNumber of Workgroup: %d\n", GSZ / LSZ);
	printf("\tLocal Memory Buffer: %lu B\n", local_sz);
	printf("\tOutput Memory Buffer: %lu KB\n", output_sz >> 10);

///=================================================================

	printf("Run kernels\n");
	double bw;

#ifndef ALTERA
	bw = run_kernel(devices[sel], "local_mem_bandwidth/linear.cl", "linear");
	printf("\tlinear:\t%7.3f GB/s\n", bw/1024/1024/1024);
	bw = run_kernel(devices[sel], "local_mem_bandwidth/single.cl", "single");
	printf("\tsingle:\t%7.3f GB/s\n", bw/1024/1024/1024);
	bw = run_kernel(devices[sel], "local_mem_bandwidth/write_linear.cl", "write_linear");
	printf("\twrite_linear:\t%7.3f GB/s\n", bw/1024/1024/1024);
#else
	bw = run_kernel(devices[sel], "local_mem_bandwidth/linear.aocx", "linear");
	printf("\tlinear:\t%7.3f GB/s\n", bw/1024/1024/1024);
	bw = run_kernel(devices[sel], "local_mem_bandwidth/single.aocx", "single");
	printf("\tsingle:\t%7.3f GB/s\n", bw/1024/1024/1024);
	bw = run_kernel(devices[sel], "local_mem_bandwidth/write_linear.aocx", "write_linear");
	printf("\twrite_linear:\t%7.3f GB/s\n", bw/1024/1024/1024);
#endif	
	

	free(output);

}

double run_kernel(cl_device_id device, char *kernel_file, char *kernel_name) {
	int i, j;
	cl_int ret;
//	kernel_t kid = to_kernel_id(kernel_name);


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
	cl_mem output_mem;
	output_mem = clCreateBuffer(context, CL_MEM_BANK_1_ALTERA | CL_MEM_WRITE_ONLY, output_sz, NULL, &ret);
	CHECK_RC(ret, "clCreateBuffer(output_mem)")
	

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
		 *	Set kernel arguments
		 */

		ret = clSetKernelArg(kernel, 0, local_sz, NULL);
		CHECK_RC(ret, "clSetKernelArg(local)")
		ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&output_mem);
		CHECK_RC(ret, "clSetKernelArg(output)")

	
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
		ret = clEnqueueReadBuffer(queue, output_mem, CL_TRUE, 0, output_sz, output, 0, NULL, NULL);
		CHECK_RC(ret, "clEnqueueReadBuffer()")
		/*if(verify && !verify_kernel(input, output, c0, golden, input_sz, output_sz, kid)) {
			printf("Verification failed\n");
			exit(-1);
		}*/
			

				
	}
	
	/**
	 *	Calculate bandwidth
	 */
	double bytes = sizeof(T) * VECT * (EXTRA + 1) * GSZ * (iter - 1);
	return bytes/seconds;
	


}
