#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <CL/cl.h>
#include "oclutil.h"

#ifdef ALTERA
#define KERNEL_EXT ".aocx"
#else
#define KERNEL_EXT ".cl"
#endif

#ifndef TESTSZ
#define TESTSZ 10
#endif

#ifndef DATASZ
#define DATASZ 28
#endif

#ifndef LOCALSZ
#define LOCALSZ 128
#endif

#ifndef MARGIN
#define MARGIN 1.0
#endif


void reduce(cl_context context, cl_command_queue queue, cl_kernel kernel, int vect, int half);


int main(int argc, char **argv) {
	int i, J;
	printf("\n>>>>> reduce.c <<<<<\n\n");


	// Get all available devices.
	cl_uint num_devices;
	cl_device_id *devices;
	oclCLDevices(&num_devices, &devices);

	// Display all available devices.
	printf("OpenCL devices:\n");
	for(i = 0; i < num_devices; i++) {
		printf("Device %d:\n", i);
		oclPrintDeviceInfo(devices[i], "\t");
	}

	// Determine the selected device from command-line input.
	int dev_sel = (argc < 2) ? 0 : atoi(argv[1]);	// the second argument
	printf("\nSelect device [%d].\n\n", dev_sel);
	if(dev_sel < 0 || dev_sel >= num_devices)
		return -1;
	cl_device_id device = devices[dev_sel];

	// Determine the kernel name and file from command-line input.
	char *kernel_name = (argc < 3) ? "reduce0" : argv[2];
	char *kernel_file = (char *)malloc(50);
	kernel_file[0] = '\0';
	strcat(kernel_file, "reduce_kernels/");
	strcat(kernel_file, (argc < 4) ? kernel_name : argv[3]);
	strcat(kernel_file, KERNEL_EXT);
	printf("Kernel Name: %s\n", kernel_name);
	printf("Kernel File: %s\n", kernel_file);

	// Determine the data vectorization from kernel filename
	char *pos = strchr(kernel_name, '_');
	int vect;
	if(pos == NULL) {
		vect = 1;
	} else {
		pos++;
		vect = (*pos == '1') ? 16 : *pos - '0';
	}
	printf("Data Vector: %d\n", vect);

	// Determine if only half of the specified workitems are needed
	int half = (kernel_name[6] >= '3') ? 1 : 0; 
	printf("Half: %d\n", half);


	// Set up OpenCL context, command queue, and kernel.
	cl_context context;
	cl_command_queue queue;
	cl_kernel kernel;
	if(oclQuickSetup(device, kernel_file, kernel_name, &context, &queue, &kernel)) {
		return -1;
	}
	reduce(context, queue, kernel, vect, half);

	return 0;
}

void reduce(cl_context context, cl_command_queue queue, cl_kernel kernel, 
	int vect, int half) {
	int i, j;
	cl_int ret;

	// Calculate numbers...
	int n = 1 << DATASZ;
	int nvec = n / vect; 
	size_t lsz = LOCALSZ;
	int nwg = ROUNDUP(nvec, lsz);
	size_t gsz = lsz * nwg;

	if(half) {
		lsz /= 2;
		gsz /= 2;
	}

	

	printf("Number of elements: %d\n", n);
	printf("Number of Workgroups: %d\n", nwg);
	printf("Local size: %u\n", (unsigned)lsz);
	printf("Global size: %u\n", (unsigned)gsz);

	// Calcualte more numbers...
	size_t in_data_sz = sizeof(float) * n;
	size_t out_data_sz = sizeof(float) * nwg;
	float *in_data = (float *)malloc(in_data_sz);
	float *out_data = (float *)malloc(out_data_sz);

	printf("Input Data Size: %u\n", (unsigned)in_data_sz);
	printf("Output Data Size: %u\n", (unsigned)out_data_sz);

	// Create memory buffers
	cl_mem in_data_mem, out_data_mem;
	CHECK(in_data_mem = clCreateBuffer(context, CL_MEM_READ_WRITE, in_data_sz, NULL, &ret))
	CHECK(out_data_mem = clCreateBuffer(context, CL_MEM_WRITE_ONLY, out_data_sz, NULL, &ret))


	float total_time = 0.0;
	float total_error = 0.0;
	float total_time_except_first = 0.0;
	float total_error_except_first = 0.0;
	printf("Run\tTime(sec)\tError(%%)\tStatus\tExpected\tActual\n");
	for(i = 0; i < TESTSZ; i++) {
		
	
		// Generate input data.
		for(j = 0; j < n; j++) {
			in_data[j] = rand_float();
			in_data[j] = (j % 2) ? in_data[j] : -in_data[j];
		}
	
		// Set kernel arguments.
		CHECKRET(ret, clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&in_data_mem))
		CHECKRET(ret, clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&out_data_mem))	
		CHECKRET(ret, clSetKernelArg(kernel, 2, sizeof(int), (void *)&nvec))	
		CHECKRET(ret, clSetKernelArg(kernel, 3, lsz * sizeof(float) * vect, NULL))		 

		// Write input data to input buffer.
		CHECKRET(ret, clEnqueueWriteBuffer(queue, in_data_mem, CL_TRUE, 0, in_data_sz, in_data, 0, NULL, NULL))

		// Run and profile the kernel.
		clFinish(queue);
		cl_event event;
		CHECKRET(ret, clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &gsz, &lsz, 0, NULL, &event))
		clWaitForEvents(1, &event);
		float time = oclExecutionTime(&event);
		
		//printf("time: %f\n", time);
	
		// Read output data from output buffer.
		//printf("out_data_sz: %u, out_data_mem: %u\n", out_data_sz, sizeof(out_data_mem));
		CHECKRET(ret, clEnqueueReadBuffer(queue, out_data_mem, CL_TRUE, 0, out_data_sz, out_data, 0, NULL, NULL))

		// Verify results
		float actual = 0.0;
		float expected = 0.0;
		for(j = 0; j < n; j++)
			expected += in_data[j];
		for(j = 0; j < nwg; j++) 
			actual += out_data[j];
		float error = (actual - expected) / expected * 100;
		error = (error < 0) ? -error : error;
		char *status = (error > MARGIN) ? "FAIL" : "pass";

		printf("%d\t%f\t%f\t%s\t%f\t%f\n", i, time, error,  status, expected, actual);

		total_time += time;
		total_error += error;

		total_time_except_first += (i == 0) ? 0.0 : time;
		total_error_except_first += (i == 0) ? 0.0 : error;
	}

	float avg_time = total_time / TESTSZ;
	float avg_error = total_error / TESTSZ;
	char *avg_status = (avg_error > MARGIN) ? "FAIL" : "pass";
	printf("%s\t%f\t%f\t%s\n", "AVG(c)", avg_time, avg_error, avg_status);

	avg_time = total_time_except_first / (TESTSZ-1);
	avg_error = total_error_except_first / (TESTSZ-1);
	avg_status = (avg_error > MARGIN) ? "FAIL" : "pass";

	printf("%s\t%f\t%f\t%s\n", "AVG(w)", avg_time, avg_error, avg_status);

	printf("finishes execution\n");
}
