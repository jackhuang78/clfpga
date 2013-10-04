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
#define DATASZ 25
#endif

#ifndef LOCALSZ
#define LOCALSZ 128
#endif

#ifndef MARGIN
#define MARGIN 1.0
#endif

#define AOCL_ALIGNMENT 64

void matrixMul(cl_context context, cl_command_queue queue, cl_kernel kernel, int vect);


int main(int argc, char **argv) {
	int i, J;
	printf("\n>>>>> matrixMul.c <<<<<\n\n");


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
	char *kernel_name = (argc < 3) ? "matrixMul" : argv[2];
	char *kernel_file = (char *)malloc(50);
	kernel_file[0] = '\0';
	strcat(kernel_file, "matrixMul_kernels/");
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

	


	// Set up OpenCL context, command queue, and kernel.
	cl_context context;
	cl_command_queue queue;
	cl_kernel kernel;
	if(oclQuickSetup(device, kernel_file, kernel_name, &context, &queue, &kernel)) {
		return -1;
	}
	//reduce(context, queue, kernel, vect, half);

	return 0;
}
