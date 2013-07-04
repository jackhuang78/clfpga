#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <CL/cl.h>
#include "oclutil.h"

#include "sad.h"


void sad_init(int argc, char **argv, cl_device_id *device, char **kernel_name, char **kernel_file, int **image, int **filter, int **out_host, int **out_kernel);
void sad_setup(int *image, int *filter);
void sad_host(int *image, int *filter, int *out);
void sad_kernel_setup(cl_context context, cl_mem *image_mem, cl_mem *filter_mem, cl_mem *out_mem);
void sad_kernel(cl_context context, cl_command_queue queue, cl_kernel kernel, cl_mem image_mem, cl_mem filter_mem, cl_mem out_mem, int *image, int *filter, int *out, double *times);
void sad_verify(int *out_host, int *out_kernel, int *diff);
void print_mat(char *msg, int s, int *M);

//==================================================================================================================

int main(int argc, char **argv) {
	printf(">>>>> sad.c <<<<<\n");

	int i;
	
	// input/output data
	int *image, *filter, *out_host, *out_kernel;

	// kernel info
	char *kernel_name, *kernel_file;
	cl_device_id device;
	cl_context context;
	cl_command_queue queue;
	cl_kernel kernel;
	cl_mem image_mem, filter_mem, out_mem;

	// profile / verification info
	int diffs[ITER];
	double times[ITER];
	double cold_time, warm_time;
	int cold_diff, warm_diff;

	// Read arguments and allocate arrays.
	sad_init(argc, argv, &device, &kernel_name, &kernel_file, &image, &filter, &out_host, &out_kernel);
	printf("Device:\n");
	oclPrintDeviceInfo(device, "\t");
	printf("Kernel Name:\t%s\n", kernel_name);
	printf("Kernel File:\t%s\n", kernel_file);
	printf("Image Size:\t%d x %d\t(%lu Bytes)\n", IMAGE_S, IMAGE_S, SIZEOF(IMAGE_S, int));
	printf("Filter Size:\t%d x %d\t(%lu Bytes)\n", FILTER_S, FILTER_S, SIZEOF(FILTER_S, int));
	printf("Output Size:\t%d x %d\t(%lu Bytes)\n", OUT_S, OUT_S, SIZEOF(FILTER_S, int));
	printf("Temporary Size:\t%d x %d\t(%lu Bytes)\n", TEMP_S, TEMP_S, SIZEOF(TEMP_S, int));

	// To simplify kernel, accept only output size that is a multiple of workgroup size
	if(OUT_S % WG_S != 0) {
		printf("Error: output size (%d) is not divisible by workgroup size (%d).\n", OUT_S, WG_S);
		return -1;
	} 	
	
	// Set up OpenCL context, command queue, and kernel.
	if(oclQuickSetup(device, kernel_file, kernel_name, &context, &queue, &kernel)) {
		printf("Setup failed.\n");
		return -1;
	}
	
	
	// Initialize OpenCL memory buffer 
	sad_kernel_setup(context, &image_mem, &filter_mem, &out_mem);
	printf("Begin executing kernels");
	for(i = 0; i < ITER; i++) {
		printf(".");

		// Create image and filter.
		sad_setup(image, filter);
		DEBUG_PRINT(print_mat("Image:", IMAGE_S, image))
		DEBUG_PRINT(print_mat("Filter:", FILTER_S, filter))

		// Run host as reference.
		sad_host(image, filter, out_host);
		DEBUG_PRINT(print_mat("Host Output:", OUT_S, out_host))

		// Run kernel.
		sad_kernel(context, queue, kernel, image_mem, filter_mem, out_mem, image, filter, out_kernel, &times[i]);
		DEBUG_PRINT(print_mat("Kernel Output:", OUT_S, out_kernel))

		// Verify results.
		sad_verify(out_host, out_kernel, &diffs[i]);

	}
	printf("\n");

	// Display time and verification result.
	cold_time = warm_time = 0.0;
	cold_diff = warm_diff = 0;
	printf("%15s%15s%15s\n", "Iter", "Time", "Diff");
	for(i = 0; i < ITER; i++) {
		printf("%15d%15f%15d\n", i, times[i], diffs[i]);
		cold_time += times[i];
		cold_diff += diffs[i];
		warm_time += (i == 0) ? 0 : times[i];
		warm_diff += (i == 0) ? 0 : diffs[i];
	}
	printf("%15s%15f%15d\n", "Avg(cold)", cold_time / ITER, cold_diff / ITER);
	printf("%15s%15f%15d\n", "Avg(warm)", warm_time / (ITER - 1), warm_diff / (ITER - 1));	
	
	

	return 0;
}

void sad_init(int argc, char **argv, cl_device_id *device, char **kernel_name, char **kernel_file, 
		int **image, int **filter, int **out_host, int **out_kernel) {

	// Get all available devices.
	cl_uint num_devices;
	cl_device_id *devices;
	oclCLDevices(&num_devices, &devices);

	// The 2nd argument is the selected device.
	int dev_sel = (argc < 2) ? 0 : atoi(argv[1]);	
	dev_sel = (dev_sel < 0 || dev_sel >= num_devices) ? 0 : dev_sel;
	*device = devices[dev_sel];

	// The 3rd argument is the kernel name.
	*kernel_name = (argc < 3) ? "sad1" : argv[2];

	// The 4th argument is the kernel file.
#ifdef ALTERA
	*kernel_file = (argc < 4) ? "sad/sad1.aocx" : argv[3];
#else
	*kernel_file = (argc < 4) ? "sad/sad1.cl" : argv[3];
#endif	

	// Allocate memory

#ifdef ALTERA
	posix_memalign ((void **)image, AOCL_ALIGNMENT, SIZEOF(IMAGE_S, int));
	posix_memalign ((void **)filter, AOCL_ALIGNMENT, SIZEOF(FILTER_S, int));
	posix_memalign ((void **)out_host, AOCL_ALIGNMENT, SIZEOF(OUT_S, int));
	posix_memalign ((void **)out_kernel, AOCL_ALIGNMENT, SIZEOF(OUT_S, int));
#else
	*image = (int *)malloc(SIZEOF(IMAGE_S, int));
	*filter = (int *)malloc(SIZEOF(FILTER_S, int));
	*out_host = (int *)malloc(SIZEOF(OUT_S, int));
	*out_kernel = (int *)malloc(SIZEOF(OUT_S, int));
#endif

	
}

void sad_setup(int *image, int *filter) {
	int i;

	RAND_INIT();
	for(i = 0; i < IMAGE_S * IMAGE_S; i++)
		(image)[i] = RAND_INT();

	for(i = 0; i < FILTER_S * FILTER_S; i++)
		(filter)[i] = RAND_INT();


}

void sad_host(int *image, int *filter, int *out) {
	int i, j, ii, jj;

	for(i = 0; i < OUT_S; i++)
		for(j = 0; j < OUT_S; j++) {
			out[IDX(i, j, OUT_S)] = 0;
			for(ii = 0; ii < FILTER_S; ii++)
				for(jj = 0; jj < FILTER_S; jj++)
					out[IDX(i, j, OUT_S)] += ABS(image[IDX(i + ii, j + jj, IMAGE_S)] - filter[IDX(ii, jj, FILTER_S)]);
		}
}

void sad_kernel_setup(cl_context context, cl_mem *image_mem, cl_mem *filter_mem, cl_mem *out_mem) {
	cl_int ret;

	// Set up memory buffer
	CHECK(*image_mem = clCreateBuffer(context, CL_MEM_READ_WRITE, SIZEOF(IMAGE_S, int), NULL, &ret))
	CHECK(*filter_mem = clCreateBuffer(context, CL_MEM_READ_ONLY, SIZEOF(FILTER_S, int), NULL, &ret))
	CHECK(*out_mem = clCreateBuffer(context, CL_MEM_WRITE_ONLY, SIZEOF(OUT_S, int), NULL, &ret))
}

void sad_kernel(cl_context context, cl_command_queue queue, cl_kernel kernel, cl_mem image_mem, cl_mem filter_mem, cl_mem out_mem,
			  int *image, int *filter, int *out, double *time) {

	int i;
	cl_int ret;
	cl_event event;

	size_t lsz[3] = {WG_S, WG_S, 1};
	size_t gsz[3] = {OUT_S, OUT_S, 1};
	
	//printf("lsz: %u, %u, %u\n", (unsigned int)lsz[0], (unsigned int)lsz[1], (unsigned int)lsz[2]);
	//printf("gsz: %u, %u, %u\n", (unsigned int)gsz[0], (unsigned int)gsz[1], (unsigned int)gsz[2]);

	

	// Set kernel arguments.
	CHECKRET(ret, clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&image_mem))
	CHECKRET(ret, clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&filter_mem))	
	CHECKRET(ret, clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&out_mem))		
	//printf("Set Kernel arguments.\n");

	// Write input data to input buffer.
	CHECKRET(ret, clEnqueueWriteBuffer(queue, image_mem, CL_TRUE, 0, SIZEOF(IMAGE_S, int), image, 0, NULL, &event))
	CHECKRET(ret, clEnqueueWriteBuffer(queue, filter_mem, CL_TRUE, 0, SIZEOF(FILTER_S, int), filter, 0, NULL, &event))
	//printf("Write input data to input buffer.\n");


	// Run and profile the kernel.
	clFinish(queue);
	CHECKRET(ret, clEnqueueNDRangeKernel(queue, kernel, 2, NULL, gsz, lsz, 0, NULL, &event))
	clWaitForEvents(1, &event);
	*time = oclExecutionTime(&event);
	
	//printf("launch kernel\n");

	// Read output data from output buffer.
	CHECKRET(ret, clEnqueueReadBuffer(queue, out_mem, CL_TRUE, 0, SIZEOF(OUT_S, int), out, 0, NULL, NULL))		
	//printf("Read output data from output buffer.\n");
	



	//for(i = 0; i < OUT_S * OUT_S; i++)
	//	out[i] = i;
	
	//*time = 1.0;

}

void sad_verify(int *out_host, int *out_kernel, int *diff) {
	*diff = 0;

	int i;
	for(i = 0; i < OUT_S * OUT_S; i++)
		*diff += out_host[i] != out_kernel[i];		


}


void print_mat(char *msg, int s, int *M) {
	int i, j;

	printf("%s\n", msg);
	for(i = 0; i < s; i++) {
		for(j = 0; j < s; j++)
			printf("%5d", M[IDX(i,j,s)]);
		printf("\n");
	}
}


