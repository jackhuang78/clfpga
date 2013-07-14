#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <CL/cl.h>
#include "oclutil.h"

#include "sad.h"




void sad_init(int argc, char **argv, cl_device_id *device, char **kernel_name, char **kernel_file, T **image, T **filter, T **out_host, T **out_kernel);
void sad_setup(T *image, T *filter);
void sad_host(T *image, T *filter, T *out);
void sad_kernel_setup(cl_context context, cl_mem *image_mem, cl_mem *filter_mem, cl_mem *out_mem);
void sad_kernel(cl_context context, cl_command_queue queue, cl_kernel kernel, cl_mem image_mem, cl_mem filter_mem, cl_mem out_mem, T *image, T *filter, T *out, double *times);
void sad_verify(T *out_host, T *out_kernel, T *diff);
void print_mat(char *msg, int s, T *M);


//==================================================================================================================

int main(int argc, char **argv) {
	printf(">>>>> sad.c <<<<<\n");
	setenv("CUDA_CACHE_DISABLE", "1", 1);
	RAND_INIT();

	int i;
	T *image, *filter, *out_host, *out_kernel;


	// kernel info
	char *kernel_name, *kernel_file;
	cl_device_id device;
	cl_context context;
	cl_command_queue queue;
	cl_kernel kernel;
	cl_mem image_mem, filter_mem, out_mem;

	// profile / verification info
	T diffs[ITER];
	T cold_diff, warm_diff;
	double times[ITER];
	double cold_time, warm_time;


	// Read arguments and allocate arrays.
	sad_init(argc, argv, &device, &kernel_name, &kernel_file, &image, &filter, &out_host, &out_kernel);
	printf("Device:\n");
	oclPrintDeviceInfo(device, "\t");
	printf("Kernel Name:\t%s\n", kernel_name);
	printf("Kernel File:\t%s\n", kernel_file);
	printf("Image Size:\t%d x %d\t(%lu Bytes)\n", IMAGE_S, IMAGE_S, IMAGE_SZ);
	printf("Filter Size:\t%d x %d\t(%lu Bytes)\n", FILTER_S, FILTER_S, FILTER_SZ);
	printf("Output Size:\t%d x %d\t(%lu Bytes)\n", OUT_S, OUT_S, OUT_SZ);
	printf("Temporary Size:\t%d x %d\t(%lu Bytes)\n", TEMP_S, TEMP_S, TEMP_SZ);
	printf("Local Size:\t%d x %d\n", LOCAL_S, LOCAL_S);
	printf("Global Size:\t%d x %d\n", GLOBAL_S, GLOBAL_S);

	// To simplify kernel, accept only output size that is a multiple of workgroup size
	 	
	
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
		DEBUG_PRINT(print_mat("Filter:", IMAGE_S, filter))

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
	printf("\nAvg throughput: %f\n", (ITER - 1) / warm_time);
	
	

	return 0;
}

void sad_init(int argc, char **argv, cl_device_id *device, char **kernel_name, char **kernel_file, 
		T **image, T **filter, T **out_host, T **out_kernel) {

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
	posix_memalign ((void **)image, AOCL_ALIGNMENT, IMAGE_SZ);
	posix_memalign ((void **)filter, AOCL_ALIGNMENT, FILTER_SZ);
	posix_memalign ((void **)out_host, AOCL_ALIGNMENT, OUT_SZ);
	posix_memalign ((void **)out_kernel, AOCL_ALIGNMENT, OUT_SZ);
#else
	*image = (T *)malloc(IMAGE_SZ);
	*filter = (T *)malloc(FILTER_SZ);
	*out_host = (T *)malloc(OUT_SZ);
	*out_kernel = (T *)malloc(OUT_SZ);
#endif

	
}

void sad_setup(T *image, T *filter) {
	int i;

	
	for(i = 0; i < IMAGE_S * IMAGE_S; i++)
		image[i] = (T)RAND_INT();

	for(i = 0; i < FILTER_S * FILTER_S; i++)
		filter[i] = (T)RAND_INT();


}

void sad_host(T *image, T *filter, T *out) {
	int i, j, ii, jj;

	for(i = 0; i < OUT_S; i++)
		for(j = 0; j < OUT_S; j++) {
			OUT(i,j) = 0;
			for(ii = 0; ii < FILTER_S; ii++)
				for(jj = 0; jj < FILTER_S; jj++)
					OUT(i,j) += ABS(IMAGE(i + ii, j + jj) - FILTER(ii, jj));
		}
}

void sad_kernel_setup(cl_context context, cl_mem *image_mem, cl_mem *filter_mem, cl_mem *out_mem) {
	cl_int ret;

	// Set up memory buffer
	CHECK(*image_mem = clCreateBuffer(context, CL_MEM_READ_ONLY, IMAGE_SZ, NULL, &ret))
	CHECK(*filter_mem = clCreateBuffer(context, CL_MEM_READ_ONLY, FILTER_SZ, NULL, &ret))
	CHECK(*out_mem = clCreateBuffer(context, CL_MEM_WRITE_ONLY, OUT_SZ, NULL, &ret))
}

void sad_kernel(cl_context context, cl_command_queue queue, cl_kernel kernel, cl_mem image_mem, cl_mem filter_mem, cl_mem out_mem,
			  T *image, T *filter, T *out, double *time) {

	int i;
	cl_int ret;
	cl_event event;

	size_t lsz[3] = {LOCAL_S, LOCAL_S, 1};
	size_t gsz[3] = {GLOBAL_S, GLOBAL_S, 1};
	
	

	// Set kernel arguments.
	CHECKRET(ret, clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&image_mem))
	CHECKRET(ret, clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&filter_mem))	
	CHECKRET(ret, clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&out_mem))	
	//printf("Set Kernel arguments.\n");

	// Write input data to input buffer.
	CHECKRET(ret, clEnqueueWriteBuffer(queue, image_mem, CL_TRUE, 0, IMAGE_SZ, image, 0, NULL, &event))
	CHECKRET(ret, clEnqueueWriteBuffer(queue, filter_mem, CL_TRUE, 0, FILTER_SZ, filter, 0, NULL, &event))
	//printf("Write input data to input buffer.\n");


	// Run and profile the kernel.
	clFinish(queue);
	CHECKRET(ret, clEnqueueNDRangeKernel(queue, kernel, 2, NULL, gsz, lsz, 0, NULL, &event))
	clWaitForEvents(1, &event);
	*time = oclExecutionTime(&event);
	
	//printf("launch kernel\n");

	// Read output data from output buffer.
	CHECKRET(ret, clEnqueueReadBuffer(queue, out_mem, CL_TRUE, 0, OUT_SZ, out, 0, NULL, NULL))		
	//printf("Read output data from output buffer.\n");
	


}

void sad_verify(T *out_host, T *out_kernel, T *diff) {
	*diff = 0;

	int i;
	for(i = 0; i < OUT_S * OUT_S; i++)
		*diff += out_host[i] != out_kernel[i];		


}


void print_mat(char *msg, int s, T *M) {
	int i, j;

	printf("%s\n", msg);
	for(i = 0; i < s; i++) {
		for(j = 0; j < s; j++)
			printf("%5d", M[i * s + j]);
		printf("\n");
	}
}


