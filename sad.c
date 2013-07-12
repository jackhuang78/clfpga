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

int image_s = IMAGE_S;
int filter_s = FILTER_S;
int out_s = OUT_S;
int temp_s = TEMP_S;

int main(int argc, char **argv) {
	printf(">>>>> sad.c <<<<<\n");
	setenv("CUDA_CACHE_DISABLE", "1", 1);

	if(out_s % WG_S != 0) {
		printf("Warning: output size (%d) is not divisible by workgroup size (%d).\n", out_s, WG_S);
		printf("Use %d for image size instead.\n", image_s - (out_s % WG_S) + WG_S);
		image_s = image_s - (out_s % WG_S) + WG_S;
		out_s = (image_s - filter_s + 1);
	}

	int i;

	
	// input/output data
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
	printf("Image Size:\t%d x %d\t(%lu Bytes)\n", image_s, image_s, SIZEOF(image_s, T));
	printf("Filter Size:\t%d x %d\t(%lu Bytes)\n", filter_s, filter_s, SIZEOF(filter_s, T));
	printf("Output Size:\t%d x %d\t(%lu Bytes)\n", out_s, out_s, SIZEOF(filter_s, T));
	printf("Temporary Size:\t%d x %d\t(%lu Bytes)\n", temp_s, temp_s, SIZEOF(temp_s, T));
	printf("Workgroup Size:\t%d x %d\n", WG_S, WG_S);

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
		DEBUG_PRINT(print_mat("Image:", image_s, image))
		DEBUG_PRINT(print_mat("Filter:", filter_s, filter))

		// Run host as reference.
		sad_host(image, filter, out_host);
		DEBUG_PRINT(print_mat("Host Output:", out_s, out_host))

		// Run kernel.
		sad_kernel(context, queue, kernel, image_mem, filter_mem, out_mem, image, filter, out_kernel, &times[i]);
		DEBUG_PRINT(print_mat("Kernel Output:", out_s, out_kernel))

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
	posix_memalign ((void **)image, AOCL_ALIGNMENT, SIZEOF(image_s, T));
	posix_memalign ((void **)filter, AOCL_ALIGNMENT, SIZEOF(filter_s, T));
	posix_memalign ((void **)out_host, AOCL_ALIGNMENT, SIZEOF(out_s, T));
	posix_memalign ((void **)out_kernel, AOCL_ALIGNMENT, SIZEOF(out_s, T));
#else
	*image = (T *)malloc(SIZEOF(image_s, T));
	*filter = (T *)malloc(SIZEOF(filter_s, T));
	*out_host = (T *)malloc(SIZEOF(out_s, T));
	*out_kernel = (T *)malloc(SIZEOF(out_s, T));
#endif

	
}

void sad_setup(T *image, T *filter) {
	int i;

	RAND_INIT();
	for(i = 0; i < image_s * image_s; i++)
		image[i] = (T)RAND_INT();

	for(i = 0; i < filter_s * filter_s; i++)
		filter[i] = (T)RAND_INT();


}

void sad_host(T *image, T *filter, T *out) {
	int i, j, ii, jj;

	for(i = 0; i < out_s; i++)
		for(j = 0; j < out_s; j++) {
			out[IDX(i, j, out_s)] = 0;
			for(ii = 0; ii < filter_s; ii++)
				for(jj = 0; jj < filter_s; jj++)
					out[IDX(i, j, out_s)] += ABS(image[IDX(i + ii, j + jj, image_s)] - filter[IDX(ii, jj, filter_s)]);
		}
}

void sad_kernel_setup(cl_context context, cl_mem *image_mem, cl_mem *filter_mem, cl_mem *out_mem) {
	cl_int ret;

	// Set up memory buffer
	CHECK(*image_mem = clCreateBuffer(context, CL_MEM_READ_ONLY, SIZEOF(image_s, T), NULL, &ret))
	CHECK(*filter_mem = clCreateBuffer(context, CL_MEM_READ_ONLY, SIZEOF(filter_s, T), NULL, &ret))
	CHECK(*out_mem = clCreateBuffer(context, CL_MEM_WRITE_ONLY, SIZEOF(out_s, T), NULL, &ret))
}

void sad_kernel(cl_context context, cl_command_queue queue, cl_kernel kernel, cl_mem image_mem, cl_mem filter_mem, cl_mem out_mem,
			  T *image, T *filter, T *out, double *time) {

	int i;
	cl_int ret;
	cl_event event;

	size_t lsz[3] = {WG_S, WG_S, 1};
	size_t gsz[3] = {out_s, out_s, 1};
	
	//printf("lsz: %u, %u, %u\n", (unsigned int)lsz[0], (unsigned int)lsz[1], (unsigned int)lsz[2]);
	//printf("gsz: %u, %u, %u\n", (unsigned int)gsz[0], (unsigned int)gsz[1], (unsigned int)gsz[2]);

	

	// Set kernel arguments.
	CHECKRET(ret, clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&image_mem))
	CHECKRET(ret, clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&filter_mem))	
	CHECKRET(ret, clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&out_mem))		
	CHECKRET(ret, clSetKernelArg(kernel, 3, SIZEOF(temp_s, T), NULL))	
//	CHECKRET(ret, clSetKernelArg(kernel, 4, SIZEOF(filter_s, T), NULL))	
	CHECKRET(ret, clSetKernelArg(kernel, 4, sizeof(int), (void *)&image_s))	
	CHECKRET(ret, clSetKernelArg(kernel, 5, sizeof(int), (void *)&filter_s))	
	CHECKRET(ret, clSetKernelArg(kernel, 6, sizeof(int), (void *)&out_s))	
	CHECKRET(ret, clSetKernelArg(kernel, 7, sizeof(int), (void *)&temp_s))	
	//printf("Set Kernel arguments.\n");

	// Write input data to input buffer.
	CHECKRET(ret, clEnqueueWriteBuffer(queue, image_mem, CL_TRUE, 0, SIZEOF(image_s, T), image, 0, NULL, &event))
	CHECKRET(ret, clEnqueueWriteBuffer(queue, filter_mem, CL_TRUE, 0, SIZEOF(filter_s, T), filter, 0, NULL, &event))
	//printf("Write input data to input buffer.\n");


	// Run and profile the kernel.
	clFinish(queue);
	CHECKRET(ret, clEnqueueNDRangeKernel(queue, kernel, 2, NULL, gsz, lsz, 0, NULL, &event))
	clWaitForEvents(1, &event);
	*time = oclExecutionTime(&event);
	
	//printf("launch kernel\n");

	// Read output data from output buffer.
	CHECKRET(ret, clEnqueueReadBuffer(queue, out_mem, CL_TRUE, 0, SIZEOF(out_s, T), out, 0, NULL, NULL))		
	//printf("Read output data from output buffer.\n");
	



	//for(i = 0; i < out_s * out_s; i++)
	//	out[i] = i;
	
	//*time = 1.0;

}

void sad_verify(T *out_host, T *out_kernel, T *diff) {
	*diff = 0;

	int i;
	for(i = 0; i < out_s * out_s; i++)
		*diff += out_host[i] != out_kernel[i];		


}


void print_mat(char *msg, int s, T *M) {
	int i, j;

	printf("%s\n", msg);
	for(i = 0; i < s; i++) {
		for(j = 0; j < s; j++)
			printf("%5d", M[IDX(i,j,s)]);
		printf("\n");
	}
}


