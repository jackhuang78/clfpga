#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <CL/cl.h>
#include "oclutil.h"

#include "sad.h"


void sad_init(int argc, char **argv, 
		  	  cl_device_id *device, char **kernel_name, char **kernel_file, 
			  int **image, int **filter, int **out_host, int **out_kernel, int image_s, int filter_s, int out_s);

void sad_setup(int *image, int *filter, int image_s, int filter_s);
void sad_host(int *image, int *filter, int *out, int image_s, int filter_s, int out_s);

void sad_kernel_setup(cl_context context, int image_s, int filter_s, int out_s,
					  cl_mem *image_mem, cl_mem *filter_mem, cl_mem *out_mem);

void sad_kernel(cl_context context, cl_command_queue queue, cl_kernel kernel, cl_mem image_mem, cl_mem filter_mem, cl_mem out_mem,
			  int *image, int *filter, int *out, int image_s, int filter_s, int out_s,
			  double *times);

void sad_verify(int *out_host, int *out_kernel, int out_s, int *diff);

void print_mat(char *msg, int s, int *M);

//==================================================================================================================

int main(int argc, char **argv) {
	printf(">>>>> sad.c <<<<<\n");

	int i, diff;
	double time;
	int *image, *filter, *out_host, *out_kernel;
	char *kernel_name, *kernel_file;
	cl_device_id device;
	cl_context context;
	cl_command_queue queue;
	cl_kernel kernel;
	cl_mem image_mem, filter_mem, out_mem;

	sad_init(argc, argv, &device, &kernel_name, &kernel_file, &image, &filter, &out_host, &out_kernel, IMAGE_S, FILTER_S, OUT_S);

	printf("Device:\n");
	oclPrintDeviceInfo(device, "\t");
	printf("Kernel Name:\t%s\n", kernel_name);
	printf("Kernel File:\t%s\n", kernel_file);
	printf("Image Size:\t%d x %d\t(%u Bytes)\n", IMAGE_S, IMAGE_S, SIZEOF(IMAGE_S, int));
	printf("Filter Size:\t%d x %d\t(%u Bytes)\n", FILTER_S, FILTER_S, SIZEOF(FILTER_S, int));
	printf("Output Size:\t%d x %d\t(%u Bytes)\n", OUT_S, OUT_S, SIZEOF(FILTER_S, int));

	
	if(OUT_S % WG_S != 0) {
		printf("Error: output size (%d) is not divisible by workgroup size (%d).\n", OUT_S, WG_S);
		return -1;
	} 	
		


	

	
	// Set up OpenCL context, command queue, and kernel.
	if(oclQuickSetup(device, kernel_file, kernel_name, &context, &queue, &kernel)) {
		printf("Setup failed.\n");
		return -1;
	}
	sad_kernel_setup(context, IMAGE_S, FILTER_S, OUT_S, &image_mem, &filter_mem, &out_mem);
	

	int diffs[ITER];
	double times[ITER];
	for(i = 0; i < ITER; i++) {
		//printf("Iter %d\n", i);

		// Create image and filter
		sad_setup(image, filter, IMAGE_S, FILTER_S);
		//print_mat("Image:", IMAGE_S, image);
		//print_mat("Filter:", FILTER_S, filter);

		// Run host as reference
		sad_host(image, filter, out_host, IMAGE_S, FILTER_S, OUT_S);
		//print_mat("Host Output:", OUT_S, out_host);

		sad_kernel(context, queue, kernel, image_mem, filter_mem, out_mem,
				image, filter, out_kernel, IMAGE_S, FILTER_S, OUT_S, &times[i]);
		//print_mat("Kernel Output:", OUT_S, out_kernel);

		sad_verify(out_host, out_kernel, OUT_S, &diffs[i]);

	}

	printf("%15s%15s%15s\n", "Iter", "Time", "Diff");
	double cold_time = 0.0, warm_time = 0.0;
	int cold_diff = 0.0, warm_diff = 0.0;
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

void sad_init(int argc, char **argv, 
			  cl_device_id *device, char **kernel_name, char **kernel_file,
			  int **image, int **filter, int **out_host, int **out_kernel, int image_s, int filter_s, int out_s) {

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
	posix_memalign ((void **)image, AOCL_ALIGNMENT, SIZEOF(image_s, int));
	posix_memalign ((void **)filter, AOCL_ALIGNMENT, SIZEOF(filter_s, int));
	posix_memalign ((void **)out_host, AOCL_ALIGNMENT, SIZEOF(out_s, int));
	posix_memalign ((void **)out_kernel, AOCL_ALIGNMENT, SIZEOF(out_s, int));
#else
	*image = (int *)malloc(SIZEOF(image_s, int));
	*filter = (int *)malloc(SIZEOF(filter_s, int));
	*out_host = (int *)malloc(SIZEOF(out_s, int));
	*out_kernel = (int *)malloc(SIZEOF(out_s, int));
#endif

	
}

void sad_setup(int *image, int *filter, int image_s, int filter_s) {
	int i;

	RAND_INIT();
	for(i = 0; i < image_s * image_s; i++)
		(image)[i] = RAND_INT();

	for(i = 0; i < filter_s * filter_s; i++)
		(filter)[i] = RAND_INT();


}

void sad_host(int *image, int *filter, int *out, int image_s, int filter_s, int out_s) {
	int i, j, ii, jj;

	for(i = 0; i < out_s; i++)
		for(j = 0; j < out_s; j++) {
			out[IDX(i, j, out_s)] = 0;
			for(ii = 0; ii < filter_s; ii++)
				for(jj = 0; jj < filter_s; jj++)
					out[IDX(i, j, out_s)] += ABS(image[IDX(i + ii, j + jj, image_s)] - filter[IDX(ii, jj, filter_s)]);
		}
}

void sad_kernel_setup(cl_context context, int image_s, int filter_s, int out_s,
					  cl_mem *image_mem, cl_mem *filter_mem, cl_mem *out_mem) {
	cl_int ret;

	// Set up memory buffer
	CHECK(*image_mem = clCreateBuffer(context, CL_MEM_READ_WRITE, SIZEOF(image_s, int), NULL, &ret))
	CHECK(*filter_mem = clCreateBuffer(context, CL_MEM_READ_ONLY, SIZEOF(filter_s, int), NULL, &ret))
	CHECK(*out_mem = clCreateBuffer(context, CL_MEM_WRITE_ONLY, SIZEOF(out_s, int), NULL, &ret))
}

void sad_kernel(cl_context context, cl_command_queue queue, cl_kernel kernel, cl_mem image_mem, cl_mem filter_mem, cl_mem out_mem,
			  int *image, int *filter, int *out, int image_s, int filter_s, int out_s,
			  double *time) {

	int i;
	cl_int ret;
	cl_event event;

	size_t lsz[3] = {WG_S, WG_S, 0};
	size_t gsz[3] = {out_s, out_s, 0};

	// Set kernel arguments.
	CHECKRET(ret, clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&image_mem))
	CHECKRET(ret, clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&filter_mem))	
	CHECKRET(ret, clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&out_mem))	
	CHECKRET(ret, clSetKernelArg(kernel, 3, sizeof(int), (void *)&image_s))	
	CHECKRET(ret, clSetKernelArg(kernel, 4, sizeof(int), (void *)&filter_s))		 
	//printf("Set Kernel arguments.\n");

	// Write input data to input buffer.
	CHECKRET(ret, clEnqueueWriteBuffer(queue, image_mem, CL_TRUE, 0, SIZEOF(image_s, int), image, 0, NULL, &event))
	CHECKRET(ret, clEnqueueWriteBuffer(queue, filter_mem, CL_TRUE, 0, SIZEOF(filter_s, int), filter, 0, NULL, &event))
	//printf("Write input data to input buffer.\n");


	// Run and profile the kernel.
	clFinish(queue);
	CHECKRET(ret, clEnqueueNDRangeKernel(queue, kernel, 2, NULL, gsz, lsz, 0, NULL, &event))
	clWaitForEvents(1, &event);
	*time = oclExecutionTime(&event);
	
	//printf("launch kernel\n");

	// Read output data from output buffer.
	CHECKRET(ret, clEnqueueReadBuffer(queue, out_mem, CL_TRUE, 0, SIZEOF(out_s, int), out, 0, NULL, NULL))		
	//printf("Read output data from output buffer.\n");
	



	//for(i = 0; i < out_s * out_s; i++)
	//	out[i] = i;
	
	//*time = 1.0;

}

void sad_verify(int *out_host, int *out_kernel, int out_s, int *diff) {
	*diff = 0;

	int i;
	for(i = 0; i < out_s * out_s; i++)
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


