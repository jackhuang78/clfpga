#include <stdio.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <CL/cl.h>
#include <CL/cl_ext.h>
#include "oclutil.h"

#include "test/test.h"

#ifndef CL_MEM_BANK_1_ALTERA
#define CL_MEM_BANK_1_ALTERA 0
#define CL_MEM_BANK_2_ALTERA 0
#endif

#define AOCL_ALIGNMENT 64

#define T float
#define blockSize 128


void getStat(long *times, long *avg, long *std);

int main(int argc, char **argv) {
	printf("======== BEGIN test.c ========\n");

	int i, j;
	cl_int ret, ret0, ret1, ret2, ret3;
	cl_uint num_devices;
	cl_device_id *devices;

	/*
		Get devices
	*/
	ret = oclGetDevices(&num_devices, &devices);
	if(ret != CL_SUCCESS) {
		printf("ERROR in oclGetDevices(): %s\n", oclReturnCodeToString(ret));
		return 1;
	}

	if(argc < 5) {
		printf("Not enough arguments\n");
		return -1;
	}
	
	
	/*
		Parse arguments
	*/
	int sel = atoi(argv[1]);
	char *kernel_file = argv[2];
	char *kernel_name = argv[3];
	int form = atoi(argv[4]);
	int num_data = atoi(argv[5]) << 20;
	int iter = atoi(argv[6]);

	printf("Running Test on device %d:\n", sel);
	printf("\tKernel file: %s\n", kernel_file);
	printf("\tKernel: %s\n", kernel_name);
	pritnf("\tKernel form: %d\n", form);
	printf("\tData size: %d\n", data_size);
	printf("\tIterations: %d\n", iter);
	if(sel >= num_devices) {
		printf("Device #%d does not exist.\n", sel);
		return -1;
	}


	/*
		Build OCL kernel
	*/
	printf("Build OCL Kernels\n");
	cl_context context;
	cl_command_queue queue;
	cl_kernel kernel;
	cl_build_status status;
	char *log;
	ret = oclKernelSetup(devices[sel], kernel_file, kernel_name, 
		&context, &queue, &kernel, &status, &log);
	if(ret != CL_SUCCESS) {
		printf("ERROR in oclKernelSetup(): %s\n", oclReturnCodeToString(ret));
		return -1;
	} else if(status != CL_BUILD_SUCCESS) {
		printf("Build status:%s\n", oclBuildStatusToString(status));
		printf("Build log:\n%s\n", log);
		return -1;
	}

	/* 
		Memory Allocation
	*/
	printf("Memory Allocation\n");
	size_t gsz = num_data;
	size_t lsz = blockSize;
	size_t nwg = data_size / lsz;
	size_t in_data_sz = sizeof(T) * num_data;
	size_t out_data_sz = sizeof(T) * nwg;
	T *in_data, *out_data;
	cl_mem in_data_mem, out_data_mem;
	printf("\tData Unit Size: %lu\n", sizeof(T));
	printf("\tGlobal Size: %lu\n", gsz);
	printf("\tLocal Size: %lu\n", lsz);
	printf("\tNumber of Workgroups: %lu\n", nwg)
	printf("\tInput Memory Size: %u \n", (unsigned)in_data_sz);
	printf("\tOutput Memory Size: %u \n", (unsigned)out_data_sz);
	
#ifdef ALTERA		
	posix_memalign ((void **)&in_data, AOCL_ALIGNMENT, in_data_sz);
	posix_memalign ((void **)&out_data, AOCL_ALIGNMENT, out_data_sz);	
#else
	in_data = (T *)malloc(in_data_sz);
	out_data = (T *)malloc(out_data_sz);		
#endif
	in_data_mem = clCreateBuffer(context, CL_MEM_BANK_1_ALTERA | CL_MEM_READ_ONLY, in_data_sz, NULL, &ret0);
	out_data_mem = clCreateBuffer(context, CL_MEM_BANK_1_ALTERA | CL_MEM_WRITE_ONLY, out_data_sz, NULL, &ret1);
	if(ret0 != CL_SUCCESS | ret1 != CL_SUCCESS) {
		printf("ERROR in clCreateBuffer(): %s, %s\n", oclReturnCodeToString(ret0), oclReturnCodeToString(ret1));
		return -1;
	}

	/*
		Set Kernel Arguments
	*/
	ret0 = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&in_data_mem);
	ret1 = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&out_data_mem);
	ret2 = clSetKernelArg(kernel, 2, sizeof(int), (void *)&num_data);
	ret3 = clSetKernelArg(kernel, 3, lsz * sizeof(T), NULL);
	if(ret0 != CL_SUCCESS | ret1 != CL_SUCCESS | ret2 != CL_SUCCESS | ret3 != CL_SUCCESS) {
		printf("ERROR in clSetKernelArg(): %s, %s, %s, %s\n", oclReturnCodeToString(ret0), oclReturnCodeToString(ret1), oclReturnCodeToString(ret2), oclReturnCodeToString(ret3));
		return -1;
	}

	long times[iter];
	int errors[iter];
	srand(time(NULL));
	for(i = 0; i < iter; i++) {
	
		/*
			Enqueue Write Buffer
		*/
		for(j = 0; j < num_data; j++) {
			in_data[j] = (float)rand()/(float)RAND_MAX;
			out_data[j] = (float)rand()/(float)RAND_MAX;
		}
		ret0 = clEnqueueWriteBuffer(queue, in_data_mem, CL_TRUE, 0, in_data_sz, in_data, 0, NULL, NULL);
		if(ret0 != CL_SUCCESS) {
			printf("ERROR in clEnqueueWriteBuffer(): %s\n", oclReturnCodeToString(ret0));
			return -1;
		}
		
		/*
			Enqueue N-D Range Kernel
		*/
		clFinish(queue);
		cl_event event;
		ret = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &gsz, &lsz, 0, NULL, &event);
		if(ret != CL_SUCCESS) {
			printf("ERROR in clEnqueueNDRangeKernel(): %s\n", oclReturnCodeToString(ret));				
			return -1;
		}
		clWaitForEvents(1, &event);

		/* 
			Get Profiling Info 
		*/
		cl_ulong queued, submit, start, end;
		ret = oclGetProfilingInfo(&event, &queued, &submit, &start, &end);
		if(ret != CL_SUCCESS) {
			printf("ERROR in oclGetProfilingInfo(): %s\n", oclReturnCodeToString(ret));				
			return -1;
		}

		/*
			Read output and verify
		*/
		ret0 = clEnqueueReadBuffer(queue, out_data0_mem, CL_TRUE, 0, out_data_sz, out_data0, 0, NULL, NULL);
		ret1 = clEnqueueReadBuffer(queue, out_data1_mem, CL_TRUE, 0, out_data_sz, out_data1, 0, NULL, NULL);
		if(ret0 != CL_SUCCESS | ret1 != CL_SUCCESS) {
			printf("ERROR in clEnqueueReadBuffer(): %s, %s\n", oclReturnCodeToString(ret0), oclReturnCodeToString(ret1));
			return -1;
		}

		errors[i] = 0;
		for(j = 0; j < gsz; j++) {
			errors[i] += in_data0[j] != out_data0[j];
			if(check == 1)
				errors[i] += in_data1[j] != out_data1[j];
		
//				printf("%f != %f\t%f!=%f\n", in_data0[j], out_data0[j], in_data1[j], out_data1[j]);

		}
		
		times[i] = end - start;
	}

	long tot_time = 0;
	int tot_error = 0;
	FILE *f = fopen("test.out", "w");
	for(i = 0; i < iter; i++) {

		if(i > 0) {
			tot_time += times[i];
			tot_error += errors[i];
			fprintf(f, "%d\n", times[i]);
		} else {
			fprintf(f, "%d\n", data_size);
		}
		
	}
	fprintf(f, "%d\n", tot_error);
	fclose(f);
	printf("Iterations:\t%d\n", iter);
	printf("AVG time:\t%ld ns\n", tot_time / (iter - 1));
	printf("Total errors:\t%d\n", tot_error);


	printf("======= END reduce.c =======\n");
	
	return 0;

}

/*
void getStat(long *times, long *avg, long *std) {
	int i;	
	long tot = 0;

	
	for(i = 0; i < iter; i++) {
		tot += times[i];
	}
	*avg = tot / iter;

	long sqtot = 0;
	for(i = 0; i < ITER; i++) {
		sqtot += (times[i] - *avg) * (times[i] - *avg);
	}
	*std = sqrt(sqtot / ITER);
		
}*/



