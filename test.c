#include <stdio.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <CL/cl.h>
#include "oclutil.h"

#include "test/test.h"

#ifndef CL_MEM_BANK_1_ALTERA
#define CL_MEM_BANK_1_ALTERA 0
#define CL_MEM_BANK_2_ALTERA 0
#endif

#define ITER 10000

void getStat(long *times, long *avg, long *std);

int main(int argc, char **argv) {
	printf("======== BEGIN test.c ========\n");

	int i, j;
	char *char_ptr;
	size_t *size_ptr;
	cl_device_type *device_type_ptr;
	cl_uint *uint_ptr;
	cl_ulong *ulong_ptr;

	cl_int ret, ret0, ret1, ret2, ret3;
	cl_uint num_devices;
	cl_device_id *devices;

	ret = oclGetDevices(&num_devices, &devices);
	if(ret != CL_SUCCESS) {
		printf("ERROR in oclGetDevices(): %s\n", oclReturnCodeToString(ret));
		return 1;
	}
	printf("Devices found: %u\n", num_devices);


	for(i = 0; i < num_devices; i++) {
		printf("Device %d:\n", i);

		ret = oclGetDeviceInfo(devices[i], CL_DEVICE_NAME, (void **)&char_ptr);
		printf("\tNAME: %s\n", char_ptr);

		ret = oclGetDeviceInfo(devices[i], CL_DEVICE_TYPE, (void **)&device_type_ptr);
		printf("\tTYPE: %s\n", oclDeviceTypeToString(device_type_ptr[0]));

		ret = oclGetDeviceInfo(devices[i], CL_DEVICE_VENDOR, (void **)&char_ptr);
		printf("\tVENDOR: %s\n", char_ptr);
		
		ret = oclGetDeviceInfo(devices[i], CL_DEVICE_VERSION, (void **)&char_ptr);
		printf("\tVERSION: %s\n", char_ptr);

		ret = oclGetDeviceInfo(devices[i], CL_DEVICE_MAX_COMPUTE_UNITS, (void **)&uint_ptr);
		printf("\tMAX_COMPUTE_UNITS: %u\n", uint_ptr[0]);

		ret = oclGetDeviceInfo(devices[i], CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, (void **)&uint_ptr);
		printf("\tMAX_WORK_ITEM_DIMENSIONS: %u\n", uint_ptr[0]);
		
		ret = oclGetDeviceInfo(devices[i], CL_DEVICE_MAX_WORK_ITEM_SIZES, (void **)&size_ptr);
		printf("\tMAX_WORK_ITEM_SIZES: %lu, %lu, %lu\n", size_ptr[0], size_ptr[1], size_ptr[2]);

		ret = oclGetDeviceInfo(devices[i], CL_DEVICE_MAX_WORK_GROUP_SIZE, (void **)&size_ptr);
		printf("\tMAX_WORK_GROUP_SIZE: %lu\n", size_ptr[0]);

		ret = oclGetDeviceInfo(devices[i], CL_DEVICE_GLOBAL_MEM_SIZE, (void **)&ulong_ptr);
		printf("\tGLOBAL_MEM_SIZE: %lu\n", ulong_ptr[0]);

		ret = oclGetDeviceInfo(devices[i], CL_DEVICE_LOCAL_MEM_SIZE, (void **)&ulong_ptr);
		printf("\tLOCAL_MEM_SIZE: %lu\n", ulong_ptr[0]);

	}

	
	if(argc > 2) {
		int sel = atoi(argv[1]);
		int kernel_num = atoi(argv[2]);
		int kernel_ver = atoi(argv[3]);
		char kernel_name[100], kernel_file[100];
		sprintf(kernel_name, "test%d_%d", kernel_num, kernel_ver);
#ifdef ALTERA
		sprintf(kernel_file, "test/%s.aocx", kernel_name);
#else
		sprintf(kernel_file, "test/%s.cl", kernel_name);
#endif
		printf("Running %s with device #%d\n", kernel_file, sel);
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
		size_t gsz = GSZ;
		size_t lsz = GSZ;
		size_t in_data_sz = sizeof(T) * GSZ;
		size_t out_data_sz = sizeof(T) * GSZ;
		T *in_data0, *out_data0;
		cl_mem in_data0_mem, out_data0_mem;
		printf("Data Unit Size: %u\n", sizeof(T));
		printf("Global Size: %u\n", gsz);
		printf("Local Size: %u\n", lsz);
		printf("Input Data Size: %u\n", (unsigned)in_data_sz);
		printf("Output Data Size: %u\n", (unsigned)out_data_sz);
		
#ifdef ALTERA		
		posix_memalign ((void **)&in_data0, AOCL_ALIGNMENT, in_data_sz);	
		posix_memalign ((void **)&out_data0, AOCL_ALIGNMENT, out_data_sz);	
#else
		in_data0 = (T *)malloc(in_data_sz);
		out_data0 = (T *)malloc(out_data_sz);		
#endif
		in_data0_mem = clCreateBuffer(context, CL_MEM_BANK_1_ALTERA | CL_MEM_READ_ONLY, in_data_sz, NULL, &ret0);
		out_data0_mem = clCreateBuffer(context, CL_MEM_BANK_1_ALTERA | CL_MEM_WRITE_ONLY, out_data_sz, NULL, &ret1);
		if(ret0 != CL_SUCCESS | ret1 != CL_SUCCESS) {
			printf("ERROR in clCreateBuffer(): %s, %s\n", oclReturnCodeToString(ret0), oclReturnCodeToString(ret1));
			return -1;
		}

		/*
			Set Kernel Arguments
		*/
		ret0 = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&in_data0_mem);
		ret1 = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&out_data0_mem);
		if(ret0 != CL_SUCCESS | ret1 != CL_SUCCESS | ret != CL_SUCCESS) {
			printf("ERROR in clSetKernelArg(): %s, %s, %s\n", 
				oclReturnCodeToString(ret0), oclReturnCodeToString(ret1), oclReturnCodeToString(ret));
			return -1;
		}


		long times[ITER];
		int errors[ITER];
		srand(time(NULL));
		for(i = 0; i < ITER; i++) {
		
			/*
				Enqueue Write Buffer
			*/
			for(j = 0; j < gsz; j++) {
				in_data0[j] = (T) rand();
			}
			ret0 = clEnqueueWriteBuffer(queue, in_data0_mem, CL_TRUE, 0, in_data_sz, in_data0, 0, NULL, NULL);
			if(ret0 != CL_SUCCESS) {
				printf("ERROR in clEnqueueWriteBuffer(): %s, %s\n", oclReturnCodeToString(ret0));
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
			if(ret0 != CL_SUCCESS) {
				printf("ERROR in clEnqueueReadBuffer(): %s, %s\n", oclReturnCodeToString(ret0));
				return -1;
			}

			errors[i] = 0;
			for(j = 0; j < GSZ / 2; j++) {
				errors[i] += (in_data0[j] != out_data0[j]);
			}
			
			times[i] = end - start;
		}

		long tot_time = 0;
		int tot_error = 0;
		for(i = 0; i < ITER; i++) {
			tot_time += times[i];
			tot_error += errors[i];
			//printf("%d\t%d\t%d\n", i, times[i], errors[i]);
		}
		printf("AVG\t%d\t%d\n", tot_time / ITER, tot_error / ITER);

		//long avg, std;
		//getStat(times, &avg, &std);
		//printf("AVG\t%d\t%d\n", avg, std);
	}

	printf("======= END test.c =======\n");
	
	return 0;

}

void getStat(long *times, long *avg, long *std) {
	int i;	
	long tot = 0;

	
	for(i = 0; i < ITER; i++) {
		tot += times[i];
	}
	*avg = tot / ITER;

	long sqtot = 0;
	for(i = 0; i < ITER; i++) {
		sqtot += (times[i] - *avg) * (times[i] - *avg);
	}
	*std = sqrt(sqtot / ITER);
		
}


