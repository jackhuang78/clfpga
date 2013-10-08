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

	
	if(argc > 5) {
		int sel = atoi(argv[1]);
		char *kernel_file = argv[2];
		char *kernel_name = argv[3];
		int data_size = atoi(argv[4]) << 20;
		int iter = atoi(argv[5]);
		int check = atoi(argv[6]);
		printf("Running Test on device %d:\n", sel);
		printf("\tKernel file: %s\n", kernel_file);
		printf("\tKernel: %s\n", kernel_name);
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
		size_t gsz = data_size;
		size_t lsz = GSZ;
		size_t in_data_sz = sizeof(T) * data_size;
		size_t out_data_sz = sizeof(T) * data_size;
		T *in_data0, *in_data1, *out_data0, *out_data1;
		cl_mem in_data0_mem, in_data1_mem, out_data0_mem, out_data1_mem;
		printf("\tData Unit Size: %lu\n", sizeof(T));
		printf("\tGlobal Size: %lu\n", gsz);
		printf("\tLocal Size: %lu\n", lsz);
		printf("\tInput Memory Size: %u (x2)\n", (unsigned)in_data_sz);
		printf("\tOutput Memory Size: %u (x2)\n", (unsigned)out_data_sz);
		
#ifdef ALTERA		
		posix_memalign ((void **)&in_data0, AOCL_ALIGNMENT, in_data_sz);
		posix_memalign ((void **)&in_data1, AOCL_ALIGNMENT, in_data_sz);		
		posix_memalign ((void **)&out_data0, AOCL_ALIGNMENT, out_data_sz);	
		posix_memalign ((void **)&out_data1, AOCL_ALIGNMENT, out_data_sz);	
#else
		in_data0 = (T *)malloc(in_data_sz);
		in_data1 = (T *)malloc(in_data_sz);
		out_data0 = (T *)malloc(out_data_sz);		
		out_data1 = (T *)malloc(out_data_sz);		
#endif
		in_data0_mem = clCreateBuffer(context, CL_MEM_BANK_1_ALTERA | CL_MEM_READ_ONLY, in_data_sz, NULL, &ret0);
		in_data1_mem = clCreateBuffer(context, CL_MEM_BANK_2_ALTERA | CL_MEM_READ_ONLY, in_data_sz, NULL, &ret1);
		out_data0_mem = clCreateBuffer(context, CL_MEM_BANK_1_ALTERA | CL_MEM_WRITE_ONLY, out_data_sz, NULL, &ret2);
		out_data1_mem = clCreateBuffer(context, CL_MEM_BANK_2_ALTERA | CL_MEM_WRITE_ONLY, out_data_sz, NULL, &ret3);
		if(ret0 != CL_SUCCESS | ret1 != CL_SUCCESS | ret2 != CL_SUCCESS | ret3 != CL_SUCCESS) {
			printf("ERROR in clCreateBuffer(): %s, %s, %s, %s\n", oclReturnCodeToString(ret0), oclReturnCodeToString(ret1), oclReturnCodeToString(ret2), oclReturnCodeToString(ret3));
			return -1;
		}

		/*
			Set Kernel Arguments
		*/
		ret0 = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&in_data0_mem);
		ret1 = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&in_data1_mem);
		ret2 = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&out_data0_mem);
		ret3 = clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *)&out_data1_mem);
		if(ret0 != CL_SUCCESS | ret1 != CL_SUCCESS | ret2 != CL_SUCCESS | ret3 != CL_SUCCESS) {
			printf("ERROR in clSetKernelArg() for global mem: %s, %s, %s, %s\n", oclReturnCodeToString(ret0), oclReturnCodeToString(ret1), oclReturnCodeToString(ret2), oclReturnCodeToString(ret3));
			return -1;
		}
		ret0 = clSetKernelArg(kernel, 4, sizeof(T) * lsz, NULL);
		ret1 = clSetKernelArg(kernel, 5, sizeof(T) * lsz, NULL);
		if(ret0 != CL_SUCCESS | ret1 != CL_SUCCESS) {
			printf("ERROR in clSetKernelArg() for local mem: %s, %s\n", oclReturnCodeToString(ret0), oclReturnCodeToString(ret1));
			return -1;
		}

		long times[iter];
		int errors[iter];
		srand(time(NULL));
		for(i = 0; i < iter; i++) {
		
			/*
				Enqueue Write Buffer
			*/
			for(j = 0; j < gsz; j++) {
				in_data0[j] = (T) rand();
				in_data1[j] = (T) rand();
				out_data0[j] = (T) rand();
				out_data1[j] = (T) rand();
			}
			ret0 = clEnqueueWriteBuffer(queue, in_data0_mem, CL_TRUE, 0, in_data_sz, in_data0, 0, NULL, NULL);
			ret1 = clEnqueueWriteBuffer(queue, in_data1_mem, CL_TRUE, 0, in_data_sz, in_data1, 0, NULL, NULL);
			if(ret0 != CL_SUCCESS | ret1 != CL_SUCCESS) {
				printf("ERROR in clEnqueueWriteBuffer(): %s, %s\n", oclReturnCodeToString(ret0), oclReturnCodeToString(ret1));
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
	}

	printf("======= END test.c =======\n");
	
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



