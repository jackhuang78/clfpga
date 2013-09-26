//===================================
//	Include Statements
//===================================
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <CL/cl.h>
#include "oclutil.h"

//===================================
//	Macro Definitions
//===================================
#define BUFFER_SIZE 1000

//===================================
//	Private Variables & Functions
//===================================
static char buffer[BUFFER_SIZE];
#define STRING(str) strcpy((char *)malloc(strlen(str) + 1), (str))

//===================================
//	Function Definitions
//===================================

/*
	Get a list of available OpenCL devices.
*/
cl_int oclGetDevices(cl_uint *num_devices_, cl_device_id **devices_) {
	int i, j, k;
	cl_int ret;
	cl_uint num_platforms;
	cl_platform_id *platforms;
	cl_uint num_devices, tot_num_devices;
	cl_device_id *devices, *tot_devices;	

	// Initially, set NUM_DEIVCES to 0.
	tot_num_devices = 0;	
	
	// Get number of platforms. Return if none found
	ret = clGetPlatformIDs(0, NULL, &num_platforms);
	if(num_platforms == 0) 
		return CL_SUCCESS;

	// Get the platofrm IDs.
	platforms = (cl_platform_id *)malloc(sizeof(cl_platform_id) * num_platforms);
	ret = clGetPlatformIDs(num_platforms, platforms, NULL);
	if(ret != CL_SUCCESS) {
		return ret;
	}

	// Iterate through the platforms, count the number of devices.
	for(i = 0; i < num_platforms; i++) {
		ret = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices);
		if(ret != CL_SUCCESS) {
			return ret;
		}
		tot_num_devices += num_devices;	
	}
	*num_devices_ = tot_num_devices;


	// Allocate for DEVICES
	devices = (cl_device_id *)malloc(sizeof(cl_device_id) * num_devices);
	tot_devices = (cl_device_id *)malloc(sizeof(cl_device_id) * num_devices);
	*devices_ = tot_devices;


	// Iterate through the platforms again, this time getting the device IDs.
	for(i = 0, k = 0; i < num_platforms; i++) {

		// Get the number of devices and  device IDs.
		ret = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices);
		ret = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, num_devices, devices, NULL);
		if(ret != CL_SUCCESS) {
			return ret;
		}
		for(j = 0; j < num_devices; j++, k++) 
			tot_devices[k] = devices[j];
	
	}

	return CL_SUCCESS;
}

cl_int oclGetDeviceInfo(cl_device_id device, cl_device_info device_info, void **info) {
	cl_int ret;

	ret = clGetDeviceInfo(device, device_info, BUFFER_SIZE, buffer, NULL);
	if(ret != CL_SUCCESS) 
		return ret;

	*info = buffer;
	return CL_SUCCESS;

}



/*
	Initialize context, command queue, and kernel given a OpenCL device,
	the kenrel file name, and the kernel name.
*/
cl_int oclKernelSetup(cl_device_id device, char *kernel_file, char *kernel_name, 
	cl_context *context, cl_command_queue *queue, cl_kernel *kernel) {

	cl_int ret;
	cl_program program;
	char *source_str;
	size_t source_sz;

	//oclPrintDevInfo(device);

	// Create context
	*context = clCreateContext(NULL, 1, &device, NULL, NULL, &ret);
	if(ret != CL_SUCCESS)
		return ret;

	// Create command queue
	*queue = clCreateCommandQueue(*context, device, CL_QUEUE_PROFILING_ENABLE, &ret);
	if(ret != CL_SUCCESS)
		return ret;

	// create program and kernel
	/*if(DEBUG) {
		printf("oclQuickSetup(): kernel_name: %s\n", kernel_name);
		printf("oclQuickSetup(): kernel_file: %s\n", kernel_file);
	}*/

	// create from source
	if(kernel_file[strlen(kernel_file) - 1] == 'l') {
		printf("Create from source\n");
		source_str = oclReadSrc(kernel_file, &source_sz);
		program = clCreateProgramWithSource(*context, 1, (const char **)&source_str, (const size_t *)&source_sz, &ret);
		if(ret != CL_SUCCESS)
			return ret;
		clBuildProgram(program, 1, &device, "-I ./", NULL, NULL);
		clGetProgramBuildInfo (program, device, CL_PROGRAM_BUILD_LOG, BUFFER_SIZE, buffer, NULL);
		if(strlen(buffer) != 0)
			printf("Build log:\n%s\n", buffer);

	// create from binary
	} else {
		printf("Create from binary\n");
		unsigned char *binary = oclReadBinary(kernel_file, &source_sz);
		cl_int status;
		program = clCreateProgramWithBinary(*context, 1, &device, &source_sz, (const unsigned char **)&binary, &status, &ret);
		if(status != CL_SUCCESS || ret != CL_SUCCESS) {
			printf("Failed to create the program from the binary (clCreateProgramWithBinary).\n");
			return -1;
		}
	}
	
	*kernel = clCreateKernel(program, kernel_name, &ret);
	if(ret != CL_SUCCESS)
		return ret;

	return CL_SUCCESS;
}

char *oclReadSrc(char *filename, size_t *src_size) {
	FILE *fp = fopen(filename, "rb");
	if(!fp) {
		printf("Failed to open %s.\n", filename);
		return NULL;
	}
	fseek(fp, 0L, SEEK_END);
	*src_size = ftell(fp);
	fclose(fp);
	fp = fopen(filename, "r");

	char *src_str = (char *)malloc(*src_size);
	*src_size = fread(src_str, 1, *src_size, fp);

	fclose(fp);
	return src_str;
	
}

unsigned char *oclReadBinary(char *filename, size_t *bin_size) {

	FILE *fp = fopen(filename, "rb");
	if (!fp) {
		printf("Failed to open %s.\n", filename);
		return NULL;
	}
	fseek(fp, 0, SEEK_END);
	size_t binary_length = ftell(fp);
	unsigned char *binary = (unsigned char*) malloc(sizeof(unsigned char) * binary_length);
	rewind(fp);
	if (fread((void*)binary, binary_length, 1, fp) == 0) {
		printf("Failed to read from %s file (fread).\n", filename);
		return NULL;
	}
	fclose(fp);
	
	*bin_size = binary_length;
	return binary;
}

const char *oclReturnCodeToString(cl_int error) {
    static const char* errorString[] = {
        "CL_SUCCESS",
        "CL_DEVICE_NOT_FOUND",
        "CL_DEVICE_NOT_AVAILABLE",
        "CL_COMPILER_NOT_AVAILABLE",
        "CL_MEM_OBJECT_ALLOCATION_FAILURE",
        "CL_OUT_OF_RESOURCES",
        "CL_OUT_OF_HOST_MEMORY",
        "CL_PROFILING_INFO_NOT_AVAILABLE",
        "CL_MEM_COPY_OVERLAP",
        "CL_IMAGE_FORMAT_MISMATCH",
        "CL_IMAGE_FORMAT_NOT_SUPPORTED",
        "CL_BUILD_PROGRAM_FAILURE",
        "CL_MAP_FAILURE",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "CL_INVALID_VALUE",
        "CL_INVALID_DEVICE_TYPE",
        "CL_INVALID_PLATFORM",
        "CL_INVALID_DEVICE",
        "CL_INVALID_CONTEXT",
        "CL_INVALID_QUEUE_PROPERTIES",
        "CL_INVALID_COMMAND_QUEUE",
        "CL_INVALID_HOST_PTR",
        "CL_INVALID_MEM_OBJECT",
        "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR",
        "CL_INVALID_IMAGE_SIZE",
        "CL_INVALID_SAMPLER",
        "CL_INVALID_BINARY",
        "CL_INVALID_BUILD_OPTIONS",
        "CL_INVALID_PROGRAM",
        "CL_INVALID_PROGRAM_EXECUTABLE",
        "CL_INVALID_KERNEL_NAME",
        "CL_INVALID_KERNEL_DEFINITION",
        "CL_INVALID_KERNEL",
        "CL_INVALID_ARG_INDEX",
        "CL_INVALID_ARG_VALUE",
        "CL_INVALID_ARG_SIZE",
        "CL_INVALID_KERNEL_ARGS",
        "CL_INVALID_WORK_DIMENSION",
        "CL_INVALID_WORK_GROUP_SIZE",
        "CL_INVALID_WORK_ITEM_SIZE",
        "CL_INVALID_GLOBAL_OFFSET",
        "CL_INVALID_EVENT_WAIT_LIST",
        "CL_INVALID_EVENT",
        "CL_INVALID_OPERATION",
        "CL_INVALID_GL_OBJECT",
        "CL_INVALID_BUFFER_SIZE",
        "CL_INVALID_MIP_LEVEL",
        "CL_INVALID_GLOBAL_WORK_SIZE",
    };

    int errorCount = sizeof(errorString) / sizeof(errorString[0]);
    int index = -error;

    return (index >= 0 && index < errorCount) ? errorString[index] : "Unspecified Error";
}

const char *oclDeviceTypeToString(cl_device_type type) {
	switch(type) {
	case CL_DEVICE_TYPE_CPU: return "CL_DEVICE_TYPE_CPU";
	case CL_DEVICE_TYPE_GPU: return "CL_DEVICE_TYPE_GPU";
	case CL_DEVICE_TYPE_ACCELERATOR: return "CL_DEVICE_TYPE_ACCELERATOR";
	case CL_DEVICE_TYPE_DEFAULT: return "CL_DEVICE_TYPE_DEFAULT";
	default: return "UNDEFINED";
	}
}


int main(int argc, char **argv) {
	printf("======== oclutil.c ========\n");

	int i;
	char *char_ptr;
	size_t *size_ptr;
	cl_device_type *device_type_ptr;
	cl_uint *uint_ptr;
	cl_ulong *ulong_ptr;

	cl_int ret;
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
		printf("\tMAX_WORK_ITEM_SIZES: %u, %u, %u\n", size_ptr[0], size_ptr[1], size_ptr[2]);

		ret = oclGetDeviceInfo(devices[i], CL_DEVICE_MAX_WORK_GROUP_SIZE, (void **)&size_ptr);
		printf("\tMAX_WORK_GROUP_SIZE: %u\n", size_ptr[0]);

		ret = oclGetDeviceInfo(devices[i], CL_DEVICE_GLOBAL_MEM_SIZE, (void **)&ulong_ptr);
		printf("\tGLOBAL_MEM_SIZE: %u\n", ulong_ptr[0]);

		ret = oclGetDeviceInfo(devices[i], CL_DEVICE_LOCAL_MEM_SIZE, (void **)&ulong_ptr);
		printf("\tLOCAL_MEM_SIZE: %u\n", ulong_ptr[0]);

	}

	return 0;

}


