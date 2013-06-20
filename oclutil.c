#include <stdio.h>
#include <time.h>
#include <CL/cl.h>
#include "oclutil.h"
#include <string.h>

#define BUFFER_SIZE 1000
static char buffer[BUFFER_SIZE];

#define CASE(expr) case (expr): return #expr;
#define CASEDEF default: return "UNDEFINED";
char *device_type_str(cl_device_type type) {
	switch(type) {
	case CL_DEVICE_TYPE_CPU: return "CL_DEVICE_TYPE_CPU";
	case CL_DEVICE_TYPE_GPU: return "CL_DEVICE_TYPE_GPU";
	case CL_DEVICE_TYPE_ACCELERATOR: return "CL_DEVICE_TYPE_ACCELERATOR";
	case CL_DEVICE_TYPE_DEFAULT: return "CL_DEVICE_TYPE_DEFAULT";
	default: return "UNDEFINED";
	}
}

const char* code_to_str(cl_int error)
{
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

    const int errorCount = sizeof(errorString) / sizeof(errorString[0]);

    const int index = -error;

    return (index >= 0 && index < errorCount) ? errorString[index] : "Unspecified Error";
}

void oclCLDevices(cl_uint *num_devices_, cl_device_id **devices_) {
	int i, j, k;
	cl_int ret;
	cl_uint num_platforms;
	cl_platform_id *platforms;
	cl_uint num_devices, tot_num_devices;
	cl_device_id *devices, *tot_devices;	

	

	

	// Initially, set NUM_DEIVCES to 0.
	tot_num_devices = 0;	
	
	// Get number of platforms.
	CHECKRET(ret, clGetPlatformIDs(0, NULL, &num_platforms))	
	if(num_platforms == 0) {
		return;
	}

	// Get the platofrm IDs.
	platforms = (cl_platform_id *)malloc(sizeof(cl_platform_id) * num_platforms);
	CHECKRET(ret, clGetPlatformIDs(num_platforms, platforms, NULL))

	// Iterate through the platforms, count the number of devices.
	for(i = 0; i < num_platforms; i++) {
		CHECKRET(ret, clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices))
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
		CHECKRET(ret, clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices))
		CHECKRET(ret, clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, num_devices, devices, NULL))
		

		for(j = 0; j < num_devices; j++, k++) {
			tot_devices[k] = devices[j];
			CHECKRET(ret, clGetDeviceInfo(devices[j], CL_DEVICE_NAME, BUFFER_SIZE, buffer, NULL))
			//printf("oclCLDevices() %d[%x]: %s\n", k, (unsigned)devices[j], buffer);
		}
	}

	

}

void oclDeviceInfo(cl_device_id device, char **name, cl_device_type *type,
				   cl_ulong *local_mem_sz) {
	cl_int ret;


	if(name != NULL) {
		CHECKRET(ret, clGetDeviceInfo(device, CL_DEVICE_NAME, BUFFER_SIZE, buffer, NULL))
		*name = strcpy((char *)malloc(strlen(buffer) + 1), buffer);
	}

	if(type != NULL) {
		CHECKRET(ret, clGetDeviceInfo(device, CL_DEVICE_TYPE, sizeof(type), type, NULL))		
	}

	if(local_mem_sz != NULL) {
		CHECKRET(ret, clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(local_mem_sz), local_mem_sz, NULL))
	}
}

void oclPrintDeviceInfo(cl_device_id device, char *prefix) {
	char *name;
	cl_device_type type;
	cl_ulong local_mem_sz;

	oclDeviceInfo(device, &name, &type, &local_mem_sz);

	printf("%sName: %s\n", prefix, name);
	printf("%sType: %s\n", prefix, device_type_str(type));
	printf("%sLocal Memory Size: %u\n", prefix, local_mem_sz);
	
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

cl_int oclQuickSetup(cl_device_id device, char *kernel_file, char *kernel_name, 
	cl_context *context, cl_command_queue *queue, cl_kernel *kernel) {

	cl_int ret;
	cl_program program;
	char *source_str;
	size_t source_sz;

	//oclPrintDevInfo(device);

	// Create context and queue
	CHECK(*context = clCreateContext(NULL, 1, &device, NULL, NULL, &ret))
	CHECK(*queue = clCreateCommandQueue(*context, device, CL_QUEUE_PROFILING_ENABLE, &ret))

	// create program and kernel
	if(DEBUG) {
		printf("oclQuickSetup(): kernel_name: %s\n", kernel_name);
		printf("oclQuickSetup(): kernel_file: %s\n", kernel_file);
	}

	// create from source
	if(kernel_file[strlen(kernel_file) - 1] == 'l') {
		printf("Create from source\n");
		source_str = oclReadSrc(kernel_file, &source_sz);
		CHECK(program = clCreateProgramWithSource(*context, 1, (const char **)&source_str, (const size_t *)&source_sz, &ret))
		clBuildProgram(program, 1, &device, NULL, NULL, NULL);
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
	
	CHECK(*kernel = clCreateKernel(program, kernel_name, &ret))

	return 0;
}






//=============================================================================

char *oclDeviceName(cl_device_id device) {
	cl_int ret;
	CHECKRET(ret, clGetDeviceInfo(device, CL_DEVICE_NAME, BUFFER_SIZE, buffer, NULL))
	return strcpy((char *)malloc(strlen(buffer) + 1), buffer);	
}

cl_device_type oclDeviceType(cl_device_id device) {
	cl_int ret;
	cl_device_type type;
	CHECKRET(ret, clGetDeviceInfo(device, CL_DEVICE_TYPE, sizeof(type), &type, NULL))
	return type;
}







void oclDisplay(void) {
	int i, j;

	cl_int ret;
	cl_uint num_platforms;
	cl_platform_id *platforms;


	// Find out how many platforms.
	CHECKRET(ret, clGetPlatformIDs(0, NULL, &num_platforms))	
	if(num_platforms == 0) {
		printf("No platform found.\n");
		return;
	}	
	
	// Get the platofrm IDs.
	platforms = (cl_platform_id *)malloc(sizeof(cl_platform_id) * num_platforms);
	CHECKRET(ret, clGetPlatformIDs(num_platforms, platforms, NULL))

	// Print out platform info.
	for(i = 0; i < num_platforms; i++) {
		printf("Platform #%d:\n", i);

		CHECKRET(ret, clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, BUFFER_SIZE, buffer, NULL))
		printf("\tName: %s\n", buffer);
		CHECKRET(ret, clGetPlatformInfo(platforms[i], CL_PLATFORM_VENDOR, BUFFER_SIZE, buffer, NULL))
		printf("\tVendor: %s\n", buffer);
		CHECKRET(ret, clGetPlatformInfo(platforms[i], CL_PLATFORM_VERSION, BUFFER_SIZE, buffer, NULL))
		printf("\tVersion: %s\n", buffer);


		cl_uint num_devices;
		cl_device_id *devices;
		cl_device_type device_type;
		cl_ulong ulong;
		cl_uint uint;
		size_t sizes[3];


		// Find out how many devices.
		CHECKRET(ret, clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices))
		
		// Get the device IDs.
		devices = (cl_device_id *)malloc(sizeof(cl_device_id) * num_devices);
		CHECKRET(ret, clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, num_devices, devices, NULL))
		
		for(j = 0; j < num_devices; j++) {
			printf("\tDevice #%d:\n", j);
			
			CHECKRET(ret, clGetDeviceInfo(devices[j], CL_DEVICE_NAME, BUFFER_SIZE, buffer, NULL))
			printf("\t\tName: %s\n", buffer);

			CHECKRET(ret, clGetDeviceInfo(devices[j], CL_DEVICE_VERSION, BUFFER_SIZE, buffer, NULL))
			printf("\t\tVersion: %s\n", buffer);

			CHECKRET(ret, clGetDeviceInfo(devices[j], CL_DEVICE_TYPE, sizeof(device_type), &device_type, NULL))
			printf("\t\tType: %s\n", device_type_str(device_type));

			CHECKRET(ret, clGetDeviceInfo(devices[j], CL_DEVICE_LOCAL_MEM_SIZE, sizeof(ulong), &ulong, NULL))
			printf("\t\tLocal Memroy (byte): %lu\n", (unsigned long)ulong);

			CHECKRET(ret, clGetDeviceInfo(devices[j], CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(uint), &uint, NULL))
			printf("\t\tClock Frequency (MHz): %u\n", (unsigned int)uint);

			CHECKRET(ret, clGetDeviceInfo(devices[j], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(uint), &uint, NULL))
			printf("\t\tCompute Units: %u\n", (unsigned int)uint);

			CHECKRET(ret, clGetDeviceInfo(devices[j], CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t)*3, sizes, NULL))
			printf("\t\tMax Work Group Size: %u\n", (unsigned int)sizes[0]);

			CHECKRET(ret, clGetDeviceInfo(devices[j], CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(uint), &uint, NULL))
			printf("\t\tMax Work Item Dimensions: %u\n", (unsigned int)uint);

			CHECKRET(ret, clGetDeviceInfo(devices[j], CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(size_t)*3, sizes, NULL))
			printf("\t\tMax Work Item Sizes: (%u, %u, %u)\n", (unsigned int)sizes[0], (unsigned int)sizes[1], (unsigned int)sizes[2]);
			

		}
	} // END retrieving platforms and devices
}

void oclPrintDevInfo(cl_device_id device) {
		cl_int ret;
		//cl_uint num_devices;
		//cl_device_id *devices;
		cl_device_type device_type;
		cl_ulong ulong;
		cl_uint uint;
		size_t sizes[3];
			printf("\tDevice Info:\n");
			
			CHECKRET(ret, clGetDeviceInfo(device, CL_DEVICE_NAME, BUFFER_SIZE, buffer, NULL))
			printf("\t\tName: %s\n", buffer);

			CHECKRET(ret, clGetDeviceInfo(device, CL_DEVICE_VERSION, BUFFER_SIZE, buffer, NULL))
			printf("\t\tVersion: %s\n", buffer);

			CHECKRET(ret, clGetDeviceInfo(device, CL_DEVICE_TYPE, sizeof(device_type), &device_type, NULL))
			printf("\t\tType: %s\n", device_type_str(device_type));

			CHECKRET(ret, clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(ulong), &ulong, NULL))
			printf("\t\tLocal Memroy (byte): %lu\n", (unsigned long)ulong);

			CHECKRET(ret, clGetDeviceInfo(device, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(uint), &uint, NULL))
			printf("\t\tClock Frequency (MHz): %u\n", (unsigned int)uint);

			CHECKRET(ret, clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(uint), &uint, NULL))
			printf("\t\tCompute Units: %u\n", (unsigned int)uint);

			CHECKRET(ret, clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t)*3, sizes, NULL))
			printf("\t\tMax Work Group Size: %u\n", (unsigned int)sizes[0]);

			CHECKRET(ret, clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(uint), &uint, NULL))
			printf("\t\tMax Work Item Dimensions: %u\n", (unsigned int)uint);

			CHECKRET(ret, clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(size_t)*3, sizes, NULL))
			printf("\t\tMax Work Item Sizes: (%u, %u, %u)\n", (unsigned int)sizes[0], (unsigned int)sizes[1], (unsigned int)sizes[2]);
}

cl_device_id oclGetDevice(int platform_idx, int device_idx) {
	cl_uint num_platforms, num_devices;
	cl_platform_id *platforms;
	cl_device_id *devices;
	cl_int ret;

	// Get the platofrm IDs.
	CHECKRET(ret, clGetPlatformIDs(0, NULL, &num_platforms))	
	if(num_platforms == 0) {
		printf("No platform found.\n");
		ret = CL_INVALID_VALUE;
		return NULL;

	} else if(platform_idx >= num_platforms) {
		printf("Platform of index %d does not exist.\n", platform_idx);
		return NULL;
	} 
	platforms = (cl_platform_id *)malloc(sizeof(cl_platform_id) * num_platforms);
	CHECKRET(ret, clGetPlatformIDs(num_platforms, platforms, NULL))


	// Get the device IDs.
	CHECKRET(ret, clGetDeviceIDs(platforms[platform_idx], CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices))
	if(device_idx >= num_devices) {
		printf("Device of index %d does not exist.\n", device_idx);
		return NULL;
	}	
	devices = (cl_device_id *)malloc(sizeof(cl_device_id) * num_devices);
	CHECKRET(ret, clGetDeviceIDs(platforms[platform_idx], CL_DEVICE_TYPE_ALL, num_devices, devices, NULL))

	return devices[device_idx];
	
}





int seeded = 0;
float rand_float(void) {
	if(!seeded) {
		srand((unsigned)time(0));
		seeded = 1;
	}
	return (float)rand() / (float)RAND_MAX;

/*
	printf("rand_float\n");
	float *data = (float *)malloc(n * sizeof(float));
	printf("rand_float2\n");

	int i;
	for(i = 0; i < n; i++)
		data[i] = (float)rand() / (float)RAND_MAX;
	
	return data;*/
}

double oclExecutionTime(cl_event *event) {
    cl_ulong start, end;
    
    clGetEventProfilingInfo(*event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
    clGetEventProfilingInfo(*event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
    
    return (double)1.0e-9 * (end - start); // convert nanoseconds to seconds on return
}



