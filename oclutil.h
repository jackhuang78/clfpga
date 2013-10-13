//===================================
//	Include Statements
//===================================
#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>

#define CHECK_RC(ret, msg) \
	if(ret != CL_SUCCESS) {\
		printf("ERROR in %s: %s\n", msg, oclReturnCodeToString(ret));\
		exit(-1);\
	}

#define CHECK_BS(ret, msg, log) \
	if(ret != CL_BUILD_SUCCESS) {\
		printf("ERROR in %s: %s\n", msg, oclBuildStatusToString(ret));\
		printf("Build log: %s\n", log);\
		exit(-1);\
	}



//#define OCL_DEBUG

//===================================
//	Public Functions
//===================================
cl_int oclGetDevices(cl_uint *num_devices_, cl_device_id **devices_);
cl_int oclGetDeviceInfo(cl_device_id device, cl_device_info device_info, void **info);
cl_int oclKernelSetup(cl_device_id device, char *kernel_file, char *kernel_name, 
	cl_context *context, cl_command_queue *queue, cl_kernel *kernel,
	cl_build_status *build_status, char **build_log);
cl_int oclGetProfilingInfo(cl_event *event, cl_ulong *queued, cl_ulong *submit, cl_ulong *start, cl_ulong *end);
char *oclDeviceInfo(cl_device_id device);
const char *oclReturnCodeToString(cl_int code);
const char *oclDeviceTypeToString(cl_device_type type);
const char *oclBuildStatusToString(cl_build_status status);
cl_int oclPrintDeviceInfo(cl_device_id device, char *pre);
char *oclReadSrc(char *filename, size_t *src_size);
unsigned char *oclReadBinary(char *filename, size_t *bin_size);






