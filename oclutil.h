//===================================
//	Include Statements
//===================================
#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>

//===================================
//	Public Functions
//===================================


cl_int oclGetDevices(cl_uint *num_devices_, cl_device_id **devices_);
cl_int oclGetDeviceInfo(cl_device_id device, cl_device_info device_info, void **info);
cl_int oclKernelSetup(cl_device_id device, char *kernel_file, char *kernel_name, 
	cl_context *context, cl_command_queue *queue, cl_kernel *kernel);
double oclGetExecTime(cl_event *event);
char *oclDeviceInfo(cl_device_id device);
const char *oclReturnCodeToString(cl_int code);
const char *oclDeviceTypeToString(cl_device_type type);
char *oclReadSrc(char *filename, size_t *src_size);
unsigned char *oclReadBinary(char *filename, size_t *bin_size);





