#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>

#define DEBUG 0

#define CHECKRET(ret, expr) {\
	ret = (expr);\
	if(ret != CL_SUCCESS) {\
		printf("Error %s: %s\n", code_to_str(ret), #expr);\
		exit(1);\
	}\
}

#define CHECK(expr) {\
	(expr);\
	if(ret != CL_SUCCESS) {\
		printf("Error %s: %s\n", code_to_str(ret), #expr);\
		exit(1);\
	}\
}

#define ROUNDUP(n, m) ((n)/(m) + (((n)%(m)) == 0 ? 0 : 1))


void oclCLDevices(cl_uint *num_devices_, cl_device_id **devices_);
void oclDeviceInfo(cl_device_id device, char **name, cl_device_type *type,
				   cl_uint *max_workitem_dim, size_t *max_workgroup_sz, size_t *max_workitem_sz,
				   cl_ulong *local_mem_sz);
void oclPrintDeviceInfo(cl_device_id device, char *prefix);
cl_int oclQuickSetup(cl_device_id device, char *kernel_file, char *kernel_name, 
	cl_context *context, cl_command_queue *queue, cl_kernel *kernel);
double oclExecutionTime(cl_event *event);
const char *code_to_str(cl_int code);
char *device_type_str(cl_device_type type);




char *oclDeviceName(cl_device_id device);
cl_device_type oclDeviceType(cl_device_id device);
float rand_float(void);
double rand_double(void);
int rand_int(void);
void oclDisplay(void);
char *oclReadSrc(char *filename, size_t *src_size);
cl_device_id oclGetDevice(int platform_idx, int device_idx);


