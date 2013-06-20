#include <CL/cl.h>
#include "oclutil.h"
#include "host_template.h"

#ifdef ALTERA
#define KERNEL_EXT ".aocx"
#else
#define KERNEL_EXT ".cl"
#endif

/*
	I wrote several helper functions in oclutil.c to make the setup easier.

	Also remember to add the new soruce codes to the makefile of choice, which
	targets different CL implementations. Current I have AMD, NVIDIA, and 
	ALTERA.

	Using this file and ALTERA device as an example, I added 
	"host_template.h" and "host_template.o" to the rule and create another 
	executable called "host_template":

		_DEPS = oclutil.h host_template.h
		_OBJS = oclutil.o host_template.o
	
		all: $(ODIR)/reduce $(ODIR)/host_template makefile.altera

		$(ODIR)/reduce: $(OBJS)
			gcc -o $@ $^ -DLINUX -DALTERA $(CLLIB)

	To make the executable for, use the command:
		make -f makefile.altera clean all

	Then run the executable with:
		./altera/host_template
	
*/
int main(int argc, char **argv) {
	cl_uint num_devices;
	cl_device_id *devices;
	cl_context context;
	cl_command_queue queue;
	cl_kernel kernel;
	cl_event event;
	float time;

	// set kernel file name and kenrel name
	char *kernel_file = ...;
	char *kennel_name ...;

	// get all available devices
	oclCLDevices(&num_devices, &devices);

	// select a device, and 
	// create context, command queue, and kernel with the specified kernel file and name
	oclQuickSetup(device[0], kernel_file, kernel_name, &context, &queue, &kernel);

	// code to create memory buffer and set kernel arguments
	clCreateBuffer(context, ...);
	clSetKernelArg(kernel, ...);

	// fill input memory buffer
	clEnqueueWriteBuffer(queue, ...);

	// execute and profile kernel
	clFinish(queue);
	clEnqueueNDRangeKernel(queue, kernel, ..., &event);
	clWaitForEvents(1, &event);
	time = oclExecutionTime(&event);

	// read output from memory buffer
	clEnqueueReadBuffer(queue, ...);	
}
