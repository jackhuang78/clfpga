#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>


//#include <libc.h>
#include <stdbool.h>
//#include <mach/mach_time.h>
 
#define OFFLINE 
#define FILENAME "soad_8x8.cl"
#define KERNELNAME "soad"
#define LINUX
#define EPSILON (1e-4f)*2

#ifdef ONLINE
  #include <OpenCL/opencl.h>
#else
  // ACL specific includes
  #include "CL/opencl.h"
#endif

// The image size is configured as 1000x1000
static const size_t imageWidth = 1000;
static const size_t imageHeight = 1000;
static const size_t imageSize = imageWidth * imageHeight;

// Buffer size configured in the kernel
static const size_t bufferHeight = 100;
static const size_t bufferWidth = 100;

// Template/Filter size in the kernel
static const size_t templateWidth = 8;
static const size_t templateHeight = 8;
static const size_t templateSize = templateWidth * templateHeight; 

static const size_t workDim = 2;
size_t globalSize[2] = {imageHeight, imageWidth};
size_t localSize[2] = {bufferHeight, bufferWidth};

// ACL runtime configuration
static cl_platform_id platform;
static cl_device_id device;
static cl_context context;
static cl_command_queue queue;
static cl_kernel kernel;
static cl_program program;
static cl_int status;

static cl_mem kernelImage, kernelTemplate, kernelOutput;

// input and output vectors
static void *Image, *templateImage, *output, *outputCheck;




// Need to align host data to 64 bytes to be able to use DMA
// LINUX/WINDOWS macros are defined in Makefiles.
#define ACL_ALIGNMENT 128

#if defined(LINUX)

  void* acl_aligned_malloc (size_t size) {
    void *result = NULL;
    posix_memalign (&result, ACL_ALIGNMENT, size);
    return result;
  }

  void acl_aligned_free (void *ptr) {
    free (ptr);
  }

#elif defined(MACOSX)

  void* acl_aligned_malloc (size_t size) {
    return malloc(size);
  }

  void acl_aligned_free (void *ptr) {
    free (ptr);
  }

#else // WINDOWS

  void* acl_aligned_malloc (size_t size) {
    return _aligned_malloc (size, ACL_ALIGNMENT);
  }

  void acl_aligned_free (void *ptr) {
    _aligned_free (ptr);
  }

#endif // LINUX


const char* errToString(cl_int err){
  switch (err) {
    case CL_SUCCESS:                            return "Success!";
    case CL_DEVICE_NOT_FOUND:                   return "Device not found.";
    case CL_DEVICE_NOT_AVAILABLE:               return "Device not available";
    case CL_COMPILER_NOT_AVAILABLE:             return "Compiler not available";
    case CL_MEM_OBJECT_ALLOCATION_FAILURE:      return "Memory object allocation failure";
    case CL_OUT_OF_RESOURCES:                   return "Out of resources";
    case CL_OUT_OF_HOST_MEMORY:                 return "Out of host memory";
    case CL_PROFILING_INFO_NOT_AVAILABLE:       return "Profiling information not available";
    case CL_MEM_COPY_OVERLAP:                   return "Memory copy overlap";
    case CL_IMAGE_FORMAT_MISMATCH:              return "Image format mismatch";
    case CL_IMAGE_FORMAT_NOT_SUPPORTED:         return "Image format not supported";
    case CL_BUILD_PROGRAM_FAILURE:              return "Program build failure";
    case CL_MAP_FAILURE:                        return "Map failure";
    case CL_INVALID_VALUE:                      return "Invalid value";
    case CL_INVALID_DEVICE_TYPE:                return "Invalid device type";
    case CL_INVALID_PLATFORM:                   return "Invalid platform";
    case CL_INVALID_DEVICE:                     return "Invalid device";
    case CL_INVALID_CONTEXT:                    return "Invalid context";
    case CL_INVALID_QUEUE_PROPERTIES:           return "Invalid queue properties";
    case CL_INVALID_COMMAND_QUEUE:              return "Invalid command queue";
    case CL_INVALID_HOST_PTR:                   return "Invalid host pointer";
    case CL_INVALID_MEM_OBJECT:                 return "Invalid memory object";
    case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:    return "Invalid image format descriptor";
    case CL_INVALID_IMAGE_SIZE:                 return "Invalid image size";
    case CL_INVALID_SAMPLER:                    return "Invalid sampler";
    case CL_INVALID_BINARY:                     return "Invalid binary";
    case CL_INVALID_BUILD_OPTIONS:              return "Invalid build options";
    case CL_INVALID_PROGRAM:                    return "Invalid program";
    case CL_INVALID_PROGRAM_EXECUTABLE:         return "Invalid program executable";
    case CL_INVALID_KERNEL_NAME:                return "Invalid kernel name";
    case CL_INVALID_KERNEL_DEFINITION:          return "Invalid kernel definition";
    case CL_INVALID_KERNEL:                     return "Invalid kernel";
    case CL_INVALID_ARG_INDEX:                  return "Invalid argument index";
    case CL_INVALID_ARG_VALUE:                  return "Invalid argument value";
    case CL_INVALID_ARG_SIZE:                   return "Invalid argument size";
    case CL_INVALID_KERNEL_ARGS:                return "Invalid kernel arguments";
    case CL_INVALID_WORK_DIMENSION:             return "Invalid work dimension";
    case CL_INVALID_WORK_GROUP_SIZE:            return "Invalid work group size";
    case CL_INVALID_WORK_ITEM_SIZE:             return "Invalid work item size";
    case CL_INVALID_GLOBAL_OFFSET:              return "Invalid global offset";
    case CL_INVALID_EVENT_WAIT_LIST:            return "Invalid event wait list";
    case CL_INVALID_EVENT:                      return "Invalid event";
    case CL_INVALID_OPERATION:                  return "Invalid operation";
    case CL_INVALID_GL_OBJECT:                  return "Invalid OpenGL object";
    case CL_INVALID_BUFFER_SIZE:                return "Invalid buffer size";
    case CL_INVALID_MIP_LEVEL:                  return "Invalid mip-map level";
    default: return "Unknown";
  }
}


static void dump_error(const char *str, cl_int status) {
  printf("%s\n", str);
  printf("Error code: %d => %s\n", status, errToString(status));
}

static void initializeVector(float* vector, int size) {
  for (int i = 0; i < size; ++i) {
    vector[i] = rand() / (float)RAND_MAX;
  }
}

static void initializeGradiantImage(cl_short* img, int imH, int imW, cl_short oVal){
  for (cl_short i=0; i<imH; i++){
    for(cl_short j=0; j<imW; j++){
        img[i*imW+j] = abs(i+j+oVal);
    }
  }
}

static void initializeNullImage(cl_short* img, int imH, int imW, cl_short oVal){
  for (cl_short i=0; i<imH; i++){
    for(cl_short j=0; j<imW; j++){
      img[i*imW+j] = 0;
    }
  }
}

// free the resources allocated during initialization
static void freeResources() {
  if(kernel)
    clReleaseKernel(kernel);
  if(program)
    clReleaseProgram(program);
  if(queue)
    clReleaseCommandQueue(queue);
  if(context)
    clReleaseContext(context);
  if(kernelImage)
    clReleaseMemObject(kernelImage);
  if(kernelTemplate)
    clReleaseMemObject(kernelTemplate);
  if(kernelOutput)
    clReleaseMemObject(kernelOutput);
  if(Image)
    acl_aligned_free(Image);
  if(templateImage)
    acl_aligned_free(templateImage);
  if(output)
    acl_aligned_free(output);
  if(outputCheck)
    acl_aligned_free(outputCheck);
}




static char* load_program_source(const char *filename)
{
    struct stat statbuf;
    FILE        *fh;
    char        *source;
 
    fh = fopen(filename, "r");
    if (fh == 0)
        return 0;
 
    stat(filename, &statbuf);
    source = (char *) malloc(statbuf.st_size + 1);
    fread(source, statbuf.st_size, 1, fh);
    source[statbuf.st_size] = '\0';
 
    return source;
}

void computeSAD(cl_short* image, int imgH, int imgW,  cl_short* temp, int tempH, int tempW, cl_int* output){

  for(int i=0; i<imgH; i++){
    for(int j=0; j<imgW; j++){
      output[i*imgW+j] = 0;
      for(int m=0; m<tempH; m++){
        for(int n=0; n<tempW; n++){
          int x;
          if(j+n < imgW && i+m < imgH) x=image[(i+m)*imgW+j+n] - temp[m*tempW + n];
          else x = temp[m*tempW + n];
          output[i*imgW+j] += (x<0)?-x:x; //abs(image[(i+m)*imgW+j+n] - temp[m*tempW + n]);
        }
      }
    }
  }
}

int main() {

  cl_uint num_platforms;
  cl_uint num_devices;

  // allocate and initialize the input vectors
  Image = (void *)acl_aligned_malloc(sizeof(cl_short) * imageHeight * imageWidth);
  templateImage = (void *)acl_aligned_malloc(sizeof(cl_short) * templateHeight * templateWidth);
  outputCheck = (void *)acl_aligned_malloc(sizeof(cl_int) * imageHeight * imageWidth);
  output = (void *)acl_aligned_malloc(sizeof(cl_int) * imageHeight * imageWidth);
  

  // get the platform ID
  status = clGetPlatformIDs(1, &platform, &num_platforms);
  if(status != CL_SUCCESS) {
    dump_error("Failed clGetPlatformIDs.", status);
    freeResources();
    return 1;
  }
  if(num_platforms != 1) {
    printf("Found %d platforms!\n", num_platforms);
    freeResources();
    return 1;
  }

  // get the device ID
  status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, &num_devices);
  if(status != CL_SUCCESS) {
    dump_error("Failed clGetDeviceIDs.", status);
    freeResources();
    return 1;
  }
  if(num_devices != 1) {
    printf("Found %d devices!\n", num_devices);
    freeResources();
    return 1;
  }

  // create a context
  context = clCreateContext(0, 1, &device, NULL, NULL, &status);
  if(status != CL_SUCCESS) {
    dump_error("Failed clCreateContext.", status);
    freeResources();
    return 1;
  }

  // create a command queue
  queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &status);
  if(status != CL_SUCCESS) {
    dump_error("Failed clCreateCommandQueue.", status);
    freeResources();
    return 1;
  }

  // create the input buffer
  kernelImage = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(cl_short) * imageHeight * imageWidth , NULL, &status);
  if(status != CL_SUCCESS) {
    dump_error("Failed clCreateBuffer.", status);
    freeResources();
    return 1;
  }

  // create the input buffer
  kernelTemplate = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(cl_short) * templateHeight * templateWidth, NULL, &status);
  if(status != CL_SUCCESS) {
    dump_error("Failed clCreateBuffer.", status);
    freeResources();
    return 1;
  }

  // create the output buffer
  kernelOutput = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(cl_int) * imageHeight * imageWidth, NULL, &status);
  if(status != CL_SUCCESS) {
    dump_error("Failed clCreateBuffer.", status);
    freeResources();
    return 1;
  }

  // create the kernel
  const char *kernel_name = KERNELNAME;
  cl_int kernel_status;


#ifdef OFFLINE

  FILE* fp = fopen(FILENAME, "rb");
  if (fp == NULL) {
    printf("Failed to open %s file (fopen).\n", FILENAME);
        return -1;
  }
  fseek(fp, 0, SEEK_END);
  size_t file_length = ftell(fp);
  unsigned char*file_content = (unsigned char*) malloc(sizeof(unsigned char) * file_length);
  assert(file_content && "Malloc failed");
  rewind(fp);
  if (fread((void*)file_content, file_length, 1, fp) == 0) {
    printf("Failed to read from %s file (fread).\n", FILENAME);
        return -1;
  }
  fclose(fp);



  program = clCreateProgramWithBinary(context, 1, &device, &file_length, (const unsigned char**)&file_content, &kernel_status, &status);
  if(status != CL_SUCCESS || kernel_status != CL_SUCCESS) {
    dump_error("Failed clCreateProgramWithBinary.", status);
    freeResources();
    return 1;
  }

#else
  char*file_content = load_program_source(FILENAME);

  // Create the compute program from the source buffer
  //
  program = clCreateProgramWithSource(context, 1, (const char **) &file_content, NULL, &status);

  if(status != CL_SUCCESS ) {
    dump_error("Failed clCreateProgramWithBinary.", status);
    freeResources();
    return 1;
  } 

#endif


  // build the program
  status = clBuildProgram(program, 0, NULL, "", NULL, NULL);
  if(status != CL_SUCCESS) {
    dump_error("Failed clBuildProgram.", status);
    freeResources();
    return 1;
  }

  // create the kernel
  kernel = clCreateKernel(program, kernel_name, &status);
  if(status != CL_SUCCESS) {
    dump_error("Failed clCreateKernel.", status);
    freeResources();
    return 1;
  }

  // set the arguments
  status = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&kernelImage);
  if(status != CL_SUCCESS) {
    dump_error("Failed set arg 0.", status);
    return 1;
  }

  // set the arguments
  status = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&kernelTemplate);
  if(status != CL_SUCCESS) {
    dump_error("Failed set arg 1.", status);
    return 1;
  }

  status = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&kernelOutput);
  if(status != CL_SUCCESS) {
    dump_error("Failed Set arg 2.", status);
    freeResources();
    return 1;
  }

  printf("Kernel initialization is complete.\n");
  printf("Starting iterative launching...\n\n");
  printf("| Run |  Write   |  Exec.   |   Read   |      Passed      |\n");

  double totalWriteTime = 0.0, totalReadTime = 0.0, totalExecTime = 0.0, totalAbsError = 0.0;
  bool totalpass =true;
  

  for(int count=0; count<21; count++){

    cl_event dataWriteEvent1;
    cl_event dataWriteEvent2;
    cl_ulong startWrite, endWrite;

    initializeGradiantImage((short*)Image, imageHeight, imageWidth, 0);
    initializeNullImage((short*)templateImage, templateHeight, templateWidth, 0);

    computeSAD((cl_short*)Image, imageHeight, imageWidth, (cl_short*)templateImage, templateHeight, templateWidth, (cl_int*)outputCheck);

    // Write the vector argument
    status = clEnqueueWriteBuffer(queue, kernelImage, CL_TRUE, 0, sizeof(cl_short) * imageHeight* imageWidth, Image, 0, NULL, &dataWriteEvent1);
    if(status != CL_SUCCESS) {
      dump_error("Failed to enqueue buffer kernelImage.", status);
      freeResources();
      return 1;
    }
    // Write the vector argument
    status = clEnqueueWriteBuffer(queue, kernelTemplate, CL_TRUE, 0, sizeof(cl_short) * templateHeight * templateWidth, templateImage, 0, NULL, &dataWriteEvent2);
    if(status != CL_SUCCESS) {
      dump_error("Failed to enqueue buffer kernelTemplate.", status);
      freeResources();
      return 1;
    }

    // Get data write time
    clGetEventProfilingInfo(dataWriteEvent1, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &endWrite, NULL);
    clGetEventProfilingInfo(dataWriteEvent1, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &startWrite, NULL);
    double writeTime= (double)1.0e-9 * (endWrite - startWrite);
    clGetEventProfilingInfo(dataWriteEvent2, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &endWrite, NULL);
    clGetEventProfilingInfo(dataWriteEvent2, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &startWrite, NULL);
    writeTime+= (double)1.0e-9 * (endWrite - startWrite);

    if(count > 0) totalWriteTime+=writeTime;
    printf("|  %2d | %7f |", count, writeTime);

    // launch kernel
    clFinish(queue);
    cl_event execEvent;
    cl_ulong startExec, endExec;


    status = clEnqueueNDRangeKernel(queue, kernel, workDim, NULL, globalSize, localSize, 0, NULL, &execEvent);
    clWaitForEvents(1, &execEvent);
    if (status != CL_SUCCESS) {
      dump_error("Failed to launch kernel.", status);
      freeResources();
      return 1;
    }    

    // Get execution time
    clGetEventProfilingInfo(execEvent, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &endExec, NULL);
    clGetEventProfilingInfo(execEvent, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &startExec, NULL);
    double execTime= (double)1.0e-9 * (endExec - startExec);
    if(count > 0) totalExecTime+=execTime;
    printf(" %7f |", execTime);

    cl_event dataReadEvent;
    cl_ulong startRead, endRead;

    // read the output
    status = clEnqueueReadBuffer(queue, kernelOutput, CL_TRUE, 0, sizeof(cl_int) * imageHeight * imageWidth, output, 0, NULL, &dataReadEvent);
    clWaitForEvents(1, &dataReadEvent);
    if(status != CL_SUCCESS) {
      dump_error("Failed to enqueue buffer kernelOutput.", status);
      freeResources();
      return 1;
    }

    // Get data write time
    clGetEventProfilingInfo(dataReadEvent, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &endRead, NULL);
    clGetEventProfilingInfo(dataReadEvent, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &startRead, NULL);
    double readTime= (double)1.0e-9 * (endRead - startRead);
    if(count > 0) totalReadTime+= readTime;
    printf(" %7f |", readTime);


    // verify the output
    bool pass = true;
    int posX, posY;

     for (int i=0; i<imageHeight && pass; i++){
        for(int j=0; j<imageWidth && pass; j++) {
          if(((cl_int*)outputCheck)[i*imageWidth + j] != ((cl_int*)output)[i*imageWidth + j]) {
                  pass = false; posX = j; posY = i;
                  totalpass =false;
                  printf("\n%d != %d  @ %d,%d\n ", ((cl_int*)outputCheck)[i*imageWidth+j], ((cl_int*)output)[i*imageWidth+j],i,j);
                  break;
            }
         }
     }

    if(pass) printf("       Ok         |\n");
    else printf(" %d !! %d |\n", posX, posY);
    if(count < 1) printf("-------------------------------------------------------------------\n");  

    // printf("\n\nOutput:\n");
    // for (int i=0; i<imageHeight && pass; i++){
    //    for(int j=0; j<imageWidth && pass; j++) {
    //        printf("%10d,",((cl_int*)output)[i*imageWidth+j]);
    //     }
    //     printf("\n");
    // }

  }
  printf("-------------------------------------------------------------------\n");
  printf("| Avg | %7f | %7f | %7f |       %s         |\n", totalWriteTime/20, totalExecTime/20, totalReadTime/20, (totalpass)?"Ok":"!!");
  printf("-------------------------------------------------------------------\n");
  // free the resources allocated
  freeResources();

  return 0;
}
