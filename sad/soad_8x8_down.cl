

#define WINDOW_BUFFER_HEIGHT (100)
#define WINDOW_BUFFER_WIDTH (100)
#define TEMPLATE_WIDTH  (8)
#define TEMPLATE_HEIGHT (8)


/*
        SumOfDiff kernel with buffering of image and template into local memory
*/

__kernel
// The image area this kernel computes within a workgroup
__attribute__((reqd_work_group_size(WINDOW_BUFFER_HEIGHT, WINDOW_BUFFER_WIDTH, 1)))
void soad(
        __global unsigned short* image,
        __global unsigned short* template,
        __global unsigned short* output) {

        // The maximum possible image window and template we support in this kernel
		__local unsigned short windowBuffer[WINDOW_BUFFER_HEIGHT + TEMPLATE_HEIGHT][WINDOW_BUFFER_WIDTH + TEMPLATE_WIDTH];
        __local unsigned short templateBuffer[TEMPLATE_HEIGHT][TEMPLATE_WIDTH];

//        __local unsigned short windowBuffer[WINDOW_BUFFER_HEIGHT + TEMPLATE_HEIGHT][WINDOW_BUFFER_WIDTH + TEMPLATE_WIDTH];
  //      __local unsigned short templateBuffer[TEMPLATE_HEIGHT][TEMPLATE_WIDTH];

        unsigned int gHeight = get_global_id(0);
        unsigned int gWidth = get_global_id(1);
        unsigned int lHeight = get_local_id(0);
        unsigned int lWidth = get_local_id(1);
        unsigned int imageHeight = get_global_size(0);
        unsigned int imageWidth = get_global_size(1);

        unsigned int loc = gHeight * get_global_size(1) + gWidth;


                int sumAbsDiff = 0l;

                        // Buffer the image in the window buffer
                        for (unsigned int i=0; i<TEMPLATE_HEIGHT; i++){
                                for(unsigned int j=0; j<TEMPLATE_WIDTH; j++){
                                        sumAbsDiff += abs(windowBuffer[i+lHeight][j+lWidth] - templateBuffer[i][j]);
                                }
                        }
                        output[loc] = sumAbsDiff;


}
