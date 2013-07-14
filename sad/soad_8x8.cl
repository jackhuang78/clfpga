

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
//__attribute__ ((num_simd_work_items(2)))
//__attribute__ ((num_compute_units(2)))
//__attribute__ ((max_share_resources(4)))
void soad(
        __global unsigned short* image,
        __global unsigned short* template,
        //const unsigned int imageHeight,
        //const unsigned int imageWidth,
        __global unsigned short* output) {

/*
        if(get_local_id(0) < WINDOW_BUFFER_HEIGHT && get_local_id(1) < WINDOW_BUFFER_WIDTH) {
                output[get_global_id(0)*get_global_size(1) + get_global_id(1)] = image[get_global_id(0)*get_global_size(1) + get_global_id(1)];
        }
*/
        // The maximum possible image window and template we support in this kernel
        __local unsigned short windowBuffer[WINDOW_BUFFER_HEIGHT + TEMPLATE_HEIGHT][WINDOW_BUFFER_WIDTH + TEMPLATE_WIDTH];
        __local unsigned short templateBuffer[TEMPLATE_HEIGHT][TEMPLATE_WIDTH];

        unsigned int gHeight = get_global_id(0);
        unsigned int gWidth = get_global_id(1);
        unsigned int lHeight = get_local_id(0);
        unsigned int lWidth = get_local_id(1);
        unsigned int imageHeight = get_global_size(0);
        unsigned int imageWidth = get_global_size(1);

        unsigned int loc = gHeight * get_global_size(1) + gWidth;

        bool region1 = (lWidth < (TEMPLATE_WIDTH));
        bool region2 = (lHeight < (TEMPLATE_HEIGHT));
        bool region3 = (region1 && region2);

        // Buffer subsection of the image into the window buffer
        if(gWidth < imageWidth && gHeight < imageHeight){
                windowBuffer[lHeight][lWidth] = image[loc];
        }else {
                windowBuffer[lHeight][lWidth] = 0;
        }

        if(region1){
                if((gWidth + WINDOW_BUFFER_WIDTH) < imageWidth && gHeight < imageHeight){
                        windowBuffer[lHeight][WINDOW_BUFFER_WIDTH + lWidth] = image[loc + WINDOW_BUFFER_WIDTH];
                }else{
                        windowBuffer[lHeight][WINDOW_BUFFER_WIDTH + lWidth] = 0;
                }
        }

        if(region2){
                if((gHeight + WINDOW_BUFFER_HEIGHT) < imageHeight && gWidth < imageWidth){
                        windowBuffer[lHeight + WINDOW_BUFFER_HEIGHT][lWidth] = image[loc + WINDOW_BUFFER_HEIGHT * imageWidth];
                }else{
                        windowBuffer[lHeight + WINDOW_BUFFER_HEIGHT][lWidth] = 0;
                }
        }
        
        if(region3){
                if((gHeight + WINDOW_BUFFER_HEIGHT) < imageHeight && (gWidth + WINDOW_BUFFER_WIDTH) < imageWidth){
                        windowBuffer[WINDOW_BUFFER_HEIGHT+lHeight][WINDOW_BUFFER_WIDTH+lWidth] = image[loc + WINDOW_BUFFER_HEIGHT*imageWidth + WINDOW_BUFFER_WIDTH];
                }else{
                        windowBuffer[WINDOW_BUFFER_HEIGHT+lHeight][WINDOW_BUFFER_WIDTH+lWidth] = 0;
                }

                // Buffer template
                if(lWidth < TEMPLATE_WIDTH && lHeight < TEMPLATE_HEIGHT){
                        templateBuffer[lHeight][lWidth] = template[lHeight*TEMPLATE_WIDTH + lWidth];
                }
        }

                barrier(CLK_LOCAL_MEM_FENCE);

                int sumAbsDiff = 0l;

                        // Buffer the image in the window buffer
                        #pragma unroll
                        for (unsigned int i=0; i<TEMPLATE_HEIGHT; i++){
                                #pragma unroll
                                for(unsigned int j=0; j<TEMPLATE_WIDTH; j++){
                                        sumAbsDiff += abs(windowBuffer[i+lHeight][j+lWidth] - templateBuffer[i][j]);
                                }
                        }
                        output[loc] = sumAbsDiff;

                        //output[loc] = windowBuffer[lHeight+TEMPLATE_HEIGHT-1][lWidth+TEMPLATE_WIDTH-1];
                        //output[loc] = windowBuffer[lHeight+0][lWidth+0];

}
