#define T float
#define blockSize 128
#define nIsPow2 1

/*
    This version is completely unrolled.  It uses a template parameter to achieve 
    optimal code for any (power of 2) number of threads.  This requires a switch 
    statement in the host code to handle all the different thread block sizes at 
    compile time.
*/
__kernel 
__attribute((reqd_work_group_size(blockSize,1,1)))
void reduce5(__global T *g_idata, __global T *g_odata, unsigned int n, __local volatile T* sdata)
{
    // perform first level of reduction,
    // reading from global memory, writing to shared memory
    unsigned int tid = get_local_id(0);
    unsigned int i = get_group_id(0)*(get_local_size(0)*2) + get_local_id(0);

    sdata[tid] = (i < n) ? g_idata[i] : 0;
    if (i + blockSize < n) 
        sdata[tid] += g_idata[i+blockSize];  

    barrier(CLK_LOCAL_MEM_FENCE);

    // do reduction in shared mem
    if (blockSize >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } barrier(CLK_LOCAL_MEM_FENCE); }
    if (blockSize >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } barrier(CLK_LOCAL_MEM_FENCE); }
    if (blockSize >= 128) { if (tid <  64) { sdata[tid] += sdata[tid +  64]; } barrier(CLK_LOCAL_MEM_FENCE); }
    
    if (tid < 32)
    {
        if (blockSize >=  64) { sdata[tid] += sdata[tid + 32]; }
        if (blockSize >=  32) { sdata[tid] += sdata[tid + 16]; }
        if (blockSize >=  16) { sdata[tid] += sdata[tid +  8]; }
        if (blockSize >=   8) { sdata[tid] += sdata[tid +  4]; }
        if (blockSize >=   4) { sdata[tid] += sdata[tid +  2]; }
        if (blockSize >=   2) { sdata[tid] += sdata[tid +  1]; }
    }
    
    // write result for this block to global mem 
    if (tid == 0) g_odata[get_group_id(0)] = sdata[0];
}