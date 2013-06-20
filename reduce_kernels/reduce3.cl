#define T float
#define blockSize 128
#define nIsPow2 1

/*
    This version uses n/2 threads --
    it performs the first level of reduction when reading from global memory
*/
__kernel 
__attribute((reqd_work_group_size(blockSize,1,1)))
void reduce3(__global T *g_idata, __global T *g_odata, unsigned int n, __local T* sdata)
{
    // perform first level of reduction,
    // reading from global memory, writing to shared memory
    unsigned int tid = get_local_id(0);
    unsigned int i = get_group_id(0)*(get_local_size(0)*2) + get_local_id(0);

    sdata[tid] = (i < n) ? g_idata[i] : 0;
    if (i + get_local_size(0) < n) 
        sdata[tid] += g_idata[i+get_local_size(0)];  

    barrier(CLK_LOCAL_MEM_FENCE);

    // do reduction in shared mem
    for(unsigned int s=get_local_size(0)/2; s>0; s>>=1) 
    {
        if (tid < s) 
        {
            sdata[tid] += sdata[tid + s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // write result for this block to global mem 
    if (tid == 0) g_odata[get_group_id(0)] = sdata[0];
}