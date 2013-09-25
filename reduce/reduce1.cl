#define T float
#define blockSize 128
#define nIsPow2 1

/* This version uses contiguous threads, but its interleaved 
   addressing results in many shared memory bank conflicts. */
__kernel 
__attribute((reqd_work_group_size(blockSize,1,1)))
//__attribute ((num_simd_work_items(2)))
__attribute ((num_compute_units(4)))
__attribute ((max_share_resources(8)))
void reduce1(__global T *g_idata, __global T *g_odata, unsigned int n, __local T* sdata)
{
    // load shared mem
    unsigned int tid = get_local_id(0);
    unsigned int i = get_global_id(0);
    
    sdata[tid] = (i < n) ? g_idata[i] : 0;
    
    barrier(CLK_LOCAL_MEM_FENCE);

    // do reduction in shared mem
	#pragma unroll
    for(unsigned int s=1; s < get_local_size(0); s *= 2) 
    {
        int index = 2 * s * tid;

        if (index < get_local_size(0)) 
        {
            sdata[index] += sdata[index + s];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // write result for this block to global mem
    if (tid == 0) g_odata[get_group_id(0)] = sdata[0];
}
