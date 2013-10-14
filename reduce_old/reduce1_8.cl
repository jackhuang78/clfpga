#define blockSize 128
#define nIsPow2 1

/* This version uses contiguous threads, but its interleaved 
   addressing results in many shared memory bank conflicts. */
__kernel 
__attribute((reqd_work_group_size(blockSize,1,1)))
void reduce1_8(__global float8 *g_idata, __global float *g_odata, unsigned int n, __local float8* sdata)
{
    // load shared mem
    unsigned int tid = get_local_id(0);
    unsigned int i = get_global_id(0);
    
    sdata[tid] = (i < n) ? g_idata[i] : 0;
    
    barrier(CLK_LOCAL_MEM_FENCE);

    // do reduction in shared mem
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
    if (tid == 0) {
    	g_odata[get_group_id(0)] = 
            sdata[0].s0 + sdata[0].s1 + sdata[0].s2 + sdata[0].s3 +
            sdata[0].s4 + sdata[0].s5 + sdata[0].s6 + sdata[0].s7;

	}
}
