/* This version uses contiguous threads, but its interleaved 
   addressing results in many shared memory bank conflicts. */
__kernel 
__attribute((reqd_work_group_size(128,1,1)))
void reduce1_16(__global float16 *g_idata, __global float *g_odata, unsigned int n, __local float16* sdata)
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
		sdata[0].s01234567 += sdata[0].s89abcdef;
		sdata[0].s0123 += sdata[0].s4567;
		sdata[0].s01 += sdata[0].s23;
		g_odata[get_group_id(0)] = sdata[0].s0 + sdata[0].s1;
	}
}
