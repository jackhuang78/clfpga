#define T float
#define blockSize 64
#define nIsPow2 1

/*
    This version uses n/2 threads --
    it performs the first level of reduction when reading from global memory
*/
__kernel 
__attribute((reqd_work_group_size(blockSize,1,1)))
__attribute ((num_simd_work_items(4)))
//__attribute ((num_compute_units(1)))
__attribute ((max_share_resources(8)))
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
	if(tid == 0){
        // write result for this block to global mem
        g_odata[get_group_id(0)] =  sdata[0] + sdata[1] + sdata[2] + sdata[3] + sdata[4] + sdata[5] + sdata[6] + sdata[7] + sdata[8] + sdata[9] + sdata[10] + sdata[11] + sdata[12] + sdata[13] + sdata[14] + sdata[15] + sdata[16] + sdata[17] + sdata[18] + sdata[19] + sdata[20] + sdata[21] + sdata[22] + sdata[23] + sdata[24] + sdata[25] + sdata[26] + sdata[27] + sdata[28] + sdata[29] + sdata[30] + sdata[31] + sdata[32] + sdata[33] + sdata[34] + sdata[35] + sdata[36] + sdata[37] + sdata[38] + sdata[39] + sdata[40] + sdata[41] + sdata[42] + sdata[43] + sdata[44] + sdata[45] + sdata[46] + sdata[47] + sdata[48] + sdata[49] + sdata[50] + sdata[51] + sdata[52] + sdata[53] + sdata[54] + sdata[55] + sdata[56] + sdata[57] + sdata[58] + sdata[59] + sdata[60] + sdata[61] + sdata[62] + sdata[63];
    }
}
