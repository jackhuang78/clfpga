#include "test/test.h"

__kernel
__attribute((reqd_work_group_size(GSZ, 1,1))) 
void test0(__global T *g_idata, __global T *g_odata, __local T *sdata) {
    unsigned int tid = get_local_id(0);
    unsigned int gid = get_global_id(0);

	sdata[tid] = g_idata[gid];
	barrier(CLK_LOCAL_MEM_FENCE);
	g_odata[gid] = sdata[tid];
}


