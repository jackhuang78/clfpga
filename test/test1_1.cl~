#include "test/test.h"

__kernel 
__attribute((reqd_work_group_size(GSZ, 1,1))) 
void test1(__global T *g_idata0, __global T *g_idata1, __global T *g_odata0, __global T *g_odata1, __local T* sdata) {
    unsigned int tid = get_local_id(0);
    unsigned int gid = get_global_id(0) / 2;
	unsigned int half = gid < (GSZ / 2);

    sdata[tid] = half ? g_idata0[gid] : g_idata1[gid];
    barrier(CLK_LOCAL_MEM_FENCE);
	if(half)
	    g_odata0[gid] = sdata[tid];
	else
		g_odata1[gid] = sdata[tid];

	barrier(CLK_LOCAL_MEM_FENCE);

    sdata[tid] = half ? g_idata0[gid] : g_idata1[gid];
    barrier(CLK_LOCAL_MEM_FENCE);
	if(half)
	    g_odata0[gid] = sdata[tid];
	else
		g_odata1[gid] = sdata[tid];
}
