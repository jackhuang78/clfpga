#include "test/test.h"

__kernel
__attribute((reqd_work_group_size(GSZ, 1,1))) 
void test2(__global T *g_idata0, 
		   __global T *g_idata1,
		   __global T *g_odata0,
		   __global T *g_odata1,
		   __local T *sdata0,
		   __local T *sdata1) {

    unsigned int gid = get_global_id(0);
    unsigned int tid = get_local_id(0);

	sdata0[tid] = g_idata0[gid];
	sdata1[tid] = g_idata1[gid];
    barrier(CLK_LOCAL_MEM_FENCE);
	g_odata0[gid] = sdata0[tid];
	g_odata1[gid] = sdata1[tid];
}


