#include "test/test.h"

__kernel
__attribute((reqd_work_group_size(GSZ, 1,1))) 
void test0_0(__global T *g_idata, __global T *g_odata) {
    unsigned int tid = get_local_id(0);
    unsigned int gid = get_global_id(0);

	g_odata[gid] = g_idata[gid];
}


