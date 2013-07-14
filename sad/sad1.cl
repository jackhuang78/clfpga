#include <sad.h>


__kernel 
__attribute((reqd_work_group_size(LOCAL_S, LOCAL_S, 1)))
void sad1(__global T *image, __constant T *filter, __global T *out) {

	int i, j;

	unsigned int global_r = get_group_id(1) * LOCAL_S + get_local_id(1);
	unsigned int global_c = get_group_id(0) * LOCAL_S + get_local_id(0);
	unsigned int local_r = get_local_id(1);
	unsigned int local_c = get_local_id(0);
	
	T sad = 0;	
	if(global_r < OUT_S && global_c < OUT_S) {
		for(i = 0; i < FILTER_S; i++)
			for(j = 0; j < FILTER_S; j++) 
				sad += ABS(FILTER(i, j) - IMAGE(global_r + i, global_c + j));


		OUT(global_r, global_c) = sad;
	}


}


