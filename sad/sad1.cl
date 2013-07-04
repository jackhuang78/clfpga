#include <sad.h>


__kernel 
__attribute((reqd_work_group_size(WG_S, WG_S, 1)))
void sad1(__global int *image, __constant int *filter, __global int *out) {

	unsigned int global_r = get_group_id(1) * WG_S + get_local_id(1);
	unsigned int global_c = get_group_id(0) * WG_S + get_local_id(0);
	
	int ii, jj;
	int sad = 0;			
	for(ii = 0; ii < FILTER_S; ii++)
		for(jj = 0; jj < FILTER_S; jj++)
			sad += ABS(FILTER(ii, jj) - IMAGE(global_r + ii, global_c + jj));

	OUT(global_r, global_c) = sad;

}


