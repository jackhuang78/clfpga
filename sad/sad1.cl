#include <sad.h>

__kernel 
__attribute((reqd_work_group_size(WG_S, WG_S, 1)))
void sad1(__global int *image, __constant int *filter, __global int *out, __local int *temp,
		  const unsigned int image_s, const unsigned int filter_s, const unsigned int out_s, const unsigned int temp_s) {


	unsigned int global_r = get_group_id(1) * WG_S + get_local_id(1);
	unsigned int global_c = get_group_id(0) * WG_S + get_local_id(0);
	unsigned int local_r = get_local_id(1);
	unsigned int local_c = get_local_id(0);
	
	int i, j;
	int sad = 0;	
	
	for(i = 0; i < filter_s; i++)
		for(j = 0; j < filter_s; j++) {
			sad += ABS(FILTER(i, j) - IMAGE(global_r + i, global_c + j));
//			sad += IMAGE(global_r + i, global_c + j);
			
		}

	OUT(global_r, global_c) = sad;


}


