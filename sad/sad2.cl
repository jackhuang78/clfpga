#include <sad.h>


__kernel 
__attribute((reqd_work_group_size(WG_S, WG_S, 1)))
void sad2(__global T *image, __constant T *filter, __global T *out, __local T *temp,
		  const unsigned int image_s, const unsigned int filter_s, const unsigned int out_s, const unsigned int temp_s) {

	int i, j;


	unsigned int global_r = get_group_id(1) * WG_S + get_local_id(1);
	unsigned int global_c = get_group_id(0) * WG_S + get_local_id(0);
	unsigned int local_r = get_local_id(1);
	unsigned int local_c = get_local_id(0);
	unsigned int bound = WG_S - 1;

	

	if(local_r < bound && local_c < bound) {
		TEMP(local_r, local_c) = IMAGE(global_r, global_c);

	} else if(local_r == bound && local_c == bound) {
		for(i = 0; i < filter_s; i++) 
			for(j = 0; j < filter_s; j++) 
				TEMP(local_r + i, local_c + j) = IMAGE(global_r + i, global_c + j);
		
	} else if(local_r == bound){
		for(i = 0; i < filter_s; i++) 
			TEMP(local_r + i, local_c) = IMAGE(global_r + i, global_c);
		
	} else {
		for(j = 0; j < filter_s; j++)
			TEMP(local_r, local_c + j) = IMAGE(global_r, global_c + j);
		
	}

	barrier(CLK_LOCAL_MEM_FENCE);


	T sad = 0;	
	for(i = 0; i < filter_s; i++)
		for(j = 0; j < filter_s; j++) {
			sad += ABS(FILTER(i, j) - TEMP(local_r + i, local_c + j));
			//sad += TEMP(local_r + i, local_c + j);
		}

	OUT(global_r, global_c) = sad;

}


