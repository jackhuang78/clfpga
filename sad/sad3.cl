#include <sad.h>


__kernel 
__attribute((reqd_work_group_size(WG_S, WG_S, 1)))
void sad3(__global T *image, __constant T *filter, __global T *out, __local T *temp,
		  const unsigned int image_s, const unsigned int filter_s, const unsigned int out_s, const unsigned int temp_s) {

	int i, j;

	unsigned int group_r = get_group_id(1) * WG_S;
	unsigned int group_c = get_group_id(0) * WG_S;
	unsigned int global_r = get_group_id(1) * WG_S + get_local_id(1);
	unsigned int global_c = get_group_id(0) * WG_S + get_local_id(0);
	unsigned int local_r = get_local_id(1);
	unsigned int local_c = get_local_id(0);
	unsigned int extra = filter_s - 1;




	
	TEMP(local_r, local_c) = IMAGE(global_r, global_c);
	if(local_r < extra) {
		TEMP(WG_S + local_r, local_c) = IMAGE(WG_S + global_r, global_c); 
		TEMP(local_c, WG_S + local_r) = IMAGE(group_r + local_c, group_c + WG_S + local_r);

	} 


	if((local_r + extra < temp_s) && (local_c + extra < temp_s)) {
		TEMP(local_r + extra, local_c + extra) = IMAGE(global_r + extra, global_c + extra);
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


