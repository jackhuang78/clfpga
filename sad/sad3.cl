#include <sad.h>


__kernel 
__attribute((reqd_work_group_size(LOCAL_S, LOCAL_S, 1)))
void sad(__global T * image, __constant T * filter, __global T * out) {

	int i, j;
	__local T image_temp[TEMP_S * TEMP_S];
	__local T filter_temp[FILTER_S * FILTER_S];


	unsigned int group_r = get_group_id(1) * LOCAL_S;
	unsigned int group_c = get_group_id(0) * LOCAL_S;
	unsigned int global_r = get_group_id(1) * LOCAL_S + get_local_id(1);
	unsigned int global_c = get_group_id(0) * LOCAL_S + get_local_id(0);
	unsigned int local_r = get_local_id(1);
	unsigned int local_c = get_local_id(0);
	unsigned int extra_s = FILTER_S - 1;


	// copy filters to __local memory
	if(local_r < FILTER_S && local_c < FILTER_S)
		FILTER_TEMP(local_r, local_c) = FILTER(local_r, local_c);
			
	// copy the portion of image to be used to __local memory
	IMAGE_TEMP(local_r, local_c) = IMAGE(global_r, global_c);
	if(local_r < extra_s) {
		IMAGE_TEMP(LOCAL_S + local_r, local_c) = IMAGE(LOCAL_S + global_r, global_c); 
		IMAGE_TEMP(local_c, LOCAL_S + local_r) = IMAGE(group_r + local_c, LOCAL_S + group_c + local_r);

	} 
	if((local_r < TEMP_S - extra_s) && (local_c < TEMP_S - extra_s)) {
		IMAGE_TEMP(local_r + extra_s, local_c + extra_s) = IMAGE(global_r + extra_s, global_c + extra_s);
	}
	barrier(CLK_LOCAL_MEM_FENCE);


	// computer SAD at for each point
	if(global_r < OUT_S && global_c < OUT_S) {
		T sad = 0;	
		for(i = 0; i < FILTER_S; i++)
			for(j = 0; j < FILTER_S; j++) 
				sad += ABS(FILTER_TEMP(i, j) - IMAGE_TEMP(local_r + i, local_c + j));

		OUT(global_r, global_c) = sad;
	}
}


