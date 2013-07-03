#include <sad.h>

__kernel 
__attribute((reqd_work_group_size(WG_S, WG_S, 1)))
void sad1(__global int *image, __constant int *filter, __global int *out, unsigned int image_s, unsigned int filter_s) {
//	unsigned int gidc = get_group_id(0);
//	unsigned int gidr = get_group_id(1);

	unsigned int out_s = image_s - filter_s + 1;
	

	unsigned int image_idx = IDX(get_group_id(1), get_group_id(0), image_s) * WG_S + IDX(get_local_id(1), get_local_id(0), image_s);
	unsigned int out_idx = IDX(get_group_id(1), get_group_id(0), out_s) * WG_S + IDX(get_local_id(1), get_local_id(0), out_s);
	
	int ii, jj;
	int sad = 0;			
	for(ii = 0; ii < filter_s; ii++)
		for(jj = 0; jj < filter_s; jj++)
			sad += ABS(filter[IDX(ii, jj, filter_s)] - image[image_idx + IDX(ii, jj, image_s)]);

	//if(get_local_id(0) == 0 && get_local_id(1) == 0) {
	//	out[out_begin] = image[filter_begin];
	//	out[out_begin+1] = image[filter_begin +1];
	//}

	//out[out_begin + get_local_id(1) * out_s + get_local_id(0)] = image[image_begin];
	//out[out_begin + get_local_id(1) * out_s + get_local_id(0)] = sad;
	out[out_idx] = sad;

}


