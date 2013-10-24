/*
    Parallel reduction kernels
*/

#ifndef _REDUCE_KERNEL_H_
#define _REDUCE_KERNEL_H_

#ifndef WORK_GROUP
#define WORK_GROUP (128)
#endif

#ifndef NUM_UNITS
#define NUM_UNITS (1)
#endif

#ifndef VECTOR_SIZE
#define VECTOR_SIZE (1)
#define T float
#endif

#ifndef T
#error "Type T is not defined"
#endif

/*
    Requirement: n must be an even number!
    Parallel sum reduction using shared memory
    - takes log(n) steps for n input elements
    - uses n threads
    - only works for power-of-2 arrays
    This version uses sequential addressing -- no divergence or bank conflicts.
*/


__kernel 
__attribute ((reqd_work_group_size(WORK_GROUP/VECTOR_SIZE,1,1)))
__attribute ((num_simd_work_items(16/VECTOR_SIZE)))
__attribute ((num_compute_units(NUM_UNITS)))
__attribute ((max_share_resources(8)))
void mreduce6(__global T *g_idata1, __global T *g_idata2, __global T *g_odata, unsigned int n)
{
    // load shared mem
    unsigned int local_index = get_local_id(0);
    unsigned int global_index = get_global_id(0);
    
    __local T sdata[WORK_GROUP];   
    
    if(global_index < n>>1) {
        sdata[local_index] = g_idata1[global_index] + g_idata2[global_index];
    }
    else{
        sdata[local_index] = 0;        
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    
    if(local_index == 0){
        // write result for this block to global mem
        g_odata[get_group_id(0)] =  
			((((((sdata[  0] + sdata[  1]) + (sdata[  2] + sdata[  3])) + ((sdata[  4] + sdata[  5]) + (sdata[  6] + sdata[  7])))    + 
			   (((sdata[  8] + sdata[  9]) + (sdata[ 10] + sdata[ 11])) + ((sdata[ 12] + sdata[ 13]) + (sdata[ 14] + sdata[ 15]))))   + 
			  ((((sdata[ 16] + sdata[ 17]) + (sdata[ 18] + sdata[ 19])) + ((sdata[ 20] + sdata[ 21]) + (sdata[ 22] + sdata[ 23])))    + 
			   (((sdata[ 24] + sdata[ 25]) + (sdata[ 26] + sdata[ 27])) + ((sdata[ 28] + sdata[ 29]) + (sdata[ 30] + sdata[ 31])))))  + 
			 (((((sdata[ 32] + sdata[ 33]) + (sdata[ 34] + sdata[ 35])) + ((sdata[ 36] + sdata[ 37]) + (sdata[ 38] + sdata[ 39])))    + 
			   (((sdata[ 40] + sdata[ 41]) + (sdata[ 42] + sdata[ 43])) + ((sdata[ 44] + sdata[ 45]) + (sdata[ 46] + sdata[ 47]))))   + 
			  ((((sdata[ 48] + sdata[ 49]) + (sdata[ 50] + sdata[ 51])) + ((sdata[ 52] + sdata[ 53]) + (sdata[ 54] + sdata[ 55])))    + 
			   (((sdata[ 56] + sdata[ 57]) + (sdata[ 58] + sdata[ 59])) + ((sdata[ 60] + sdata[ 61]) + (sdata[ 62] + sdata[ 63])))))) + 
			((((((sdata[ 64] + sdata[ 65]) + (sdata[ 66] + sdata[ 67])) + ((sdata[ 68] + sdata[ 69]) + (sdata[ 70] + sdata[ 71])))    + 
			   (((sdata[ 72] + sdata[ 73]) + (sdata[ 74] + sdata[ 75])) + ((sdata[ 76] + sdata[ 77]) + (sdata[ 78] + sdata[ 79]))))   + 
			  ((((sdata[ 80] + sdata[ 81]) + (sdata[ 82] + sdata[ 83])) + ((sdata[ 84] + sdata[ 85]) + (sdata[ 86] + sdata[ 87])))    + 
			   (((sdata[ 88] + sdata[ 89]) + (sdata[ 90] + sdata[ 91])) + ((sdata[ 92] + sdata[ 93]) + (sdata[ 94] + sdata[ 95])))))  + 
			 (((((sdata[ 96] + sdata[ 97]) + (sdata[ 98] + sdata[ 99])) + ((sdata[100] + sdata[101]) + (sdata[102] + sdata[103])))    + 
			   (((sdata[104] + sdata[105]) + (sdata[106] + sdata[107])) + ((sdata[108] + sdata[109]) + (sdata[110] + sdata[111]))))   + 
			  ((((sdata[112] + sdata[113]) + (sdata[114] + sdata[115])) + ((sdata[116] + sdata[117]) + (sdata[118] + sdata[119])))    + 
			   (((sdata[120] + sdata[121]) + (sdata[122] + sdata[123])) + ((sdata[124] + sdata[125]) + (sdata[126] + sdata[127])))))) ;
    }
}
#endif



