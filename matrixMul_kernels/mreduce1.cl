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
    Parallel sum reduction using shared memory
    - takes log(n) steps for n input elements
    - uses n threads
    - only works for power-of-2 arrays
    This version uses sequential addressing -- no divergence or bank conflicts.
*/

// Kernel attributes

__kernel 
__attribute ((reqd_work_group_size(WORK_GROUP/VECTOR_SIZE,1,1))) 
__attribute ((num_simd_work_items(16/VECTOR_SIZE))) 
__attribute ((num_compute_units(NUM_UNITS))) 
__attribute ((max_share_resources(8))) 
void mreduce1(__global T *g_idata, __global T *g_odata, unsigned int n)
{
    // load shared mem
    unsigned int local_index = get_local_id(0);
    unsigned int global_index = get_global_id(0);
    
    __local T sdata[WORK_GROUP];   
 
    sdata[local_index] = (global_index < n) ? g_idata[global_index] : 0;
    
    barrier(CLK_LOCAL_MEM_FENCE);

    // do reduction in shared mem
    #pragma unroll
    for(unsigned int s=get_local_size(0)/2; s>0; s>>=1) 
    {
        if (local_index < s) 
        {
            sdata[local_index] += sdata[local_index + s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // write result for this block to global mem
    if (local_index == 0) g_odata[get_group_id(0)] = sdata[0];
}
#endif
