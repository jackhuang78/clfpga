#define WGSZ 1024

__kernel
__attribute__((reqd_work_group_size (WGSZ,1,1)))
void local_read(	
	__constant uint * restrict input,		
	__local uint * restrict shared,		
	__global uint * restrict output, 		
	const uint iterations
) {

	uint i;
	uint gid = get_global_id(0);
	uint tid = get_local_id(0);
	uint next = tid;
	
	// Copy data from global to local memory
	shared[tid + 0] = input[tid + 0];
	shared[tid + 1] = input[tid + 1];
	shared[tid + 2] = input[tid + 2];
	shared[tid + 3] = input[tid + 3];
	shared[tid + 4] = input[tid + 4];
	shared[tid + 5] = input[tid + 5];
	shared[tid + 6] = input[tid + 6];
	shared[tid + 7] = input[tid + 7];
	shared[tid + 8] = input[tid + 8];
	shared[tid + 9] = input[tid + 9];
	shared[tid + 10] = input[tid + 10];
	shared[tid + 11] = input[tid + 11];
	shared[tid + 12] = input[tid + 12];
	shared[tid + 13] = input[tid + 13];
	shared[tid + 14] = input[tid + 14];
	shared[tid + 15] = input[tid + 15];
	//barrier(CLK_LOCAL_MEM_FENCE);

	// traverse local memory
	for(i = 0; i < iterations; i++) {

		// chase pointer through the local memory once
		next = shared[next];
		next = shared[next];
		next = shared[next];
		next = shared[next];
		next = shared[next];
		next = shared[next];
		next = shared[next];
		next = shared[next];	// 8
		next = shared[next];
		next = shared[next];
		next = shared[next];
		next = shared[next];
		next = shared[next];
		next = shared[next];
		next = shared[next];
		next = shared[next];	// 16

	}
	output[gid] = next;
}
