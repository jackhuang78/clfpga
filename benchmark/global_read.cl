#define WGSZ 1024

__kernel
__attribute__((reqd_work_group_size (WGSZ,1,1)))
void global_read(	
	__global uint * restrict input,
	__global uint * restrict output,
	const uint iterations
) {

	uint i;
	uint gid = get_global_id(0);
	uint next = gid;
	
	// read global memory
	for(i = 0; i < iterations; i++) {
		next = input[next];
		next = input[next];
		next = input[next];
		next = input[next];
		next = input[next];
		next = input[next];
		next = input[next];
		next = input[next];	// 8
		next = input[next];
		next = input[next];
		next = input[next];
		next = input[next];
		next = input[next];
		next = input[next];
		next = input[next];
		next = input[next];	// 16
	}

	output[gid] = next;
}
