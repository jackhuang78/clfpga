#define T float16
#define GSZ 1024

__kernel
__attribute__((reqd_work_group_size (GSZ,1,1)))
void add(	
	__constant T * restrict input,
	__global T * restrict output,
	const unsigned int iterations
) {
	unsigned int i;
	unsigned int gid = get_global_id(0);

	T a = input[0];
	T c;

	for(i = 0; i < iterations; i++) {
		c += a;
		c += a;
		c += a;
		c += a;
		c += a;
		c += a;
		c += a;
		c += a;
		c += a;
		c += a;
		c += a;
		c += a;
		c += a;
		c += a;
		c += a;
		c += a;
	}


	output[gid] = c;
}
