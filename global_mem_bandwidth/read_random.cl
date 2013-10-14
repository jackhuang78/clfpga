#define STRIDE 512
#define NUM_READS 32
#define DATATYPE float
#define LSZ 256
#define GSZ (1024 * 4 *  LSZ)

__kernel void read_random(__global DATATYPE *input,__global DATATYPE *output)
{
DATATYPE val = (DATATYPE)(0.0f);
uint gid = get_global_id(0);
uint index = 0;
val = val + input[gid];			// 0
val = val + input[val & GSZ];
val = val + input[val & GSZ];
val = val + input[val & GSZ];
val = val + input[val & GSZ];
val = val + input[val & GSZ];	// 5
val = val + input[val & GSZ];
val = val + input[val & GSZ];
val = val + input[val & GSZ];
val = val + input[val & GSZ];
val = val + input[val & GSZ];	// 10
val = val + input[val & GSZ];
val = val + input[val & GSZ];
val = val + input[val & GSZ];
val = val + input[val & GSZ];
val = val + input[val & GSZ];	// 15
val = val + input[val & GSZ];
val = val + input[val & GSZ];
val = val + input[val & GSZ];
val = val + input[val & GSZ];
val = val + input[val & GSZ];	// 20
val = val + input[val & GSZ];
val = val + input[val & GSZ];
val = val + input[val & GSZ];
val = val + input[val & GSZ];
val = val + input[val & GSZ];	// 25
val = val + input[val & GSZ];
val = val + input[val & GSZ];
val = val + input[val & GSZ];
val = val + input[val & GSZ];
val = val + input[val & GSZ];	// 30
val = val + input[val & GSZ];

output[gid] = val;

}
