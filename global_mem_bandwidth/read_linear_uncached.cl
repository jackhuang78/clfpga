#define STRIDE 512
#define NUM_READS 32
#define DATATYPE float
#define OFFSET 4096

__kernel void read_linear_uncached(__global DATATYPE *input,__global DATATYPE *output)
{
DATATYPE val = (DATATYPE)(0.0f);
uint gid = get_global_id(0);
uint index = gid;
val = val + input[index += OFFSET];
val = val + input[index += OFFSET];
val = val + input[index += OFFSET];
val = val + input[index += OFFSET];
val = val + input[index += OFFSET];
val = val + input[index += OFFSET];
val = val + input[index += OFFSET];
val = val + input[index += OFFSET];
val = val + input[index += OFFSET];
val = val + input[index += OFFSET];
val = val + input[index += OFFSET];
val = val + input[index += OFFSET];
val = val + input[index += OFFSET];
val = val + input[index += OFFSET];
val = val + input[index += OFFSET];
val = val + input[index += OFFSET];
val = val + input[index += OFFSET];
val = val + input[index += OFFSET];
val = val + input[index += OFFSET];
val = val + input[index += OFFSET];
val = val + input[index += OFFSET];
val = val + input[index += OFFSET];
val = val + input[index += OFFSET];
val = val + input[index += OFFSET];
val = val + input[index += OFFSET];
val = val + input[index += OFFSET];
val = val + input[index += OFFSET];
val = val + input[index += OFFSET];
val = val + input[index += OFFSET];
val = val + input[index += OFFSET];
val = val + input[index += OFFSET];
val = val + input[index += OFFSET];
output[gid] = val;
}
