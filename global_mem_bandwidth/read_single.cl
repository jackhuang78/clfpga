#define STRIDE 512
#define NUM_READS 32
#define DATATYPE float

__kernel void read_single(__global DATATYPE *input,__global DATATYPE *output)
{
DATATYPE val = (DATATYPE)(0.0f);
uint gid = get_global_id(0);
uint index = 0;
val = val + input[index + 0];
val = val + input[index + 1];
val = val + input[index + 2];
val = val + input[index + 3];
val = val + input[index + 4];
val = val + input[index + 5];
val = val + input[index + 6];
val = val + input[index + 7];
val = val + input[index + 8];
val = val + input[index + 9];
val = val + input[index + 10];
val = val + input[index + 11];
val = val + input[index + 12];
val = val + input[index + 13];
val = val + input[index + 14];
val = val + input[index + 15];
val = val + input[index + 16];
val = val + input[index + 17];
val = val + input[index + 18];
val = val + input[index + 19];
val = val + input[index + 20];
val = val + input[index + 21];
val = val + input[index + 22];
val = val + input[index + 23];
val = val + input[index + 24];
val = val + input[index + 25];
val = val + input[index + 26];
val = val + input[index + 27];
val = val + input[index + 28];
val = val + input[index + 29];
val = val + input[index + 30];
val = val + input[index + 31];
output[gid] = val;

}
