#define STRIDE 512
#define NUM_READS 32
#define DATATYPE float

__kernel void write_linear(__constant DATATYPE *c0, __global DATATYPE *output)
{
uint gid = get_global_id(0);
output[gid + 0 * (STRIDE / sizeof(DATATYPE))] =  *c0;
output[gid + 1 * (STRIDE / sizeof(DATATYPE))] =  *c0;
output[gid + 2 * (STRIDE / sizeof(DATATYPE))] =  *c0;
output[gid + 3 * (STRIDE / sizeof(DATATYPE))] =  *c0;
output[gid + 4 * (STRIDE / sizeof(DATATYPE))] =  *c0;
output[gid + 5 * (STRIDE / sizeof(DATATYPE))] =  *c0;
output[gid + 6 * (STRIDE / sizeof(DATATYPE))] =  *c0;
output[gid + 7 * (STRIDE / sizeof(DATATYPE))] =  *c0;
output[gid + 8 * (STRIDE / sizeof(DATATYPE))] =  *c0;
output[gid + 9 * (STRIDE / sizeof(DATATYPE))] =  *c0;
output[gid + 10 * (STRIDE / sizeof(DATATYPE))] =  *c0;
output[gid + 11 * (STRIDE / sizeof(DATATYPE))] =  *c0;
output[gid + 12 * (STRIDE / sizeof(DATATYPE))] =  *c0;
output[gid + 13 * (STRIDE / sizeof(DATATYPE))] =  *c0;
output[gid + 14 * (STRIDE / sizeof(DATATYPE))] =  *c0;
output[gid + 15 * (STRIDE / sizeof(DATATYPE))] =  *c0;
output[gid + 16 * (STRIDE / sizeof(DATATYPE))] =  *c0;
output[gid + 17 * (STRIDE / sizeof(DATATYPE))] =  *c0;
output[gid + 18 * (STRIDE / sizeof(DATATYPE))] =  *c0;
output[gid + 19 * (STRIDE / sizeof(DATATYPE))] =  *c0;
output[gid + 20 * (STRIDE / sizeof(DATATYPE))] =  *c0;
output[gid + 21 * (STRIDE / sizeof(DATATYPE))] =  *c0;
output[gid + 22 * (STRIDE / sizeof(DATATYPE))] =  *c0;
output[gid + 23 * (STRIDE / sizeof(DATATYPE))] =  *c0;
output[gid + 24 * (STRIDE / sizeof(DATATYPE))] =  *c0;
output[gid + 25 * (STRIDE / sizeof(DATATYPE))] =  *c0;
output[gid + 26 * (STRIDE / sizeof(DATATYPE))] =  *c0;
output[gid + 27 * (STRIDE / sizeof(DATATYPE))] =  *c0;
output[gid + 28 * (STRIDE / sizeof(DATATYPE))] =  *c0;
output[gid + 29 * (STRIDE / sizeof(DATATYPE))] =  *c0;
output[gid + 30 * (STRIDE / sizeof(DATATYPE))] =  *c0;
output[gid + 31 * (STRIDE / sizeof(DATATYPE))] =  *c0;
}
