#define IMAGE_S		4100
#define FILTER_S	5	// must be odd
#define OUT_S		(IMAGE_S - FILTER_S + 1)
#define TEMP_S		(WG_S + FILTER_S - 1)

#define IMAGE(r,c)	image[(r) * IMAGE_S + (c)]
#define FILTER(r,c)	filter[(r) * FILTER_S + (c)]
#define OUT(r,c)	out[(r) * OUT_S + (c)]
#define TEMP(r,c)	temp[(r) * TEMP_S + (c)]

#define MIN 0
#define MAX 256
#define ITER 10

#define RAND_INIT() srand((unsigned)time(0))
#define RAND_INT() (rand() % (MAX-MIN) - MIN)
#define IDX(r, c, w) ((r) * (w) + (c))		// convert from row/column to index
#define ABS(x) ((x) >= 0 ? (x) : -(x))		// absolute value function
#define SIZEOF(s, t) (sizeof(t) * s * s)

#define AOCL_ALIGNMENT 64
#define WG_S		16	// must be less than sqrt(MAX_WORKGROUP_DIMENSION)









