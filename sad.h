#define IMAGE_S		1024
#define FILTER_S	17	// must be odd
#define OUT_S		(IMAGE_S - FILTER_S + 1)

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









