#define T int

#define IMAGE_S		1024
#define FILTER_S	10
#define OUT_S		(IMAGE_S - FILTER_S + 1)
#define TEMP_S		(WG_S + FILTER_S - 1)

#define IMAGE(r,c)	image[(r) * image_s + (c)]
#define FILTER(r,c)	filter[(r) * filter_s + (c)]
#define OUT(r,c)	out[(r) * out_s + (c)]
#define TEMP(r,c)	temp[(r) * temp_s + (c)]

#define MIN 0
#define MAX 256
#define ITER 5

#define RAND_INIT() srand((unsigned)time(0))
#define RAND_INT() (rand() % (MAX-MIN) - MIN)
#define IDX(r, c, w) ((r) * (w) + (c))		// convert from row/column to index
#define ABS(x) ((x) >= 0 ? (x) : -(x))		// absolute value function
#define SIZEOF(s, t) (sizeof(t) * s * s)

#define AOCL_ALIGNMENT 64
#define WG_S 16	

#define DEBUG 0
#define DEBUG_PRINT(x) if(DEBUG) { x; }






