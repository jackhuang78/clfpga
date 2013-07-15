#define T int

#define IMAGE_S		1024
#define FILTER_S	15
#define OUT_S		(IMAGE_S - FILTER_S + 1)
#define TEMP_S		(LOCAL_S + FILTER_S - 1)
#define LOCAL_S		32
#define GLOBAL_S	((OUT_S + (LOCAL_S - 1)) / LOCAL_S * LOCAL_S)

#define IMAGE_SZ	(IMAGE_S * IMAGE_S * sizeof(T))
#define FILTER_SZ	(FILTER_S * FILTER_S * sizeof(T))
#define OUT_SZ		(OUT_S * OUT_S * sizeof(T))
#define TEMP_SZ		(TEMP_S * TEMP_S * sizeof(T))
#define LOCAL_SZ	(IMAGE_S * IMAGE_S)
#define GLOBAL_SZ	(GLOBAL_S * GLOBAL_S)

#define IMAGE(r,c)	image[(r) * IMAGE_S + (c)]
#define FILTER(r,c)	filter[(r) * FILTER_S + (c)]
#define OUT(r,c)	out[(r) * OUT_S + (c)]
#define IMAGE_TEMP(i, j) image_temp[(i) * TEMP_S + (j)]
#define FILTER_TEMP(i, j) filter_temp[(i) * FILTER_S + (j)]

#define MIN 0
#define MAX 256
#define ITER 5

//srand((unsigned)time(0))
#define RAND_INIT() srand(10)
#define RAND_INT() (rand() % (MAX-MIN) - MIN)
#define ABS(x) ((x) >= 0 ? (x) : -(x))		// absolute value function

#define AOCL_ALIGNMENT 64

#define DEBUG 0
#define DEBUG_PRINT(x) if(DEBUG) { x; }






