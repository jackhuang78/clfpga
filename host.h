#ifdef ALTERA
#define KERNEL_EXT ".aocx"
#else
#define KERNEL_EXT ".cl"
#endif

#ifndef TESTSZ
#define TESTSZ 10
#endif

#ifndef DATASZ
#define DATASZ 23
#endif

#ifndef VECTSZ
#define VECTSZ 2
#endif

#ifndef LOCALSZ
#define LOCALSZ 128
#endif

#ifndef MARGIN
#define MARGIN 0.01
#endif
