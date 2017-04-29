#ifndef HEADER_FILE_MAIN
#define HEADER_FILE_MAIN

#define min(a,b) (a <= b)? a : b
#define max(a,b) (a <  b)? b : a

typedef uint32_t vec_t;

void MergePathSplitter(
    vec_t * A, uint32_t A_length,
    vec_t * B, uint32_t B_length,
    vec_t * C, uint32_t C_length,
    uint32_t threads, uint32_t* ASplitters, uint32_t* BSplitters);

void singlePathMergePathSplitter(
    vec_t * A, uint32_t A_length,
    vec_t * B, uint32_t B_length,
    vec_t * C, uint32_t C_length,
    uint32_t thread, uint32_t threads,
    uint32_t* ASplitters, uint32_t* BSplitters);

int hostBasicCompare(const void * a, const void * b);

#endif
