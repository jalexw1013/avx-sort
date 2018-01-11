#ifndef HEADER_FILE_MAIN
#define HEADER_FILE_MAIN

#define min(a,b) (a <= b)? a : b
#define max(a,b) (a <  b)? b : a

typedef uint32_t vec_t;

struct AlgoArgs {
    vec_t* A;
    uint64_t A_length;
    vec_t* B;
    uint64_t B_length;
    vec_t* C;
    uint64_t C_length;
    vec_t* CUnsorted;
    uint32_t threadNum;
    uint32_t numThreads;
    uint32_t* ASplitters;
    uint32_t* BSplitters;
    uint32_t* arraySizes;
};

void MergePathSplitter(
    vec_t * A, uint64_t A_length,
    vec_t * B, uint64_t B_length,
    vec_t * C, uint64_t C_length,
    uint32_t threads, uint32_t* ASplitters, uint32_t* BSplitters);

int hostBasicCompare(const void * a, const void * b);

#endif
