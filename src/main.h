#ifndef HEADER_FILE_MAIN
#define HEADER_FILE_MAIN

typedef uint32_t vec_t;

void MergePathSplitter(
    vec_t * A, uint32_t A_length,
    vec_t * B, uint32_t B_length,
    vec_t * C, uint32_t C_length,
    uint32_t threads, uint32_t* splitters);
#endif
