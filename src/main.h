#ifndef HEADER_FILE_MAIN
#define HEADER_FILE_MAIN

#include <stdio.h>
#include "xmalloc.h"
#include <sys/time.h>
#include <stdint.h>
#include <float.h>
#include <getopt.h>
#include <assert.h>
#include <errno.h>
#include "util.h"
#include <stdlib.h>
#include <omp.h>
#include <time.h>
#include <string.h>

#include <malloc.h>

#include <x86intrin.h>
#include <avx512bwintrin.h>
#include <avx512cdintrin.h>
#include <avx512dqintrin.h>
#include <avx512erintrin.h>
#include <avx512fintrin.h>
#include <avx512ifmaintrin.h>
#include <avx512ifmavlintrin.h>
#include <avx512pfintrin.h>
#include <avx512vbmiintrin.h>
#include <avx512vbmivlintrin.h>
#include <avx512vlbwintrin.h>
#include <avx512vldqintrin.h>
#include <avx512vlintrin.h>


#include "sorts.h"

typedef uint32_t vec_t;

void MergePathSplitter(
    vec_t * A, uint32_t A_length,
    vec_t * B, uint32_t B_length,
    vec_t * C, uint32_t C_length,
    uint32_t threads, uint32_t* splitters);
#endif
