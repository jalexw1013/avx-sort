#ifndef HEADER_FILE_SORTS
#define HEADER_FILE_SORTS

#include "main.h"

void serialMerge(
    vec_t* A, uint32_t A_length,
    vec_t* B, uint32_t B_length,
    vec_t* C, uint32_t C_length);

void serialMergeNoBranch(
    vec_t* A, uint32_t A_length,
    vec_t* B, uint32_t B_length,
    vec_t* C, uint32_t C_length);

void bitonicMergeReal(
    vec_t* A, uint32_t A_length,
    vec_t* B, uint32_t B_length,
    vec_t* C, uint32_t C_length);

#ifdef AVX512
void avx512Merge(
    vec_t* A, uint32_t A_length,
    vec_t* B, uint32_t B_length,
    vec_t* C, uint32_t C_length);
#endif

void quickSort(vec_t** array, uint32_t array_length, const uint32_t splitNumber);

//Iterative Merge Sort
template <void (*Sort)(vec_t**, uint32_t),
    void (*Merge)(vec_t*,uint32_t,vec_t*,uint32_t,vec_t*,uint32_t)>
void iterativeMergeSort(
    vec_t** array, uint32_t array_length, const uint32_t splitNumber);

void parallelIterativeMergeSort(vec_t** array, uint32_t array_length);

#endif
