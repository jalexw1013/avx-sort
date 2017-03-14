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

#ifdef __INTEL_COMPILER
void serialMergeAVX512(
    vec_t* A, int32_t A_length,
    vec_t* B, int32_t B_length,
    vec_t* C, uint32_t C_length,
    uint32_t* ASplitters, uint32_t* BSplitters);

void knightMergeOutPlace(
    uint32_t* input, uint32_t*output, int left, int mid, int right);

void iterativeComboMergeSortAVX512(vec_t* array, uint32_t array_length);

#endif

void iterativeComboMergeSort(
    vec_t* array,
    uint32_t array_length/*,
    void(*mergeFunction)(vec_t*,int32_t,vec_t*,int32_t,vec_t*,uint32_t)*/);

void iterativeNonParallelComboMergeSort(
    vec_t* array,
    uint32_t array_length,
    uint32_t numThreads,
    void(*mergeFunction)(vec_t*,int32_t,vec_t*,int32_t,vec_t*,uint32_t));

void quicksort(uint32_t * a, uint32_t p, uint32_t r);

#endif
