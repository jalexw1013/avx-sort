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

void serialMergeAVX512(
    vec_t* A, int32_t A_length,
    vec_t* B, int32_t B_length,
    vec_t* C, uint32_t C_length,
    uint32_t* ASplitters, uint32_t* BSplitters);

void iterativeComboMergeSort(
    vec_t* array,
    uint32_t array_length/*,
    void(*mergeFunction)(vec_t*,int32_t,vec_t*,int32_t,vec_t*,uint32_t)*/);

void iterativeNonParallelComboMergeSort(
    vec_t* array,
    uint32_t array_length,
    uint32_t numThreads,
    void(*mergeFunction)(vec_t*,int32_t,vec_t*,int32_t,vec_t*,uint32_t));

#endif
