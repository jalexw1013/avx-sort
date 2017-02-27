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

/*void serialMergeIntrinsic(
    vec_t* A, int32_t A_length,
    vec_t* B, int32_t B_length,
    vec_t* C, uint32_t C_length);*/

void serialMergeAVX512(
    vec_t* A, int32_t A_length,
    vec_t* B, int32_t B_length,
    vec_t* C, uint32_t C_length,
    uint32_t* ASplitters, uint32_t* BSplitters);

/*void mergeNetwork(
    vec_t* A, int32_t A_length,
    vec_t* B, int32_t B_length,
    vec_t* C, uint32_t C_length);*/

/*void quickSortRecursive(
    vec_t* arr, uint32_t arr_length);*/

void parallelComboSort(vec_t* array, uint32_t array_length,void(*mergeFunction)(vec_t*,int32_t,vec_t*,int32_t,vec_t*,uint32_t), int threads);

static void AVX512mergeOutPlace(uint32_t* input, uint32_t*output, int left, int mid, int right);

#endif
