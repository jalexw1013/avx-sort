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

int uint32Compare(const void *one, const void *two);

void parallelComboSort(vec_t* array, uint32_t array_length,void(*mergeFunction)(vec_t*,int32_t,vec_t*,int32_t,vec_t*,uint32_t), int cpus);


#endif
