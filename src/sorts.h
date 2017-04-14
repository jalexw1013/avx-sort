#ifndef HEADER_FILE_SORTS
#define HEADER_FILE_SORTS

#include "main.h"

/*
 * Merging Functions
 */
typedef void (*MergeTemplate)(vec_t*,uint32_t,vec_t*,uint32_t,vec_t*,uint32_t);

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

/*
 * Sorting Functions
 */
typedef void (*SortTemplate)(vec_t**, uint32_t, const uint32_t, MergeTemplate);

void quickSort(
    vec_t** array, uint32_t array_length, const uint32_t splitNumber,
    MergeTemplate Merge);

void iterativeMergeSort(
    vec_t** array, uint32_t array_length, const uint32_t splitNumber,
    MergeTemplate Merge);

void parallelIterativeMergeSort(
    vec_t** array, uint32_t array_length, const uint32_t splitNumber,
    MergeTemplate Merge);

void srinivasMergeSort(
    vec_t** array, uint32_t array_length, const uint32_t splitNumber,
    MergeTemplate Merge);

void alexRecursiveQuickSort(
    vec_t** array, uint32_t array_length, const uint32_t splitNumber,
    MergeTemplate Merge);

void recursiveMergeSort(
    vec_t** array, uint32_t array_length, const uint32_t splitNumber,
    MergeTemplate Merge);

void srinivasSSEMergeSort(
    vec_t** array, uint32_t array_length, const uint32_t splitNumber,
    MergeTemplate Merge);

#endif
