#ifndef HEADER_FILE_SORTS
#define HEADER_FILE_SORTS

#include "main.h"
typedef uint32_t vec_t;

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

void serialMergeIntrinsic(
    vec_t* A, int32_t A_length,
    vec_t* B, int32_t B_length,
    vec_t* C, uint32_t C_length);

void serialMergeAVX512(
    vec_t* A, int32_t A_length,
    vec_t* B, int32_t B_length,
    vec_t* C, uint32_t C_length);

void serialMergeAVX2(
    vec_t* A, int32_t A_length,
    vec_t* B, int32_t B_length,
    vec_t* C, uint32_t C_length);

void mergeNetwork(
    vec_t* A, int32_t A_length,
    vec_t* B, int32_t B_length,
    vec_t* C, uint32_t C_length);

#endif
