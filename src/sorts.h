#ifndef HEADER_FILE_SORTS
#define HEADER_FILE_SORTS

#include "main.h"

/*
 * Merging Functions
 */

typedef void (*AlgoTemplate)(struct AlgoArgs*);

extern void serialMerge(struct AlgoArgs *args);
extern void serialMergeNoBranch(struct AlgoArgs *args);
extern void bitonicMergeReal(struct AlgoArgs *args);
extern void avx512Merge(struct AlgoArgs *args);

template <AlgoTemplate Merge>
void parallelMerge(struct AlgoArgs *args);

/*
 * Sorting Functions
 */

void quickSort(struct AlgoArgs *args);

void ippSort(struct AlgoArgs *args);
void ippRadixSort(struct AlgoArgs *args);
void tbbSort(struct AlgoArgs *args);
void haichuanwangSort(struct AlgoArgs *args);

template <AlgoTemplate Merge>
void avx512SortNoMergePathV2(struct AlgoArgs *args);

template <AlgoTemplate Merge>
void iterativeMergeSort(struct AlgoArgs *args);

template <AlgoTemplate Sort, AlgoTemplate Merge>
void parallelIterativeMergeSort(struct AlgoArgs *args);

#endif
