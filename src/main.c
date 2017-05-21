#include <stdio.h>
#include <sys/time.h>
#include <stdint.h>
#include <float.h>
#include <getopt.h>
#include <assert.h>
#include <errno.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>
#include <string.h>
#include <malloc.h>
#include <x86intrin.h>
#include <unistd.h>
#include <inttypes.h>
#include <immintrin.h>

#include "utils/util.h"
#include "utils/xmalloc.h"
#include "sorts.h"
#include "main.h"

// Random Tuning Parameters
//////////////////////////////
#define INFINITY_VALUE 1073741824
#define NEGATIVE_INFINITY_VALUE 0

// Function Prototypes
//////////////////////////////
void hostParseArgs(
    int argc, char** argv);
void tester(
    vec_t**, uint32_t,vec_t**, uint32_t,
    vec_t**, uint32_t,vec_t**, uint32_t, vec_t**, uint32_t);
void initArrays(
    vec_t** A, uint32_t A_length,
    vec_t** B, uint32_t B_length,
    vec_t** C, uint32_t C_length,
    vec_t** CSorted, uint32_t Ct_length,
    vec_t** CUnsorted);
void insertData(
    vec_t** A, uint32_t A_length,
    vec_t** B, uint32_t B_length,
    vec_t** C, uint32_t C_length,
    vec_t** CSorted, uint32_t Ct_length,
    vec_t** CUnsorted);
void freeGlobalData();
#ifdef MERGE
void initMergeFilePointer(FILE** fp);
#endif
#ifdef SORT
void initSortFilePointer(FILE** fp);
#endif
#ifdef PARALLELSORT
void initParallelSortFilePointer(FILE** fp);
#endif


// Global Host Variables
////////////////////////////
vec_t*    globalA;
vec_t*    globalB;
vec_t*    globalC;
vec_t*    CSorted;
vec_t*    CUnsorted;
#ifdef MERGE
FILE *mergeFile;
#endif
#ifdef SORT
FILE *sortFile;
#endif
#ifdef PARALLELSORT
FILE *parallelSortFile;
#endif
uint32_t  h_ui_A_length                = 50;
uint32_t  h_ui_B_length                = 84;
uint32_t  h_ui_C_length                = 134; //array to put values in
uint32_t  h_ui_Ct_length               = 134; //for unsorted and sorted
uint32_t  RUNS                         = 1;
uint32_t  entropy                      = 28;
uint32_t  OutToFile                    = 0; // 1 if out put to file

//These Variables for a full testing run
const uint32_t testingEntropies[] = {1, 2, 4, 6, 10, 15, 18, 22, 25, 28, 32};
const uint32_t testingEntropiesLength = 11;
const uint32_t testingSizes[] = {100, 101, 102, 103, 104, 105, 106, 107};
const uint32_t testingSizesLength = 8;
const uint32_t testingThreads[] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100};
const uint32_t testingThreadsLength = 100;

// Host Functions
////////////////////////////
int main(int argc, char** argv)
{
    // parse langths of A and B if user entered
    hostParseArgs(argc, argv);

    uint32_t seed = time(0);
    srand(seed);

    #ifdef MERGE
    if (OutToFile) {
        initMergeFilePointer(&mergeFile);
    }
    #endif

    #ifdef SORT
    if (OutToFile) {
        initSortFilePointer(&sortFile);
    }
    #endif

    #ifdef PARALLELSORT
    if (OutToFile) {
        initParallelSortFilePointer(&parallelSortFile);
    }
    #endif

    if (!OutToFile) {
        initArrays(
            &globalA, h_ui_A_length,
            &globalB, h_ui_B_length,
            &globalC, h_ui_C_length,
            &CSorted, h_ui_Ct_length,
            &CUnsorted);

        insertData(
            &globalA, h_ui_A_length,
            &globalB, h_ui_B_length,
            &globalC, h_ui_C_length,
            &CSorted, h_ui_Ct_length,
            &CUnsorted);

        tester(
            &globalA, h_ui_A_length,
            &globalB, h_ui_B_length,
            &globalC, h_ui_C_length,
            &CSorted, h_ui_Ct_length,
            &CUnsorted, RUNS);

            freeGlobalData();
    } else {
        omp_set_dynamic(0);
        for (uint32_t i = 0; i < testingSizesLength; i++) {
            initArrays(
                &globalA, testingSizes[i]/2,
                &globalB, testingSizes[i]/2 + testingSizes[i]%2,
                &globalC, testingSizes[i],
                &CSorted, testingSizes[i],
                &CUnsorted);
            for (uint32_t j = 0; j < testingThreadsLength; j++) {
                omp_set_num_threads(testingThreads[j]);
                for (uint32_t e = 0; e < testingEntropiesLength; e++) {
                    entropy = testingEntropies[e];
                    insertData(
                        &globalA, testingSizes[i]/2,
                        &globalB, testingSizes[i]/2 + testingSizes[i]%2,
                        &globalC, testingSizes[i],
                        &CSorted, testingSizes[i],
                        &CUnsorted);
                    tester(
                        &globalA, testingSizes[i]/2,
                        &globalB, testingSizes[i]/2 + testingSizes[i]%2,
                        &globalC, testingSizes[i],
                        &CSorted, testingSizes[i],
                        &CUnsorted, RUNS);
                }
            }
            freeGlobalData();
        }
    }


}

int hostBasicCompare(const void * a, const void * b) {
    return (int) (*(vec_t *)a - *(vec_t *)b);
}

/**
 * Allocates the arrays A,B,C,CSorted, and CUnsorted
 */
void initArrays(vec_t** A, uint32_t A_length,
          vec_t** B, uint32_t B_length,
          vec_t** C, uint32_t C_length,
          vec_t** CSorted, uint32_t Ct_length,
          vec_t** CUnsorted)
{
    (*A)  = (vec_t*) xmalloc((A_length  + 8) * (sizeof(vec_t)));
    (*B)  = (vec_t*) xmalloc((B_length  + 8) * (sizeof(vec_t)));
    (*C)  = (vec_t*) xmalloc((C_length  + 32) * (sizeof(vec_t)));
    (*CSorted) = (vec_t*) xmalloc((Ct_length + 32) * (sizeof(vec_t)));
    (*CUnsorted) = (vec_t*) xmalloc((Ct_length + 32) * (sizeof(vec_t)));
}

/**
 * inserts the data into the arrays A,B,CUnsorted, and CSorted.
 * This can be called over and over to re randomize the data
 */
void insertData(vec_t** A, uint32_t A_length,
                vec_t** B, uint32_t B_length,
                vec_t** C, uint32_t C_length,
                vec_t** CSorted, uint32_t Ct_length,
                vec_t** CUnsorted)
{
    for(uint32_t i = 0; i < A_length; ++i) {
        (*A)[i] = rand() % (1 << (entropy - 1));
        (*CUnsorted)[i] = (*A)[i];
    }

    for(uint32_t i = 0; i < B_length; ++i) {
        (*B)[i] = rand() % (1 << (entropy - 1));
        (*CUnsorted)[i+A_length] = (*B)[i];
    }

    qsort((*A), A_length, sizeof(vec_t), hostBasicCompare);
    qsort((*B), B_length, sizeof(vec_t), hostBasicCompare);

    for(int i = 0; i < 8; ++i) {
        (*A)[A_length + i] = INFINITY_VALUE; (*B)[B_length + i] = INFINITY_VALUE;
    }

    // reference 'CORRECT' results
    uint32_t ai = 0,bi = 0,ci = 0;
    while(ai < A_length && bi < B_length) {
        (*CSorted)[ci++] = (*A)[ai] < (*B)[bi] ? (*A)[ai++] : (*B)[bi++];
    }
    while(ai < A_length) (*CSorted)[ci++] = (*A)[ai++];
    while(bi < B_length) (*CSorted)[ci++] = (*B)[bi++];
    // for (uint32_t i = 0; i < Ct_length; i++) {
    //     printf("InitCSorted[%i]:%i\n", i, (*CSorted)[i]);
    // }
}

void freeGlobalData() {
      free(globalA);
      free(globalB);
      free(globalC);
      free(CSorted);
      free(CUnsorted);
}

#ifdef MERGE
void initMergeFilePointer(FILE** fp) {
    char fileName[50];
    sprintf(fileName, "results/MergeResults.%li.txt", time(0));
    (*fp) = fopen(fileName, "w+");
    fprintf((*fp), "Name, Entropy, A Size, B Size, Elements Per Second, Total Time");
}

void writeToMergeOut(const char* name, uint32_t entropy, uint32_t ASize, uint32_t BSize, float time) {
    fprintf(mergeFile, "\n%s, %i, %i, %i, %i, %.20f", name, entropy, ASize, BSize, (int)((float)(ASize + BSize)/time), time);
}
#endif

#ifdef SORT
void initSortFilePointer(FILE** fp) {
    char fileName[50];
    sprintf(fileName, "results/SortResults.%li.txt", time(0));
    (*fp) = fopen(fileName, "w+");
    fprintf((*fp), "Name, Entropy, C Size, Elements Per Second, Total Time");
}

void writeToSortOut(const char* name, uint32_t entropy, uint32_t CSize, float time) {
    fprintf(sortFile, "\n%s, %i, %i, %i, %.20f", name, entropy, CSize, (int)((float)(CSize)/time), time);
}
#endif

#ifdef PARALLELSORT
void initParallelSortFilePointer(FILE** fp) {
    char fileName[50];
    sprintf(fileName, "results/ParallelSortResults.%li.txt", time(0));
    (*fp) = fopen(fileName, "w+");
    fprintf((*fp), "Name, Entropy, C Size, Number of Threads, Elements Per Second, Total Time");
}

void writeToParallelSortOut(const char* name, uint32_t entropy, uint32_t CSize, uint32_t numThreads, float time) {
    fprintf(parallelSortFile, "\n%s, %i, %i, %i, %i, %.20f", name, entropy, CSize, numThreads, (int)((float)(CSize)/time), time);
}
#endif

int verifyOutput(vec_t* output, vec_t* sortedData, uint32_t length, const char* name) {
    for(uint32_t i = 0; i < length; i++) {
        if(output[i] != sortedData[i]) {
            printf(ANSI_COLOR_RED "    Error: %s Failed To Produce Correct Results.\n", name);
            printf("    Index:%d, Given Value:%d, Correct "
            "Value:%d" ANSI_COLOR_RESET "\n", i, output[i], sortedData[i]);
            return 0;
        }
    }
    return 1;
}

void clearArray(vec_t* array, uint32_t length) {
    for (uint32_t i = 0; i < length; i++) {
        array[i] = 0;
    }
}

template <void (*Merge)(vec_t*,uint32_t,vec_t*,uint32_t,vec_t*,uint32_t)>
float testMerge(
    vec_t** A, uint32_t A_length,
    vec_t** B, uint32_t B_length,
    vec_t** C, uint32_t C_length,
    vec_t** CSorted, uint32_t runs,
    const char* algoName) {

    //create input copies in case the algorithm acciedently changes the input
    vec_t* ACopy = (vec_t*)xmalloc((A_length + 8) * sizeof(vec_t));
    memcpy(ACopy, (*A), A_length * sizeof(vec_t));

    vec_t* BCopy = (vec_t*)xmalloc((B_length + 8) * sizeof(vec_t));
    memcpy(BCopy, (*B), B_length * sizeof(vec_t));

    //clear out array just to be sure
    clearArray((*C), C_length);

    //setup timing mechanism
    float time = 0.0;

    //reset timer
    tic_reset();

    //perform actual merge
    Merge((*A), A_length, (*B), B_length, (*C), C_length);

    //get timing
    time = tic_sincelast();

    //verify output is valid
    if (!verifyOutput((*C), (*CSorted), C_length, algoName)) {
        time = -1.0;
    }

    //restore original values
    clearArray((*C), C_length);
    memcpy( (*A), ACopy, A_length * sizeof(vec_t));
    memcpy( (*B), BCopy, B_length * sizeof(vec_t));

    // printf("%s:  ", algoName);
    // printf("%18.10f\n", 1e9*((time/runs) / (float)(C_length)));
    return time;
}

template <SortTemplate Sort>
float testSort(
    vec_t** CUnsorted, uint32_t C_length,
    vec_t** CSorted, uint32_t Ct_length,
    uint32_t runs, const uint32_t splitNumber,
    const char* algoName) {

    //printf("Entering Sort\n");

    //setup timing mechanism
    float time = 0.0;

    //printf("1:%i\n", Ct_length);

    //store old values
    vec_t* unsortedCopy = (vec_t*)xmalloc(Ct_length * sizeof(vec_t));
    //printf("2\n");
    memcpy(unsortedCopy, (*CUnsorted), Ct_length * sizeof(vec_t));
    //printf("3\n");

    //reset timer
    tic_reset();

    //perform actual sort
    Sort(CUnsorted, C_length, splitNumber);

    // for (uint32_t i = 0; i < Ct_length; i++) {
    //     printf("CSorted[%i]:%i\n", i,(*CSorted)[i]);
    // }

    //get timing
    time += tic_sincelast();

    //verify output is valid
    if (!verifyOutput((*CUnsorted), (*CSorted), C_length, algoName)) {
        time = -1.0;
    }

    //restore original values
    memcpy((*CUnsorted), unsortedCopy, Ct_length * sizeof(vec_t));

    free(unsortedCopy);

    return time;
}

void printfcomma(int n) {
    if (n < 0) {
        printf ("-");
        printfcomma (-n);
        return;
    }
    if (n < 1000) {
        printf ("%d", n);
        return;
    }
    printfcomma (n/1000);
    printf (",%03d", n%1000);
}


void tester(
    vec_t** A, uint32_t A_length,
    vec_t** B, uint32_t B_length,
    vec_t** C, uint32_t C_length,
    vec_t** CSorted, uint32_t Ct_length,
    vec_t** CUnsorted, uint32_t runs)
{
    if (!OutToFile) {
        printf("Parameters\n");
        printf("Entropy: %d\n", entropy);
        printf("Runs: %i\n", runs);
        #ifdef MERGE
        printf("A Size: %" PRIu32 "\n", A_length);
        printf("B Size: %" PRIu32 "\n", B_length);
        #endif
        #ifdef SORT || PARALLELSORT
        printf("Array Size: %" PRIu32 "\n", C_length);
        #endif
        printf("\n");
    }

    #ifdef MERGE

    float serialMergeTime = 0.0;
    float serialMergeNoBranchTime = 0.0;
    float bitonicMergeRealTime = 0.0;
    #ifdef AVX512
    float avx512MergeTime = 0.0;
    #endif

    for (uint32_t run = 0; run < RUNS; run++) {
        if (serialMergeTime >= 0.0) {
            serialMergeTime += testMerge<serialMerge>(
                A, A_length, B, B_length,
                C, Ct_length, CSorted,
                runs, "Serial Merge");
        }

        if (serialMergeNoBranchTime >= 0.0) {
            serialMergeNoBranchTime += testMerge<serialMergeNoBranch>(
                A, A_length, B, B_length,
                C, Ct_length, CSorted,
                runs, "Branchless Merge");
        }

        if (bitonicMergeRealTime >= 0.0) {
            bitonicMergeRealTime += testMerge<bitonicMergeReal>(
                A, A_length, B, B_length,
                C, Ct_length, CSorted,
                runs, "Bitonic Merge");
        }

        #ifdef AVX512
        if (avx512MergeTime >= 0.0) {
            avx512MergeTime += testMerge<avx512Merge>(
                A, A_length, B, B_length,
                C, Ct_length, CSorted,
                runs, "AVX-512 Merge");
        }
        #endif

        insertData(
            A, A_length,
            B, B_length,
            C, C_length,
            CSorted, Ct_length,
            CUnsorted);
    }

    serialMergeTime /= RUNS;
    serialMergeNoBranchTime /= RUNS;
    bitonicMergeRealTime /= RUNS;
    #ifdef AVX512
    avx512MergeTime /= RUNS;
    #endif

    if (OutToFile) {
        writeToMergeOut("Serial Merge", entropy, A_length, B_length, serialMergeTime);
        writeToMergeOut("Serial Merge Branchless", entropy, A_length, B_length, serialMergeNoBranchTime);
        writeToMergeOut("Bitonic Merge", entropy, A_length, B_length, bitonicMergeRealTime);
        #ifdef AVX512
        writeToMergeOut("AVX512 Merge", entropy, A_length, B_length, avx512MergeTime);
        #endif
    }

    if (!OutToFile) {
        printf("Merging Results        :  Elements per Second\n");
        printf("Serial Merge           :     ");printfcomma((int)((float)Ct_length/serialMergeTime));printf("\n");
        printf("Serial Merge Branchless:     ");printfcomma((int)((float)Ct_length/serialMergeNoBranchTime));printf("\n");
        printf("Bitonic Merge          :     ");printfcomma((int)((float)Ct_length/bitonicMergeRealTime));printf("\n");
        #ifdef AVX512
        printf("AVX512 Merge           :     ");printfcomma((int)((float)Ct_length/avx512MergeTime));printf("\n");
        #endif
        printf("\n\n");
        #endif
    }

    #ifdef SORT

    float quickSortTime = 0.0;
    float serialMergeSortTime = 0.0;
    float serialMergeNoBranchSortTime = 0.0;
    float bitonicMergeRealSortTime = 0.0;
    #ifdef AVX512
    float avx512MergeSortTime = 0.0;
    #endif

    for (uint32_t run = 0; run < RUNS; run++) {
        if (quickSortTime >= 0.0) {
            quickSortTime += testSort<quickSort>(
                CUnsorted, C_length,
                CSorted, Ct_length,
                runs, 64, "Quick Sort");
        }

        if (serialMergeSortTime >= 0.0) {
            serialMergeSortTime += testSort<iterativeMergeSort<serialMerge>>(
                CUnsorted, C_length,
                CSorted, Ct_length,
                runs, 64, "Merge Sort Using Serial Merge");
        }

        if (serialMergeNoBranchSortTime >= 0.0) {
            serialMergeNoBranchSortTime += testSort<iterativeMergeSort<serialMergeNoBranch>>(
                CUnsorted, C_length,
                CSorted, Ct_length,
                runs, 64, "Merge Sort Using Branchless Merge");
        }

        if (bitonicMergeRealSortTime >= 0.0) {
            bitonicMergeRealSortTime += testSort<iterativeMergeSort<bitonicMergeReal>>(
                CUnsorted, C_length,
                CSorted, Ct_length,
                runs, 64, "Merge Sort Using Bitonic Merge");
        }

        #ifdef AVX512
        if (avx512MergeSortTime >= 0.0) {
            avx512MergeSortTime += testSort<iterativeMergeSort<avx512Merge>>(
                CUnsorted, C_length,
                CSorted, Ct_length,
                runs, 64, "Merge Sort Using AVX512 Merge");
        }
        #endif

        insertData(
            A, A_length,
            B, B_length,
            C, C_length,
            CSorted, Ct_length,
            CUnsorted);
    }

    quickSortTime /= RUNS;
    serialMergeSortTime /= RUNS;
    serialMergeNoBranchSortTime /= RUNS;
    bitonicMergeRealSortTime /= RUNS;
    #ifdef AVX512
    avx512MergeSortTime /= RUNS;
    #endif

    if (OutToFile) {
        writeToSortOut("Quick Sort", entropy, C_length, quickSortTime);
        writeToSortOut("Merge Sort using Serial Merge", entropy, C_length, serialMergeSortTime);
        writeToSortOut("Merge Sort using Serial Merge Branchless", entropy, C_length, serialMergeNoBranchSortTime);
        writeToSortOut("Merge Sort using Bitonic Merge", entropy, C_length, bitonicMergeRealSortTime);
        #ifdef AVX512
        writeToSortOut("Merge Sort using AVX512 Merge", entropy, C_length, avx512MergeSortTime);
        #endif
    }

    if (!OutToFile) {
        printf("Sorting Results                         :  Elements per Second\n");
        printf("Quick Sort                              :     ");printfcomma((int)((float)Ct_length/quickSortTime));printf("\n");
        printf("Merge Sort using Serial Merge           :     ");printfcomma((int)((float)Ct_length/serialMergeSortTime));printf("\n");
        printf("Merge Sort using Serial Merge Branchless:     ");printfcomma((int)((float)Ct_length/serialMergeNoBranchSortTime));printf("\n");
        printf("Merge Sort using Bitonic Merge          :     ");printfcomma((int)((float)Ct_length/bitonicMergeRealSortTime));printf("\n");
        #ifdef AVX512
        printf("Merge Sort using AVX512 Merge           :     ");printfcomma((int)((float)Ct_length/avx512MergeSortTime));printf("\n");
        #endif
        printf("\n\n");
        #endif
    }

    #ifdef PARALLELSORT

        float serialMergeParallelSortTime = 0.0;
        float serialMergeNoBranchParallelSortTime = 0.0;
        float bitonicMergeRealParallelSortTime = 0.0;
        #ifdef AVX512
        float avx512MergeParallelSortTime = 0.0;
        #endif

        uint32_t numberOfThreads = 0;
        //Check how many threads are running
        #pragma omp parallel
        {
            #pragma omp single
            {
                numberOfThreads = omp_get_num_threads();
            }
        }

        for (uint32_t run = 0; run < RUNS; run++) {
            if (serialMergeParallelSortTime >= 0.0) {
                serialMergeParallelSortTime += testSort<parallelIterativeMergeSort<serialMerge>>(
                                                    CUnsorted, C_length,
                                                    CSorted, Ct_length,
                                                    runs, 64, "Parallel Merge Sort Using Serial Merge    ");
            }

            if (serialMergeNoBranchParallelSortTime >= 0.0) {
                serialMergeNoBranchParallelSortTime += testSort<parallelIterativeMergeSort<serialMergeNoBranch>>(
                                                            CUnsorted, C_length,
                                                            CSorted, Ct_length,
                                                            runs, 64, "Parallel Merge Sort Using Branchless Merge");
            }

            if (bitonicMergeRealParallelSortTime >= 0.0) {
                bitonicMergeRealParallelSortTime += testSort<parallelIterativeMergeSort<bitonicMergeReal>>(
                                                        CUnsorted, C_length,
                                                        CSorted, Ct_length,
                                                        runs, 64, "Parallel Merge Sort Using Bitonic Merge   ");
            }

            #ifdef AVX512
            if (avx512MergeParallelSortTime >= 0.0)
                avx512MergeParallelSortTime += testSort<parallelIterativeMergeSort<avx512Merge>>(
                                                    CUnsorted, C_length,
                                                    CSorted, Ct_length,
                                                    runs, 64, "Parallel Merge Sort Using AVX512 Merge   ");
            #endif

            insertData(
                A, A_length,
                B, B_length,
                C, C_length,
                CSorted, Ct_length,
                CUnsorted);
        }

        serialMergeParallelSortTime /= RUNS;
        serialMergeNoBranchParallelSortTime /= RUNS;
        bitonicMergeRealParallelSortTime /= RUNS;
        #ifdef AVX512
        avx512MergeParallelSortTime /= RUNS;
        #endif

        if (OutToFile) {
            writeToParallelSortOut("Parallel Merge Sort using Serial Merge", entropy, C_length, numberOfThreads, serialMergeParallelSortTime);
            writeToParallelSortOut("Parallel Merge Sort using Serial Merge Branchless", entropy, C_length, numberOfThreads, serialMergeNoBranchParallelSortTime);
            writeToParallelSortOut("Parallel Merge Sort using Bitonic Merge", entropy, C_length, numberOfThreads, bitonicMergeRealParallelSortTime);
            #ifdef AVX512
            writeToParallelSortOut("Parallel Merge Sort using AVX512 Merge", entropy, C_length, numberOfThreads, avx512MergeParallelSortTime);
            #endif
        }

        if (!OutToFile) {
            printf("Parallel Sorting Results                         :  Elements per Second\n");
            printf("Parallel Merge Sort using Serial Merge           :     ");printfcomma((int)((float)Ct_length/serialMergeParallelSortTime));printf("\n");
            printf("Parallel Merge Sort using Serial Merge Branchless:     ");printfcomma((int)((float)Ct_length/serialMergeNoBranchParallelSortTime));printf("\n");
            printf("Parallel Merge Sort using Bitonic Merge          :     ");printfcomma((int)((float)Ct_length/bitonicMergeRealParallelSortTime));printf("\n");
            #ifdef AVX512
            printf("Parallel Merge Sort using AVX512 Merge           :     ");printfcomma((int)((float)Ct_length/avx512MergeParallelSortTime));printf("\n");
            #endif
            printf("\n\n");
        }

    #endif
}

void MergePathSplitter(
    vec_t * A, uint32_t A_length,
    vec_t * B, uint32_t B_length,
    vec_t * C, uint32_t C_length,
    uint32_t threads, uint32_t* ASplitters, uint32_t* BSplitters)
{

    // for (uint32_t i = 0; i < A_length; i++) {
    //     printf("A[%i]:%i\n", i, A[i]);
    // }
    //
    // for (uint32_t i = 0; i < B_length; i++) {
    //     printf("B[%i]:%i\n", i, B[i]);
    // }

  for (uint32_t i = 0; i <= threads; i++) {
      ASplitters[i] = A_length;
      BSplitters[i] = B_length;
  }

  // printf("A_Length:%i\n", A_length);
  // printf("B_Length:%i\n", B_length);

  // for (uint32_t i = 0; i <= threads; i++) {
  //     printf("SPLITTER:ASplitters:%i\n", ASplitters[i]);
  //     printf("SPLITTER:BSplitters:%i\n", BSplitters[i]);
  // }

  uint32_t minLength = A_length > B_length ? B_length : A_length;

  for (uint32_t thread=0; thread<threads;thread++)
  {
    // uint32_t thread = omp_get_thread_num();
    uint32_t combinedIndex = thread * (minLength * 2) / threads;
    uint32_t x_top, y_top, x_bottom, current_x, current_y, offset, oldx, oldy;
    x_top = combinedIndex > minLength ? minLength : combinedIndex;
    y_top = combinedIndex > (minLength) ? combinedIndex - (minLength) : 0;
    x_bottom = y_top;

    // printf("%i:combinedIndex:%i\n", thread, combinedIndex);
    // printf("%i:X_top:%i\n", thread, combinedIndex);
    // printf("%i:y_top:%i\n", thread, combinedIndex);
    // printf("%i:x_bottom:%i\n", thread, combinedIndex);

    oldx = -1;
    oldy = -1;

    vec_t Ai, Bi;
    while(1) {
      offset = (x_top - x_bottom) / 2;
      if (x_top < x_bottom) {
          offset = 0;
      }
      current_y = y_top + offset;
      current_x = x_top - offset;

    //   printf("%i:y_top:%i\n", thread, y_top);
    //   printf("%i:x_top:%i\n", thread, x_top);
    //   printf("%i:x_bottom:%i\n", thread, x_bottom);
    //   printf("%i:offset:%i\n", thread, offset);
    //   printf("%i:current_x:%i\n", thread, current_x);
    //   printf("%i:current_y:%i\n", thread, current_y);

      if (current_x == oldx || current_y == oldy) {
          return;
      }

      oldx = current_x;
      oldy = current_y;

      if(current_x > A_length - 1 || current_y < 1) {
        Ai = 1;Bi = 0;
      } else {
        Ai = A[current_x];Bi = B[current_y - 1];
      }
      if(Ai > Bi) {
        if(current_y > B_length - 1 || current_x < 1) {
          Ai = 0;Bi = 1;
        } else {
          Ai = A[current_x - 1];Bi = B[current_y];
        }

        if(Ai <= Bi) {//Found it
            // printf("%i:A:%i\n", thread, current_x);
            // printf("%i:B:%i\n", thread, current_y);
          ASplitters[thread]   = current_x;
          BSplitters[thread] = current_y;
          break;
        } else {//Both zeros
          x_top = current_x - 1;y_top = current_y + 1;
        }
      } else {// Both ones
        x_bottom = current_x + 1;
      }
  }
}

for (uint32_t i = 0; i <= threads; i++) {
    // printf("InMergePathASplitters[%i]:%i\n", i, ASplitters[i]);
    // printf("InMergePathBSplitters[%i]:%i\n", i, BSplitters[i]);
}

  // printf("%i:A:%i\n", threads, ASplitters[threads]);
  // printf("%i:B:%i\n", threads, ASplitters[threads]);
}

#define PRINT_ARRAY_INDEX(ARR,IND) for (int t=0; t<threads;t++){printf("%10d, ",ARR[IND[t]]);}printf("\n");

void hostParseArgs(int argc, char** argv)
{
  static struct option long_options[] = {
    {"Alength", required_argument, 0, 'A'},
    {"Blength", required_argument, 0, 'B'},
    {"Runs"   , optional_argument, 0, 'R'},
    {"Entropy", optional_argument, 0, 'E'},
    {"FileOut", no_argument      , 0, 'F'},
    {"help"   , no_argument      , 0, 'h'},
    {0        , 0                , 0,  0 }
  };

  while(1) {
    int option_index = 0;
    int c = getopt_long(argc, argv, "A:B:R:E:h:F",
  long_options, &option_index);
    extern char * optarg;
    //extern int    optind, opterr, optopt;
    int intout = 0;

    if(-1 == c)
      break;

    switch(c) {
      default:
        printf("Unrecognized option: %c\n\n", c);
      case 'h':
        printf("\nCreates two arrays A and B and "
            "merges them into array C in parallel on OpenMP.");
        printf("\n\nUsage"
            "\n====="
            "\n\n\t-A --Alength <number>"
            "\n\t\tSpecify the length of randomly "
            "generated array A.\n"
            "\n\t-B --Blength <number>"
            "\n\t\tSpecify the length of randomly "
            "generated array B.\n"
            "\n\t-R --Runs <number>"
            "\n\t\tSpecify the number of runs to be used for "
            "serial and parallel algorithms.\n"
            "\n\t-E --Entropy <number>"
            "\n\t\tSpecify the number of bits of entropy "
            "to be used for random number generation\n"
            "\n\t-F --FileOut 0 or 1"
            "\n\t\tSpecify weather to write full "
            "output to a file\n"
            );
         exit(0);
         break;
      case 'A':
        errno = 0;
        intout = strtol(optarg, NULL, 10);
        if(errno || intout < 0) {
          printf("Error - Alength %s\n", optarg);
          exit(-1);
        }
        h_ui_A_length = intout;
        break;
      case 'B':
        errno = 0;
        intout = strtol(optarg, NULL, 10);
        if(errno || intout < 0) {
          printf("Error - Blength %s\n", optarg);
          exit(-1);
        }
        h_ui_B_length = intout;
        break;
      case 'R':
        intout = strtol(optarg, NULL, 10);
        if(errno || intout < 0) {
          printf("Error - Runs %s\n", optarg);
          exit(-1);
        }
        RUNS = intout;
        break;
      case 'E':
        intout = strtol(optarg, NULL, 10);
        if(errno || intout < 0) {
          printf("Error - Entropy %s\n", optarg);
          exit(-1);
        }
        entropy = (intout < 1) ? 1: intout;
        break;
      case 'F':
        OutToFile = 1;
        break;
    }
  }
  h_ui_C_length = h_ui_A_length + h_ui_B_length;
  h_ui_Ct_length = h_ui_C_length;
}
