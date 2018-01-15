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
#include <stdbool.h>

#include "utils/util.h"
#include "utils/xmalloc.h"
#include "sorts.h"
#include "main.h"
#include "ipp.h"
#include "ippcore_l.h"
#include "ipps_l.h"

// Random Tuning Parameters
//////////////////////////////
#define INFINITY_VALUE 1073741824
#define NEGATIVE_INFINITY_VALUE 0

// Function Prototypes
//////////////////////////////
void hostParseArgs(
    int argc, char** argv);
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
#ifdef MERGE
void mergeTester(
    vec_t** A, uint32_t A_length,
    vec_t** B, uint32_t B_length,
    vec_t** C, uint32_t C_length,
    vec_t** CSorted, uint32_t Ct_length,
    vec_t** CUnsorted, uint32_t runs);
void parallelMergeTester(
    vec_t** A, uint32_t A_length,
    vec_t** B, uint32_t B_length,
    vec_t** C, uint32_t C_length,
    vec_t** CSorted, uint32_t Ct_length,
    vec_t** CUnsorted, uint32_t runs);
#endif
#ifdef SORT
void sortTester(
    vec_t** A, uint32_t A_length,
    vec_t** B, uint32_t B_length,
    vec_t** C, uint32_t C_length,
    vec_t** CSorted, uint32_t Ct_length,
    vec_t** CUnsorted, uint32_t runs);
#endif
#ifdef PARALLELSORT
void parallelTester(
    vec_t** A, uint32_t A_length,
    vec_t** B, uint32_t B_length,
    vec_t** C, uint32_t C_length,
    vec_t** CSorted, uint32_t Ct_length,
    vec_t** CUnsorted, uint32_t runs);
#endif

void freeGlobalData();
#ifdef MERGE
void initMergeFilePointer(FILE** fp);
void initParallelMergeFilePointer(FILE** fp);
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
uint64_t  globalALength;
vec_t*    globalB;
uint64_t  globalBLength;
vec_t*    globalC;
uint64_t  globalCLength;
vec_t*    CSorted;
vec_t*    CUnsorted;

FILE *mergeFile;
FILE *parallelMergeFile;
FILE *sortFile;
FILE *parallelSortFile;

uint64_t  h_ui_A_length                = 0;
uint64_t  h_ui_B_length                = 0;
uint64_t  h_ui_C_length                = 0; //array to put values in
uint64_t  h_ui_Ct_length               = 0; //for unsorted and sorted
uint32_t  RUNS                         = 1;
uint32_t  entropy                      = 28;
uint32_t  OutToFile                    = 0; // 1 if output to file

uint32_t testingEntropies[] = {28};
uint32_t testingEntropiesLength = 1;
uint32_t testingSizes[] = {1048576};
uint32_t testingSizesLength = 1;
uint32_t testingThreads[] = {64};
uint32_t testingThreadsLength = 1;

// Host Functions
////////////////////////////

int hostBasicCompare(const void * a, const void * b) {
    return (int) (*(vec_t *)a - *(vec_t *)b);
}

/**
 * Allocates the arrays A,B,C,CSorted, and CUnsorted
 */
void initArrays(vec_t** A, uint64_t A_length,
          vec_t** B, uint64_t B_length,
          vec_t** C, uint64_t C_length,
          vec_t** CSorted, uint64_t Ct_length,
          vec_t** CUnsorted)
{
    (*A)  = (vec_t*) xmalloc((A_length  + 8) * (sizeof(vec_t)));
    (*B)  = (vec_t*) xmalloc((B_length  + 8) * (sizeof(vec_t)));
    (*C)  = (vec_t*) xmalloc((C_length  + 32) * (sizeof(vec_t)));
    #ifdef VERIFYOUTPUT
    (*CSorted) = (vec_t*) xmalloc((Ct_length + 32) * (sizeof(vec_t)));
    #endif
    (*CUnsorted) = (vec_t*) xmalloc((Ct_length + 32) * (sizeof(vec_t)));
}

/**
 * inserts the data into the arrays A,B,CUnsorted, and CSorted.
 * This can be called over and over to re randomize the data
 */
void insertData(vec_t** A, uint64_t A_length,
                vec_t** B, uint64_t B_length,
                vec_t** C, uint64_t C_length,
                vec_t** CSorted, uint64_t Ct_length,
                vec_t** CUnsorted)
{
    for(uint64_t i = 0; i < A_length; ++i) {
        (*A)[i] = rand() % (1 << (entropy - 1));
        (*CUnsorted)[i] = (*A)[i];
    }

    for(uint64_t i = 0; i < B_length; ++i) {
        (*B)[i] = rand() % (1 << (entropy - 1));
        (*CUnsorted)[i+A_length] = (*B)[i];
    }

    qsort((*A), A_length, sizeof(vec_t), hostBasicCompare);
    qsort((*B), B_length, sizeof(vec_t), hostBasicCompare);

    for(int i = 0; i < 8; ++i) {
        (*A)[A_length + i] = INFINITY_VALUE; (*B)[B_length + i] = INFINITY_VALUE;
    }

    // reference 'CORRECT' results
    #ifdef VERIFYOUTPUT
    uint32_t ai = 0,bi = 0,ci = 0;
    while(ai < A_length && bi < B_length) {
        (*CSorted)[ci++] = (*A)[ai] < (*B)[bi] ? (*A)[ai++] : (*B)[bi++];
    }
    while(ai < A_length) (*CSorted)[ci++] = (*A)[ai++];
    while(bi < B_length) (*CSorted)[ci++] = (*B)[bi++];
    #endif
}

void freeGlobalData() {
      free(globalA);
      free(globalB);
      free(globalC);
      #ifdef VERIFYOUTPUT
      free(CSorted);
      #endif
      free(CUnsorted);
}

void initMergeFilePointer(FILE** fp) {
    char fileName[50];
    sprintf(fileName, "../results/MergeResults.%li.csv", time(0));
    (*fp) = fopen(fileName, "w+");
    fprintf((*fp), "Name,Entropy,A Size,B Size,Elements Per Second,Total Time");
}

void writeToMergeOut(const char* name, uint32_t entropy, uint64_t ASize, uint64_t BSize, float time) {
    fprintf(mergeFile, "\n%s,%i,%lu,%lu,%i,%.20f", name, entropy, ASize, BSize, (int)((float)(ASize + BSize)/time), time);
}

void initParallelMergeFilePointer(FILE** fp) {
    char fileName[50];
    sprintf(fileName, "../results/ParallelMergeResults.%li.csv", time(0));
    (*fp) = fopen(fileName, "w+");
    fprintf((*fp), "Name,Entropy,A Size,B Size,Number of Threads,Elements Per Second,Total Time");
}

void writeToParallelMergeOut(const char* name, uint32_t entropy, uint64_t ASize, uint64_t BSize, uint32_t numThreads,float time) {
    fprintf(parallelMergeFile, "\n%s,%i,%lu,%lu,%i,%i,%.20f", name, entropy, ASize, BSize, numThreads,(int)((float)(ASize + BSize)/time), time);
}

void initSortFilePointer(FILE** fp) {
    char fileName[50];
    sprintf(fileName, "../results/SortResults.%li.csv", time(0));
    (*fp) = fopen(fileName, "w+");
    fprintf((*fp), "Name,Entropy,C Size,Elements Per Second,Total Time");
}

void writeToSortOut(const char* name, uint32_t entropy, uint64_t CSize, float time) {
    fprintf(sortFile, "\n%s,%i,%lu,%i,%.20f", name, entropy, CSize, (int)((float)(CSize)/time), time);
}

void initParallelSortFilePointer(FILE** fp) {
    char fileName[50];
    sprintf(fileName, "../results/ParallelSortResults.%li.csv", time(0));
    (*fp) = fopen(fileName, "w+");
    fprintf((*fp), "Name,Entropy,C Size,Number of Threads,Elements Per Second,Total Time");
}

void writeToParallelSortOut(const char* name, uint32_t entropy, uint64_t CSize, uint32_t numThreads, float time) {
    fprintf(parallelSortFile, "\n%s,%i,%lu,%i,%i,%.20f", name, entropy, CSize, numThreads, (int)((float)(CSize)/time), time);
}

int verifyUnsignedOutput(vec_t* output, vec_t* sortedData, uint64_t length, const char* name, uint32_t numThreads) {
    for(uint64_t i = 0; i < length; i++) {
        if(output[i] != sortedData[i]) {
            printf(ANSI_COLOR_RED "    Error: %s Failed To Produce Correct Results.\n", name);
            printf("    Index:%lu, Given Value:%d, Correct "
            "Value:%d, ArraySize: %lu NumThreads: %d" ANSI_COLOR_RESET "\n", i, output[i], sortedData[i], length, numThreads);
            return 0;
        }
    }
    return 1;
}

int verifyOutput(vec_t* outputU, vec_t* CU, vec_t* sortedDataU, uint64_t length, const char* name, uint32_t numThreads, bool isSigned) {
    if (isSigned) {
        // int32_t* output = (int32_t*)outputU;
        // int32_t* sortedData = (int32_t*)sortedDataU;
        // for(uint64_t i = 0; i < length; i++) {
        //     if(output[i] != sortedData[i]) {
        //         printf(ANSI_COLOR_RED "    Error: %s Failed To Produce Correct Results.\n", name);
        //         printf("    Index:%lu, Given Value:%d, Correct "
        //         "Value:%d, ArraySize: %lu NumThreads: %d" ANSI_COLOR_RESET "\n", i, output[i], sortedData[i], length, numThreads);
        //         return 0;
        //     }
        // }
    } else {
        vec_t* output = (vec_t*)outputU;
        vec_t* COut = (vec_t*)CU;
        vec_t* sortedData = (vec_t*)sortedDataU;
        for(uint64_t i = 0; i < length; i++) {
            if(output[i] != sortedData[i] && COut[i] != sortedData[i]) {
                printf(ANSI_COLOR_RED "    Error: %s Failed To Produce Correct Results.\n", name);
                printf("    Index:%lu, Given Value:%d, Correct "
                "Value:%d, ArraySize: %lu NumThreads: %d" ANSI_COLOR_RESET "\n", i, output[i], sortedData[i], length, numThreads);
                return 0;
            }
        }
    }
    return 1;
}

void clearArray(vec_t* array, uint64_t length) {
    for (uint64_t i = 0; i < length; i++) {
        array[i] = 0;
    }
}

int hostBasicCompare2(const void * a, const void * b) {
    return (int) (*(int32_t *)a - *(int32_t *)b);
}

void printfcomma(int n) {
    if (n < 0) {
        printf ("N/A");
        return;
    }
    if (n < 1000) {
        printf ("%d", n);
        return;
    }
    printfcomma (n/1000);
    printf (",%03d", n%1000);
}

template <AlgoTemplate Algo>
void testAlgo(const char* algoName, bool threadSpawn, bool isSigned) {

    // Create input copies in case the algorithm accidentally changes the input
    #ifdef VERIFYOUTPUT
    vec_t* ACopy = (vec_t*)xmalloc((globalALength + 8) * sizeof(vec_t));
    memcpy(ACopy, globalA, globalALength * sizeof(vec_t));
    vec_t* BCopy = (vec_t*)xmalloc((globalBLength + 8) * sizeof(vec_t));
    memcpy(BCopy, globalB, globalBLength * sizeof(vec_t));
    #endif

    // Copy C since we expect it to be modified
    vec_t* CCopy = (vec_t*)xmalloc((globalCLength) * sizeof(vec_t));
    memcpy(CCopy, CUnsorted, (globalCLength) * sizeof(vec_t));

    uint32_t numberOfThreads = 1;

    // Allocate Splitters
    uint32_t* ASplitters = (uint32_t*)xcalloc((numberOfThreads + 1)*numberOfThreads, sizeof(uint32_t));
    uint32_t* BSplitters = (uint32_t*)xcalloc((numberOfThreads + 1)*numberOfThreads, sizeof(uint32_t));
    uint32_t* arraySizes = (uint32_t*)xcalloc((numberOfThreads + 1)*numberOfThreads, sizeof(uint32_t));

    // Set Algorithm Arguments
    struct AlgoArgs *algoArgs = (struct AlgoArgs*)xcalloc(1, sizeof(struct AlgoArgs));
    algoArgs->A = globalA;
    algoArgs->A_length = globalALength;
    algoArgs->B = globalB;
    algoArgs->B_length = globalBLength;
    algoArgs->C = globalC;
    algoArgs->C_length = globalCLength;
    algoArgs->CUnsorted = CUnsorted;
    // algoArgs->threadNum;
    // algoArgs->numThreads;
    algoArgs->ASplitters;
    algoArgs->BSplitters;
    algoArgs->arraySizes;


    //setup timing mechanism
    float time = 0.0;

    // Run Algorithm
    if (threadSpawn) {
        #pragma omp parallel
        {
            #pragma omp single
            {
                numberOfThreads = omp_get_num_threads();
                tic_reset();
            }
            Algo(algoArgs);
            #pragma omp barrier
            #pragma omp single
            {
                time = tic_total();
            }
        }
    } else {
        tic_reset();
        Algo(algoArgs);
        time = tic_total();
    }

    // Output Results
    if (OutToFile) {
        // TODO
        //writeToMergeOut("Basic Merge", entropy, A_length, B_length, serialMergeTime);
        //writeToMergeOut("Serial Merge Branchless", entropy, A_length, B_length, serialMergeNoBranchTime);
        //writeToMergeOut("SSE Bitonic Merge", entropy, A_length, B_length, bitonicMergeRealTime);
        //writeToMergeOut("AVX-512 Merge Path Based Merge", entropy, A_length, B_length, avx512MergeTime);
        //writeToMergeOut("AVX512 Parallel Merge", entropy, A_length, B_length, avx512ParallelMergeTime);
    } else {
        // Pad the input string
        // First find length of name
        char a;
        int size = 0;
        int found = 0;
        for (int i = 0; i < 30; i++) {
            if (algoName[i] == '\0') {
                found = 1;
                break;
            }
            size++;
        }
        if (!found) {
            printf("Error in Algorithm Name. Must be less than 30 characters\n");
            return;
        }
        char name[31];
        for (int i = 0; i < size; i++) {
            name[i] = algoName[i];
        }
        for (int i = size; i < 30; i++) {
            name[i] = ' ';
        }
        name[30] = '\0';

        // Print the results
        printf("%s:     ", name);
        if (time > 0.0) {
            printfcomma((int)((float)globalCLength/time));
        } else if (time == 0.0) {
            printf("∞");
        } else {
            printf("N/A");
        }
        printf("\n");
    }

    // Restore original values
    #ifdef VERIFYOUTPUT
    verifyOutput(CUnsorted, globalC, CSorted, globalCLength, algoName, numberOfThreads, isSigned);
    clearArray(globalC, globalCLength);
    memcpy(globalA, ACopy, globalALength * sizeof(vec_t));
    memcpy(globalB, BCopy, globalBLength * sizeof(vec_t));
    free(ACopy);
    free(BCopy);
    #endif

    memcpy(CUnsorted, CCopy, (globalCLength) * sizeof(vec_t));
    free(CCopy);

    // free(ASplitters);
    // free(BSplitters);
    free(algoArgs);
}

template <void (*Merge)(vec_t*,uint64_t,vec_t*,uint64_t,vec_t*,uint64_t, struct memPointers*)>
float testMerge(
    vec_t** A, uint64_t A_length,
    vec_t** B, uint64_t B_length,
    vec_t** C, uint64_t C_length,
    vec_t** CSorted, uint32_t runs,
    const char* algoName) {

    if (verifyOutput) {
        //create input copies in case the algorithm acciedently changes the input
        vec_t* ACopy = (vec_t*)xmalloc((A_length + 8) * sizeof(vec_t));
        memcpy(ACopy, (*A), A_length * sizeof(vec_t));

        vec_t* BCopy = (vec_t*)xmalloc((B_length + 8) * sizeof(vec_t));
        memcpy(BCopy, (*B), B_length * sizeof(vec_t));
    }

    // uint32_t* ASplitters = (uint32_t*)xcalloc(17, sizeof(uint32_t));
    // uint32_t* BSplitters = (uint32_t*)xcalloc(17, sizeof(uint32_t));
    struct memPointers* pointers = (struct memPointers*)xcalloc(1, sizeof(struct memPointers));
    // pointers->ASplitters = ASplitters;
    // pointers->BSplitters = BSplitters;

    if (verifyOutput) {
        //clear out array just to be sure
        clearArray((*C), C_length);
    }

    // Lastly, clear cache for fair results
    // FILE *fp = fopen ("/proc/sys/vm/drop_caches", "w");
    // fprintf (fp, "3");
    // fclose (fp);

    //setup timing mechanism
    float time = 0.0;

    //reset timer
    tic_reset();

    //perform actual merge
    Merge((*A), A_length, (*B), B_length, (*C), C_length, pointers);

    //get timing
    time = tic_total();

    //verify output is valid
    if (verifyOutput) {
        verifyOutput((*C), (*CSorted), C_length, algoName, 0);
    }

    //restore original values
    if (verifyOutput) {
        clearArray((*C), C_length);
        memcpy( (*A), ACopy, A_length * sizeof(vec_t));
        memcpy( (*B), BCopy, B_length * sizeof(vec_t));
    }

    if (verifyOutput) {
        free(ACopy);
        free(BCopy);
        //free(ASplitters);
        //free(BSplitters);
    }
    free(pointers);

    return time;
}

template <ParallelMergeTemplate ParallelMerge>
float testParallelMerge(
    vec_t** A, uint64_t A_length,
    vec_t** B, uint64_t B_length,
    vec_t** C, uint64_t C_length,
    vec_t** CSorted, uint32_t runs,
    const char* algoName) {

    //create input copies in case the algorithm acciedently changes the input
    if (verifyOutput) {
        vec_t* ACopy = (vec_t*)xmalloc((A_length + 8) * sizeof(vec_t));
        memcpy(ACopy, (*A), A_length * sizeof(vec_t));

        vec_t* BCopy = (vec_t*)xmalloc((B_length + 8) * sizeof(vec_t));
        memcpy(BCopy, (*B), B_length * sizeof(vec_t));
    }

    // Check how many threads are running
    // This it technically not guaranteed but this is the easiest way
    uint32_t numberOfThreads = 0;
    #pragma omp parallel
    {
        #pragma omp single
        {
            numberOfThreads = omp_get_num_threads();
        }
    }

    //allocate Variables
    uint32_t* ASplitters = (uint32_t*)xcalloc((numberOfThreads + 1)*numberOfThreads, sizeof(uint32_t));
    uint32_t* BSplitters = (uint32_t*)xcalloc((numberOfThreads + 1)*numberOfThreads, sizeof(uint32_t));
    struct memPointers* pointers = (struct memPointers*)xcalloc(1, sizeof(struct memPointers));
    pointers->ASplitters = ASplitters;
    pointers->BSplitters = BSplitters;

    //clear out array just to be sure
    if (verifyOutput) {
        clearArray((*C), C_length);
    }

    //setup timing mechanism
    float time = 0.0;

    //reset timer
    tic_reset();

    //perform actual merge
    ParallelMerge((*A), A_length, (*B), B_length, (*C), C_length, pointers);

    //get timing
    time = tic_total();

    //verify output is valid
    // if (verifyOutput) {
    //     verifyOutput((*C), (*CSorted), C_length, algoName, numberOfThreads);
    // }

    //restore original values
    // if (verifyOutput) {
    //     clearArray((*C), C_length);
    //     memcpy( (*A), ACopy, A_length * sizeof(vec_t));
    //     memcpy( (*B), BCopy, B_length * sizeof(vec_t));
    // }

    // if (verifyOutput) {
    //     free(ACopy);
    //     free(BCopy);
    // }
    // free(ASplitters);
    // free(BSplitters);
    // free(pointers);

    return time;
}

template <SortTemplate Sort>
float testSort(
    vec_t** CUnsorted, uint64_t C_length,
    vec_t** CSorted, uint64_t Ct_length,
    uint32_t runs, const uint32_t splitNumber,
    const char* algoName, int isSigned) {

    //setup timing mechanism
    float time = 0.0;

    //store old values
    if (verifyOutput) {
        vec_t* unsortedCopy = (vec_t*)xmalloc(Ct_length * sizeof(vec_t));
        memcpy(unsortedCopy, (*CUnsorted), Ct_length * sizeof(vec_t));
    }

    //allocate Variables
    //vec_t* C = (vec_t*)xcalloc((C_length + 32), sizeof(vec_t));
    // uint32_t* ASplitters = (uint32_t*)xcalloc(17, sizeof(uint32_t));
    // uint32_t* BSplitters = (uint32_t*)xcalloc(17, sizeof(uint32_t));
    struct memPointers* pointers = (struct memPointers*)xcalloc(1, sizeof(struct memPointers));
    // pointers->ASplitters = ASplitters;
    // pointers->BSplitters = BSplitters;

    //reset timer
    tic_reset();

    //perform actual sort
    Sort((*CUnsorted), C, C_length, splitNumber, pointers);

    //get timing
    time += tic_sincelast();
    if (verifyOutput) {
        if (isSigned) {
            int32_t* CSortedCopy = (int32_t*)xmalloc((C_length) * sizeof(int32_t));
            memcpy(CSortedCopy, (*CSorted), C_length * sizeof(int32_t));
            qsort((void*)CSortedCopy, C_length, sizeof(int32_t), hostBasicCompare2);
            verifySignedOutput((int32_t*)(*CUnsorted), CSortedCopy, C_length, algoName, 0);
            free(CSortedCopy);
        } else {
            verifyOutput((*CUnsorted), (*CSorted), C_length, algoName, 0);
        }
    }

    //restore original values
    // if (verifyOutput) {
    //     memcpy((*CUnsorted), unsortedCopy, Ct_length * sizeof(vec_t));
    // }

    //deallocate variables
    //free(C);
    // if (verifyOutput) {
    //     free(unsortedCopy);
    // }
    // free(ASplitters);
    // free(BSplitters);
    // free(pointers);

    return time;
}

template <ParallelSortTemplate ParallelSort>
float testParallelSort(
    vec_t** CUnsorted, uint64_t C_length,
    vec_t** CSorted, uint64_t Ct_length,
    uint32_t runs, const uint32_t splitNumber,
    const char* algoName) {

    //setup timing mechanism
    float time = 0.0;

    //store old values
    //vec_t* unsortedCopy = (vec_t*)xmalloc(Ct_length * sizeof(vec_t));
    //memcpy(unsortedCopy, (*CUnsorted), Ct_length * sizeof(vec_t));

    //Check how many threads are running
    uint32_t numberOfThreads = 0;
    #pragma omp parallel
    {
        #pragma omp single
        {
            numberOfThreads = omp_get_num_threads();
        }
    }

    //allocate Variables
    vec_t* C = (vec_t*)xcalloc((C_length + 32), sizeof(vec_t));
    uint32_t* ASplitters = (uint32_t*)xcalloc((numberOfThreads + 1)*numberOfThreads, sizeof(uint32_t));
    uint32_t* BSplitters = (uint32_t*)xcalloc((numberOfThreads + 1)*numberOfThreads, sizeof(uint32_t));
    uint32_t* arraySizes = (uint32_t*)xcalloc(numberOfThreads*numberOfThreads, sizeof(uint32_t));
    struct memPointers* pointers = (struct memPointers*)xcalloc(1, sizeof(struct memPointers));
    pointers->ASplitters = ASplitters;
    pointers->BSplitters = BSplitters;
    pointers->arraySizes = arraySizes;

    //reset timer
    tic_reset();

    //perform actual sort
    ParallelSort((*CUnsorted), C, C_length, splitNumber, pointers);

    //get timing
    time += tic_sincelast();

    //verify output is valid
    //if (!verifyOutput((*CUnsorted), (*CSorted), C_length, algoName, numberOfThreads)) {
        //time = -1.0;
    //}

    //restore original values
    //memcpy((*CUnsorted), unsortedCopy, Ct_length * sizeof(vec_t));

    //deallocate variables
    // free(C);
    // //free(unsortedCopy);
    // free(ASplitters);
    // free(BSplitters);
    // free(arraySizes);
    // free(pointers);

    return time;
}



#ifdef MERGE
void mergeTester(
    vec_t** A, uint64_t A_length,
    vec_t** B, uint64_t B_length,
    vec_t** C, uint64_t C_length,
    vec_t** CSorted, uint32_t Ct_length,
    vec_t** CUnsorted, uint32_t runs)
{
    return;
    if (!OutToFile) {
        printf("Parameters\n");
        printf("Entropy: %d\n", entropy);
        printf("Runs: %i\n", runs);
        printf("A Size: %lu\n", A_length);
        printf("B Size: %lu\n", B_length);
        printf("\n");
    }

    double serialMergeTime = 0.0;
    //double serialMergeNoBranchTime = 0.0;
    double bitonicMergeRealTime = 0.0;
    #ifdef AVX512
    double avx512MergeTime = 0.0;
    //double avx512ParallelMergeTime = 0.0;
    #endif

    for (uint32_t run = 0; run < RUNS; run++) {
        // serialMergeTime += testMerge<serialMerge>(
        //     A, A_length, B, B_length,
        //     C, Ct_length, CSorted,
        //     runs, "Serial Merge", 1);
        //
        // serialMergeNoBranchTime += testMerge<serialMergeNoBranch>(
        //     A, A_length, B, B_length,
        //     C, Ct_length, CSorted,
        //     runs, "Branchless Merge");
        //
        // bitonicMergeRealTime += testMerge<bitonicMergeReal>(
        //     A, A_length, B, B_length,
        //     C, Ct_length, CSorted,
        //     runs, "Bitonic Merge", 1);
        //
        // #ifdef AVX512
        // avx512MergeTime += testMerge<avx512Merge>(
        //     A, A_length, B, B_length,
        //     C, Ct_length, CSorted,
        //     runs, "AVX-512 Merge", 1);
        //
        // /*avx512ParallelMergeTime += testMerge<avx512ParallelMerge>(
        //     A, A_length, B, B_length,
        //     C, Ct_length, CSorted,
        //     runs, "AVX-512 Parallel Merge");*/
        // #endif
        //
        // insertData(
        //     A, A_length,
        //     B, B_length,
        //     C, C_length,
        //     CSorted, Ct_length,
        //     CUnsorted);
    }

    serialMergeTime /= RUNS;
    //serialMergeNoBranchTime /= RUNS;
    bitonicMergeRealTime /= RUNS;
    #ifdef AVX512
    avx512MergeTime /= RUNS;
    //avx512ParallelMergeTime /= RUNS;
    #endif

    if (OutToFile) {
        writeToMergeOut("Basic Merge", entropy, A_length, B_length, serialMergeTime);
        //writeToMergeOut("Serial Merge Branchless", entropy, A_length, B_length, serialMergeNoBranchTime);
        writeToMergeOut("SSE Bitonic Merge", entropy, A_length, B_length, bitonicMergeRealTime);
        #ifdef AVX512
        writeToMergeOut("AVX-512 Merge Path Based Merge", entropy, A_length, B_length, avx512MergeTime);
        //writeToMergeOut("AVX512 Parallel Merge", entropy, A_length, B_length, avx512ParallelMergeTime);
        #endif
    } else {
        printf("Merging Results                : Elements per Second\n");
        printf("Basic Merge                    :     ");
        if (serialMergeTime > 0.0) {
            printfcomma((int)((float)Ct_length/serialMergeTime));
        } else if (serialMergeTime == 0.0) {
            printf("∞");
        } else {
            printf("N/A");
        }
        printf("\n");
        // printf("Serial Merge Branchless:     ");
        // if (serialMergeNoBranchTime > 0.0) {
        //     printfcomma((int)((float)Ct_length/serialMergeNoBranchTime));
        // } else if (serialMergeNoBranchTime == 0.0) {
        //     printf("∞");
        // } else {
        //     printf("N/A");
        // }
        // printf("\n");
        printf("SSE Bitonic Merge              :     ");
        if (bitonicMergeRealTime > 0.0) {
            printfcomma((int)((float)Ct_length/bitonicMergeRealTime));
        } else if (bitonicMergeRealTime == 0.0) {
            printf("∞");
        } else {
            printf("N/A");
        }
        printf("\n");
        #ifdef AVX512
        printf("AVX-512 Merge Path Based Merge :     ");
        if (avx512MergeTime > 0.0) {
            printfcomma((int)((float)Ct_length/avx512MergeTime));
        } else if (avx512MergeTime == 0.0) {
            printf("∞");
        } else {
            printf("N/A");
        }
        printf("\n");
        // printf("AVX-512 Merge Path Based Merge   :     ");
        // if (avx512ParallelMergeTime > 0.0) {
        //     printfcomma((int)((double)Ct_length/avx512ParallelMergeTime));
        // } else if (avx512ParallelMergeTime == 0.0) {
        //     printf("∞");
        // } else {
        //     printf("N/A");
        // }
        // printf("\n");
        #endif
        printf("\n\n");

        // printf("Serial Merge           :     %f\n", serialMergeTime);
        // printf("Serial Merge Branchless:     %f\n", serialMergeNoBranchTime);
        // printf("Bitonic Merge          :     %f\n", bitonicMergeRealTime);
        // #ifdef AVX512
        // printf("AVX512 Merge           :     %f\n", avx512MergeTime);
        // printf("AVX512 Parallel Merge  :     %f\n", avx512ParallelMergeTime);
        // #endif
    }
}

void parallelMergeTester(
    vec_t** A, uint64_t A_length,
    vec_t** B, uint64_t B_length,
    vec_t** C, uint64_t C_length,
    vec_t** CSorted, uint64_t Ct_length,
    vec_t** CUnsorted, uint32_t runs)
{
    return;
    uint32_t numberOfThreads = 0;
    #pragma omp parallel
    {
        #pragma omp single
        {
            numberOfThreads = omp_get_num_threads();
        }
    }

    if (!OutToFile) {
        printf("Parameters\n");
        printf("Entropy: %d\n", entropy);
        printf("Runs: %i\n", runs);
        printf("A Size: %lu\n", A_length);
        printf("B Size: %lu\n", B_length);
        printf("Number Of Threads: %d\n", numberOfThreads);
        printf("\n");
    }

    double serialMergeParallelTime = 0.0;
    double bitonicMergeParallelTime = 0.0;
    #ifdef AVX512
    double avx512ParallelMergeTime = 0.0;
    #endif

    for (uint32_t run = 0; run < RUNS; run++) {
        // serialMergeParallelTime += testParallelMerge<parallelMerge<serialMerge>>(
        //     A, A_length, B, B_length,
        //     C, Ct_length, CSorted,
        //     runs, "Standard");
        // bitonicMergeParallelTime += testParallelMerge<parallelMerge<bitonicMergeReal>>(
        //     A, A_length, B, B_length,
        //     C, Ct_length, CSorted,
        //     runs, "Bitonic");
        #ifdef AVX512
        // avx512ParallelMergeTime += testParallelMerge<parallelMerge<avx512Merge>>(
        //     A, A_length, B, B_length,
        //     C, Ct_length, CSorted,
        //     runs, "AVX-512 MP");
        #endif

        insertData(
            A, A_length,
            B, B_length,
            C, C_length,
            CSorted, Ct_length,
            CUnsorted);
    }

    #ifdef AVX512
    avx512ParallelMergeTime /= RUNS;
    #endif
    if (OutToFile) {
        writeToParallelMergeOut("Standard", entropy, A_length, B_length, numberOfThreads,serialMergeParallelTime);
        writeToParallelMergeOut("Bitonic", entropy, A_length, B_length, numberOfThreads,bitonicMergeParallelTime);
        #ifdef AVX512
        //writeToParallelMergeOut("AVX-512 MP", entropy, A_length, B_length, numberOfThreads,avx512ParallelMergeTime);
        #endif
    } else {
        printf("Parallel Merging Results :  Elements per Second\n");
        printf("Standard     :     ");
        if (serialMergeParallelTime > 0.0) {
            printfcomma((int)((double)Ct_length/serialMergeParallelTime));
        } else if (serialMergeParallelTime == 0.0) {
            printf("∞");
        } else {
            printf("N/A");
        }
        printf("\n");
        printf("Bitonic     :     ");
        if (bitonicMergeParallelTime > 0.0) {
            printfcomma((int)((double)Ct_length/bitonicMergeParallelTime));
        } else if (bitonicMergeParallelTime == 0.0) {
            printf("∞");
        } else {
            printf("N/A");
        }
        printf("\n");
        #ifdef AVX512
        printf("AVX512 MP   :     ");
        if (avx512ParallelMergeTime > 0.0) {
            printfcomma((int)((double)Ct_length/avx512ParallelMergeTime));
        } else if (avx512ParallelMergeTime == 0.0) {
            printf("∞");
        } else {
            printf("N/A");
        }
        printf("\n");
        #endif
        printf("\n\n");

        // #ifdef AVX512
        // printf("AVX512 Parallel Merge  :     %f\n", avx512ParallelMergeTime);
        // #endif
    }
}
#endif

void libinfo(void) {
  const IppLibraryVersion* lib = ippsGetLibVersion();
  printf("%s %s %d.%d.%d.%d\n",
  lib->Name, lib->Version,
  lib->major,
  lib->minor, lib->majorBuild, lib->build);
}

#ifdef SORT
void sortTester(
    vec_t** A, uint64_t A_length,
    vec_t** B, uint64_t B_length,
    vec_t** C, uint64_t C_length,
    vec_t** CSorted, uint64_t Ct_length,
    vec_t** CUnsorted, uint32_t runs)
{
    return;
    libinfo();
    if (!OutToFile) {
        printf("Parameters\n");
        printf("Entropy: %d\n", entropy);
        printf("Runs: %d\n", runs);
        ippSetNumThreads(256);
        printf("IPP Threads: %d", ippGetNumThreads((int *)0));
        //printf("Array Size: %" PRIu32 "\n", C_length);
        printf("\n");
    }

    float quickSortTime = 0.0;
    float serialMergeSortTime = 0.0;
    //float serialMergeNoBranchSortTime = 0.0;
    float bitonicMergeRealSortTime = 0.0;
    #ifdef AVX512
    //float avx512SortNoMergePathSerialTime = 0.0;
    //float avx512SortNoMergePathV2SerialTime = 0.0;
    //float avx512SortNoMergePathBranchlessTime = 0.0;
    float avx512SortNoMergePathV2Time = 0.0;
    //float avx512SortNoMergePathavxTime = 0.0;
    float avx512MergeSortTime = 0.0;
    float ippSortTime = 0.0;
    #endif

    for (uint32_t run = 0; run < RUNS; run++) {
            // quickSortTime += testSort<quickSort>(
            //     CUnsorted, C_length,
            //     CSorted, Ct_length,
            //     runs, 64, "Quick Sort", 0, 1);

            // serialMergeSortTime += testSort<iterativeMergeSort<serialMerge>>(
            //     CUnsorted, C_length,
            //     CSorted, Ct_length,
            //     runs, 64, "Merge Sort Standard", 0, 1);
        //
        // if (serialMergeNoBranchSortTime >= 0.0) {
        //     serialMergeNoBranchSortTime += testSort<iterativeMergeSort<serialMergeNoBranch>>(
        //         CUnsorted, C_length,
        //         CSorted, Ct_length,
        //         runs, 64, "Branch Avoiding Sort");
        // }
        //
            // bitonicMergeRealSortTime += testSort<iterativeMergeSort<bitonicMergeReal>>(
            //     CUnsorted, C_length,
            //     CSorted, Ct_length,
            //     runs, 64, "Bitonic Based Merge Sort", 0, 1);

        //
        #ifdef AVX512
        //
        // // if (avx512SortNoMergePathSerialTime >= 0.0) {
        // //     avx512SortNoMergePathSerialTime += testSort<avx512SortNoMergePath<serialMerge>>(
        // //         CUnsorted, C_length,
        // //         CSorted, Ct_length,
        // //         runs, 64, "AVX-512 Sort Without Merge Path Serial");
        // // }
        //
            // if (isPowerOfTwo(C_length)) {
            // avx512SortNoMergePathV2Time += testSort<avx512SortNoMergePathV2<avx512Merge>>(
            //     CUnsorted, C_length,
            //     CSorted, Ct_length,
            //     runs, 64, "AVX-512 Sort Without Merge Path Bitonic", 0, 1);
            // }
        //
        // // if (avx512SortNoMergePathBranchlessTime >= 0.0) {
        // //     avx512SortNoMergePathBranchlessTime += testSort<avx512SortNoMergePath<serialMergeNoBranch>>(
        // //         CUnsorted, C_length,
        // //         CSorted, Ct_length,
        // //         runs, 64, "AVX-512 Sort Without Merge Path Branchless");
        // // }
        // //
            // avx512SortNoMergePathBitonicTime += testSort<avx512SortNoMergePath<bitonicMergeReal>>(
            //     CUnsorted, C_length,
            //     CSorted, Ct_length,
            //     runs, 64, "AVX-512 Sort Without Merge Path Bitonic");

        // //
        // // if (avx512SortNoMergePathavxTime >= 0.0) {
        // //     avx512SortNoMergePathavxTime += testSort<avx512SortNoMergePath<avx512Merge>>(
        // //         CUnsorted, C_length,
        // //         CSorted, Ct_length,
        // //         runs, 64, "AVX-512 Sort Without Merge Path AVX");
        // // }
        //
            // avx512MergeSortTime += testSort<iterativeMergeSort<avx512Merge>>(
            //     CUnsorted, C_length,
            //     CSorted, Ct_length,
            //     runs, 64, "AVX-512 Merge Sort", 0, 1);
        // ippSortTime += testSort<ippSort>(
        //     CUnsorted, C_length,
        //     CSorted, Ct_length,
        //     runs, 64, "Ipp Sort", 1, 1);
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
    //serialMergeNoBranchSortTime /= RUNS;
    bitonicMergeRealSortTime /= RUNS;
    #ifdef AVX512
    // avx512SortNoMergePathSerialTime /= RUNS;
    // avx512SortNoMergePathV2SerialTime /= RUNS;
    // avx512SortNoMergePathBranchlessTime /= RUNS;
    // avx512SortNoMergePathBitonicTime /= RUNS;
    avx512SortNoMergePathV2Time /= RUNS;
    avx512MergeSortTime /= RUNS;
    ippSortTime /= RUNS;
    #endif

    if (OutToFile) {
        //writeToSortOut("Merge Sort Standard", entropy, C_length, serialMergeSortTime);
        //writeToSortOut("Branch Avoiding Sort", entropy, C_length, serialMergeNoBranchSortTime);
        //writeToSortOut("SSE Bitonic Based Merge Sort", entropy, C_length, bitonicMergeRealSortTime);
        #ifdef AVX512
        writeToSortOut("AVX-512 Merge Path Based Sort", entropy, C_length, avx512MergeSortTime);
        // if (isPowerOfTwo(C_length)) {
            //writeToSortOut("AVX-512 Optimized", entropy, C_length, avx512SortNoMergePathV2Time);
        // }
        #endif
        //writeToSortOut("Intel IPP Sort", entropy, C_length, ippSortTime);
        //writeToSortOut("Quick Sort", entropy, C_length, quickSortTime);
    }

    if (!OutToFile) {
        printf("Sorting Results                         :  Elements per Second\n");
        printf("Merge Sort Standard                     :     ");
        if (serialMergeSortTime > 0.0) {
            printfcomma((int)((float)Ct_length/serialMergeSortTime));
        } else if (serialMergeSortTime == 0.0) {
            printf("∞");
        } else {
            printf("N/A");
        }
        printf("\n");
        // printf("Branch Avoiding Sort                    :     ");
        // if (serialMergeNoBranchSortTime > 0.0) {
        //     printfcomma((int)((float)Ct_length/serialMergeNoBranchSortTime));
        // } else if (serialMergeNoBranchSortTime == 0.0) {
        //     printf("∞");
        // } else {
        //     printf("N/A");
        // }
        // printf("\n");
        printf("Bitonic Based Merge Sort                :     ");
        if (bitonicMergeRealSortTime > 0.0) {
            printfcomma((int)((float)Ct_length/bitonicMergeRealSortTime));
        } else if (bitonicMergeRealSortTime == 0.0) {
            printf("∞");
        } else {
            printf("N/A");
        }
        printf("\n");
        #ifdef AVX512

        // printf("AVX-512 Sort Without Merge Path Branchless:     ");
        // if (avx512SortNoMergePathBranchlessTime > 0.0) {
        //     printfcomma((int)((float)Ct_length/avx512SortNoMergePathBranchlessTime));
        // } else if (avx512SortNoMergePathBranchlessTime == 0.0) {
        //     printf("∞");
        // } else {
        //     printf("N/A");
        // }
        // printf("\n");
        // printf("AVX-512 Sort Without Merge Path Bitonic   :     ");
        // if (avx512SortNoMergePathBitonicTime > 0.0) {
        //     printfcomma((int)((float)Ct_length/avx512SortNoMergePathBitonicTime));
        // } else if (avx512SortNoMergePathBitonicTime == 0.0) {
        //     printf("∞");
        // } else {
        //     printf("N/A");
        // }
        // printf("\n");
        // printf("AVX-512 Sort Without Merge Path AVX       :     ");
        // if (avx512SortNoMergePathavxTime > 0.0) {
        //     printfcomma((int)((float)Ct_length/avx512SortNoMergePathavxTime));
        // } else if (avx512SortNoMergePathavxTime == 0.0) {
        //     printf("∞");
        // } else {
        //     printf("N/A");
        // }
        // printf("\n");
        printf("AVX-512 Merge Sort                      :     ");
        if (avx512MergeSortTime > 0.0) {
            printfcomma((int)((float)Ct_length/avx512MergeSortTime));
        } else if (avx512MergeSortTime == 0.0) {
            printf("∞");
        } else {
            printf("N/A");
        }
        printf("\n");
        printf("AVX-512 Hybrid Merge Sort               :     ");
        if (avx512SortNoMergePathV2Time > 0.0) {
            printfcomma((int)((float)Ct_length/avx512SortNoMergePathV2Time));
        } else if (avx512SortNoMergePathV2Time == 0.0) {
            printf("∞");
        } else {
            printf("N/A");
        }
        printf("\n");
        printf("ippSort                                 :     ");
        if (ippSortTime > 0.0) {
            printfcomma((int)((float)Ct_length/ippSortTime));
        } else if (ippSortTime == 0.0) {
            printf("∞");
        } else {
            printf("N/A");
        }
        printf("\n");
        printf("Quick Sort                              :     ");
        if (quickSortTime > 0.0) {
            printfcomma((int)((float)Ct_length/quickSortTime));
        } else if (quickSortTime == 0.0) {
            printf("∞");
        } else {
            printf("N/A");
        }
        printf("\n");
        #endif
        printf("\n\n");


        // printf("Serial Merge           :     %f\n", serialMergeSortTime);
        // printf("Serial Merge Branchless:     %f\n", serialMergeNoBranchSortTime);
        // printf("Bitonic Merge          :     %f\n", bitonicMergeRealSortTime);
        // #ifdef AVX512
        // printf("AVX512 Sort W/O MPS    :     %f\n", avx512SortNoMergePathSerialTime);
        // printf("AVX512 Sort V2 W/O MPS :     %f\n", avx512SortNoMergePathV2SerialTime);
        // // printf("AVX512 Sort W/O MPB    :     %f\n", avx512SortNoMergePathBranchlessTime);
        // // printf("AVX512 Sort W/O MPBi   :     %f\n", avx512SortNoMergePathBitonicTime);
        // // printf("AVX512 Sort W/O MPA    :     %f\n", avx512SortNoMergePathavxTime);
        // printf("AVX512 Merge           :     %f\n", avx512MergeSortTime);
        // #endif
    }
}
#endif

#ifdef PARALLELSORT
void parallelTester(
    vec_t** A, uint64_t A_length,
    vec_t** B, uint64_t B_length,
    vec_t** C, uint64_t C_length,
    vec_t** CSorted, uint64_t Ct_length,
    vec_t** CUnsorted, uint32_t runs)
{
    uint32_t numberOfThreads = 0;
    //Check how many threads are running
    #pragma omp parallel
    {
        #pragma omp single
        {
            numberOfThreads = omp_get_num_threads();
        }
    }

    if (!OutToFile) {
        printf("Parameters\n");
        printf("Entropy: %d\n", entropy);
        printf("Runs: %i\n", runs);
        //printf("Array Size: %" PRIu32 "\n", C_length);
        printf("Number of Threads: %i\n", numberOfThreads);
        printf("\n");
    }

    float serialMergeParallelSortTime = 0.0;
    //float serialMergeNoBranchParallelSortTime = 0.0;
    float bitonicMergeRealParallelSortTime = 0.0;
    #ifdef AVX512
    float avx512MergeParallelSortNewTime = 0.0;
    float avx512MergeParallelSortTime = 0.0;
    float ippParallelTime = 0.0;
    #endif

    // float serialMergeParallelSortTimeMax = FLT_MAX;
    // float serialMergeNoBranchParallelSortTimeMax = FLT_MAX;
    // float bitonicMergeRealParallelSortTimeMax = FLT_MAX;
    // #ifdef AVX512
    // float avx512MergeParallelSortNewTimeMax = FLT_MAX;
    // float avx512MergeParallelSortTimeMax = FLT_MAX;
    // #endif

    //float temp = 0.0;

    for (uint32_t run = 0; run < RUNS; run++) {
            // serialMergeParallelSortTime += testParallelSort<parallelIterativeMergeSort<iterativeMergeSort<serialMerge>,serialMerge>>(
            //                                     CUnsorted, C_length,
            //                                     CSorted, Ct_length,
            //                                     runs, 64, "Standard Parallel Merge Sort");

        // if (serialMergeNoBranchParallelSortTime >= 0.0) {
        //     temp = serialMergeNoBranchParallelSortTime;
        //     serialMergeNoBranchParallelSortTime += testParallelSort<parallelIterativeMergeSort<iterativeMergeSort<serialMergeNoBranch>,serialMergeNoBranch>>(
        //                                                 CUnsorted, C_length,
        //                                                 CSorted, Ct_length,
        //                                                 runs, 64, "Branchless Merge Sort");
        //     temp = serialMergeNoBranchParallelSortTime - temp;
        //     if (temp < serialMergeNoBranchParallelSortTimeMax) {
        //         serialMergeNoBranchParallelSortTimeMax = temp;
        //     }
        // }

            // bitonicMergeRealParallelSortTime += testParallelSort<parallelIterativeMergeSort<iterativeMergeSort<bitonicMergeReal>,bitonicMergeReal>>(
            //                                         CUnsorted, C_length,
            //                                         CSorted, Ct_length,
            //                                         runs, 64, "Bitonic Based Merge Sort");

        #ifdef AVX512
        //if (isPowerOfTwo(C_length)) {
//             avx512MergeParallelSortNewTime += testParallelSort<parallelIterativeMergeSort<avx512SortNoMergePathV2<bitonicMergeReal>,bitonicMergeReal>>(
//                                                 CUnsorted, C_length,
//                                                 CSorted, Ct_length,
//                                                 runs, 64, "AVX-512 Based Merge Sort New");
// //}
        // if (avx512MergeParallelSortTime >= 0.0) {
        //     temp = avx512MergeParallelSortTime;
            // avx512MergeParallelSortTime += testParallelSort<parallelIterativeMergeSort<iterativeMergeSort<avx512Merge>,avx512Merge>>(
            //                                     CUnsorted, C_length,
            //                                     CSorted, Ct_length,
            //                                     runs, 64, "AVX-512 Based Merge Sort");
        //     temp = avx512MergeParallelSortTime - temp;
        //     if (temp < avx512MergeParallelSortTimeMax) {
        //         avx512MergeParallelSortTimeMax = temp;
        //     }
        // }

        // ippParallelTime += testSort<ippSort>(
        //                                     CUnsorted, C_length,
        //                                     CSorted, Ct_length,
        //                                     runs, 64, "IPP Parallel Sort", 1, 1);
        #endif

        insertData(
            A, A_length,
            B, B_length,
            C, C_length,
            CSorted, Ct_length,
            CUnsorted);
        }

        serialMergeParallelSortTime /= RUNS;
        //serialMergeNoBranchParallelSortTime /= RUNS;
        bitonicMergeRealParallelSortTime /= RUNS;
        #ifdef AVX512
        avx512MergeParallelSortNewTime /= RUNS;
        avx512MergeParallelSortTime /= RUNS;
        ippParallelTime /= RUNS;
        #endif

        if (OutToFile) {
            //writeToParallelSortOut("Merge Sort Standard", entropy, C_length, numberOfThreads, serialMergeParallelSortTime);
            //writeToParallelSortOut("Branchless Merge Sort", entropy, C_length, numberOfThreads, serialMergeNoBranchParallelSortTime);
            //writeToParallelSortOut("SSE Bitonic Based Merge Sort", entropy, C_length, numberOfThreads, bitonicMergeRealParallelSortTime);
            #ifdef AVX512
            //writeToParallelSortOut("AVX-512 Merge Path Based Sort", entropy, C_length, numberOfThreads, avx512MergeParallelSortTime);
            //if (isPowerOfTwo(C_length)) {
                writeToParallelSortOut("AVX-512 Hybrid Merge Sort", entropy, C_length, numberOfThreads, avx512MergeParallelSortNewTime);
            //}
            #endif

            // writeToParallelSortOut("Standard Merge Sort Max", entropy, C_length, numberOfThreads, serialMergeParallelSortTimeMax);
            // writeToParallelSortOut("Branchless Merge Sort Max", entropy, C_length, numberOfThreads, serialMergeNoBranchParallelSortTimeMax);
            // writeToParallelSortOut("Bitonic Based Merge Sort Max", entropy, C_length, numberOfThreads, bitonicMergeRealParallelSortTimeMax);
            #ifdef AVX512
            // writeToParallelSortOut("AVX-512 Based Merge Sort Max", entropy, C_length, numberOfThreads, avx512MergeParallelSortTimeMax);
            #endif
        }

        if (!OutToFile) {
            printf("Parallel Sorting Results          :  Elements per Second\n");
            printf("Merge Sort Standard               :     ");
            if (serialMergeParallelSortTime > 0.0) {
                printfcomma((int)((float)Ct_length/serialMergeParallelSortTime));
            } else if (serialMergeParallelSortTime == 0.0) {
                printf("∞");
            } else {
                printf("N/A");
            }
            printf("\n");
            // printf("Branchless Merge Sort   :     ");
            // if (serialMergeNoBranchParallelSortTime > 0.0) {
            //     printfcomma((int)((float)Ct_length/serialMergeNoBranchParallelSortTime));
            // } else if (serialMergeNoBranchParallelSortTime == 0.0) {
            //     printf("∞");
            // } else {
            //     printf("N/A");
            // }
            // printf("\n");
            printf("Bitonic Based Merge Sort           :     ");
            if (bitonicMergeRealParallelSortTime > 0.0) {
                printfcomma((int)((float)Ct_length/bitonicMergeRealParallelSortTime));
            } else if (bitonicMergeRealParallelSortTime == 0.0) {
                printf("∞");
            } else {
                printf("N/A");
            }
            printf("\n");
            #ifdef AVX512
            printf("AVX-512 Hybrid Merge Sort          :     ");
            if (avx512MergeParallelSortNewTime > 0.0) {
                printfcomma((int)((float)Ct_length/avx512MergeParallelSortNewTime));
            } else if (avx512MergeParallelSortNewTime == 0.0) {
                printf("∞");
            } else {
                printf("N/A");
            }
            printf("\n");
            printf("AVX-512 Merge PathBased Merge Sort :     ");
            if (avx512MergeParallelSortTime > 0.0) {
                printfcomma((int)((float)Ct_length/avx512MergeParallelSortTime));
            } else if (avx512MergeParallelSortTime == 0.0) {
                printf("∞");
            } else {
                printf("N/A");
            }
            printf("\n");
            printf("IPP Parallel Sort                  :     ");
            if (ippParallelTime > 0.0) {
                printfcomma((int)((float)Ct_length/ippParallelTime));
            } else if (ippParallelTime == 0.0) {
                printf("∞");
            } else {
                printf("N/A");
            }
            printf("\n");
            #endif
            printf("\n\n");


            // printf("Standard Merge Sort Max     :     ");
            // if (serialMergeParallelSortTime > 0.0) {
            //     printfcomma((int)((float)Ct_length/serialMergeParallelSortTimeMax));
            // } else if (serialMergeParallelSortTime == 0.0) {
            //     printf("∞");
            // } else {
            //     printf("N/A");
            // }
            // printf("\n");
            // printf("Branchless Merge Sort Max   :     ");
            // if (serialMergeNoBranchParallelSortTime > 0.0) {
            //     printfcomma((int)((float)Ct_length/serialMergeNoBranchParallelSortTimeMax));
            // } else if (serialMergeNoBranchParallelSortTime == 0.0) {
            //     printf("∞");
            // } else {
            //     printf("N/A");
            // }
            // printf("\n");
            // printf("Bitonic Based Merge Sort Max:     ");
            // if (bitonicMergeRealParallelSortTime > 0.0) {
            //     printfcomma((int)((float)Ct_length/bitonicMergeRealParallelSortTimeMax));
            // } else if (bitonicMergeRealParallelSortTime == 0.0) {
            //     printf("∞");
            // } else {
            //     printf("N/A");
            // }
            // printf("\n");
            // #ifdef AVX512
            // printf("AVX-512 Based Merge Sort Max:     ");
            // if (avx512MergeParallelSortNewTimeMax > 0.0) {
            //     printfcomma((int)((float)Ct_length/avx512MergeParallelSortNewTimeMax));
            // } else if (avx512MergeParallelSortNewTimeMax == 0.0) {
            //     printf("∞");
            // } else {
            //     printf("N/A");
            // }
            // printf("\n");
            // printf("AVX-512 Based Merge Sort Max:     ");
            // if (avx512MergeParallelSortTime > 0.0) {
            //     printfcomma((int)((float)Ct_length/avx512MergeParallelSortTimeMax));
            // } else if (avx512MergeParallelSortTime == 0.0) {
            //     printf("∞");
            // } else {
            //     printf("N/A");
            // }
            // printf("\n");
            // #endif
            // printf("\n\n");
            //

        }
}
#endif

void MergePathSplitter(
    vec_t * A, uint64_t A_length,
    vec_t * B, uint64_t B_length,
    vec_t * C, uint64_t C_length,
    uint32_t threads, uint32_t* ASplitters, uint32_t* BSplitters)
{
    for (uint32_t i = 0; i <= threads; i++) {
        ASplitters[i] = A_length;
        BSplitters[i] = B_length;
    }

    uint64_t minLength = A_length > B_length ? B_length : A_length;

    for (uint32_t thread=0; thread<threads;thread++)
    {
      uint64_t combinedIndex = thread * (minLength * 2) / threads;
      uint64_t x_top, y_top, x_bottom, current_x, current_y, offset, oldx, oldy;
      x_top = combinedIndex > minLength ? minLength : combinedIndex;
      y_top = combinedIndex > (minLength) ? combinedIndex - (minLength) : 0;
      x_bottom = y_top;

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

int main(int argc, char** argv)
{
    hostParseArgs(argc, argv);

    uint32_t seed = time(0);
    srand(seed);

    if (OutToFile) {
        initMergeFilePointer(&mergeFile);
    }
    if (OutToFile) {
        initParallelMergeFilePointer(&parallelMergeFile);
    }
    if (OutToFile) {
        initSortFilePointer(&sortFile);
    }
    if (OutToFile) {
        initParallelSortFilePointer(&parallelSortFile);
    }

    omp_set_dynamic(0);
    for (uint32_t i = 0; i < testingSizesLength; i++) {
        globalCLength = testingSizes[i];
        globalALength = globalCLength/2;
        globalBLength = globalCLength/2 + globalCLength%2;
        initArrays(
            &globalA, globalALength,
            &globalB, globalBLength,
            &globalC, globalCLength,
            &CSorted, globalCLength,
            &CUnsorted);
        for (uint32_t e = 0; e < testingEntropiesLength; e++) {
            entropy = testingEntropies[e];
            insertData(
                &globalA, globalALength,
                &globalB, globalBLength,
                &globalC, globalCLength,
                &CSorted, globalCLength,
                &CUnsorted);

            // Single Threaded Merge Algorithms
            testAlgo<serialMerge>("Standard", false, false);
            testAlgo<bitonicMergeReal>("Bitonic", false, false);
            testAlgo<avx512Merge>("AVX-512 MP", false, false);

            // Single Threaded Sort Algorithms
            testAlgo<iterativeMergeSort<serialMerge>>("Standard", false, false);
            testAlgo<iterativeMergeSort<bitonicMergeReal>>("Bitonic", false, false);
            testAlgo<avx512SortNoMergePathV2<avx512Merge>>("AVX-512 Optimized", false, false);
            testAlgo<ippSort>("IPP", false, true);
            // // testAlgo<ippSort>("IPP Radix", false, true);
            testAlgo<quickSort>("Quick Sort", false, false);

            for (uint32_t j = 0; j < testingThreadsLength; j++) {
                omp_set_num_threads(testingThreads[j]);

                // Parallel Sort Algorithms
                testAlgo<parallelIterativeMergeSort<iterativeMergeSort<serialMerge>, serialMerge>>("Standard", false, false);
                testAlgo<parallelIterativeMergeSort<iterativeMergeSort<bitonicMergeReal>, bitonicMergeReal>>("Bitonic", false, false);
                testAlgo<parallelIterativeMergeSort<avx512SortNoMergePathV2<avx512Merge>, avx512Merge>>("AVX-512 Optimized", false, false);
            }
        }
        freeGlobalData();
    }
}
