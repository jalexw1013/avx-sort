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
uint32_t  globalALength;
vec_t*    globalB;
uint32_t  globalBLength;
vec_t*    globalC;
uint32_t  globalCLength;
vec_t*    CSorted;
vec_t*    CUnsorted;

FILE *mergeFile;
FILE *parallelMergeFile;
FILE *sortFile;
FILE *parallelSortFile;

uint32_t  h_ui_A_length                = 0;
uint32_t  h_ui_B_length                = 0;
uint32_t  h_ui_C_length                = 0; //array to put values in
uint32_t  h_ui_Ct_length               = 0; //for unsorted and sorted
uint32_t  RUNS                         = 1;
uint32_t  entropy                      = 28;
uint32_t  OutToFile                    = 1; // 1 if output to file

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
void initArrays(vec_t** A, uint32_t A_length,
          vec_t** B, uint32_t B_length,
          vec_t** C, uint32_t C_length,
          vec_t** CSorted, uint32_t Ct_length,
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

void initFilePointers() {
    char mergeFileName[50];
    sprintf(mergeFileName, "../results/MergeResults.%li.csv", time(0));
    mergeFile = fopen(mergeFileName, "w+");
    fprintf(mergeFile, "Name,Entropy,A Size,B Size,Elements Per Second,Total Time");

    char parallelMergeFileName[50];
    sprintf(parallelMergeFileName, "../results/ParallelMergeResults.%li.csv", time(0));
    parallelMergeFile = fopen(parallelMergeFileName, "w+");
    fprintf(parallelMergeFile, "Name,Entropy,A Size,B Size,Number of Threads,Elements Per Second,Total Time");

    char sortFileName[50];
    sprintf(sortFileName, "../results/SortResults.%li.csv", time(0));
    sortFile = fopen(sortFileName, "w+");
    fprintf(sortFile, "Name,Entropy,C Size,Elements Per Second,Total Time");

    char parallelSortFileName[50];
    sprintf(parallelSortFileName, "../results/ParallelSortResults.%li.csv", time(0));
    parallelSortFile = fopen(parallelSortFileName, "w+");
    fprintf(parallelSortFile, "Name,Entropy,C Size,Number of Threads,Elements Per Second,Total Time");
}

void writeOut(const char* name, uint32_t entropy, uint32_t ASize, uint32_t BSize, uint32_t CSize, uint32_t numThreads, double time, AlgoType algoType) {
    if (algoType == Merge) {
        fprintf(mergeFile, "\n%s,%u,%u,%u,%lu,%.20f", name, entropy, ASize, BSize, (uint64_t)((double)(ASize + BSize)/time), time);
        fflush(mergeFile);
    } else if (algoType == ParallelMerge) {
        fprintf(parallelMergeFile, "\n%s,%u,%u,%u,%u,%lu,%.20f", name, entropy, ASize, BSize, numThreads,(uint64_t)((double)(ASize + BSize)/time), time);
        fflush(parallelMergeFile);
    } else if (algoType == Sort) {
        fprintf(sortFile, "\n%s,%u,%u,%lu,%.20f", name, entropy, CSize, (uint64_t)((double)(CSize)/time), time);
        fflush(sortFile);
    } else if (algoType == ParallelSort) {
        fprintf(parallelSortFile, "\n%s,%u,%u,%u,%lu,%.20f", name, entropy, CSize, numThreads, (uint64_t)((double)(CSize)/time), time);
        fflush(parallelSortFile);
    }
}

int verifyOutput(vec_t* outputU, vec_t* CU, vec_t* sortedDataU, uint32_t length, const char* name, uint32_t numThreads, bool isSigned) {
    if (isSigned) {
    } else {
        vec_t* output = (vec_t*)outputU;
        vec_t* COut = (vec_t*)CU;
        vec_t* sortedData = (vec_t*)sortedDataU;
        for(uint32_t i = 0; i < length; i++) {
            if(output[i] != sortedData[i] && COut[i] != sortedData[i]) {
                printf(ANSI_COLOR_RED "    Error: %s Failed To Produce Correct Results.\n", name);
                printf("    Index:%u, Given Value:%d, Correct "
                "Value:%d, ArraySize: %u NumThreads: %d" ANSI_COLOR_RESET "\n", i, output[i], sortedData[i], length, numThreads);
                return 0;
            }
        }
    }
    return 1;
}

template <AlgoTemplate Algo>
void testAlgo(const char* algoName, bool threadSpawn, bool isSigned, AlgoType algoType) {

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

    // Make sure something goes wrong if thread spawn fails
    uint32_t numberOfThreads = -1;
    #pragma omp parallel
    {
        #pragma omp single
        {
            numberOfThreads = omp_get_num_threads();
        }
    }

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
    algoArgs->ASplitters = ASplitters;
    algoArgs->BSplitters = BSplitters;
    algoArgs->arraySizes = arraySizes;


    //setup timing mechanism
    double time = 0.0;

    // Run Algorithm
    if (threadSpawn) {
        #pragma omp parallel
        {
            #pragma omp single
            {
                numberOfThreads = omp_get_num_threads();
                tic_reset();
            }
            // TODO put for loop for runs in here
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
        printf("JKLDKDLFDJKFJLFKLFJKLJFKLJFKLF\n");
        writeOut(algoName, entropy, globalALength, globalBLength, globalCLength, numberOfThreads, time, algoType);
    }

    // Pad the input string
    // First find length of name
    char a;
    int size = 0;
    int found = 0;
    for (int i = 0; i < 33; i++) {
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
    char name[34];
    for (int i = 0; i < size; i++) {
        name[i] = algoName[i];
    }
    for (int i = size; i < 33; i++) {
        name[i] = ' ';
    }
    name[33] = '\0';

    // Print the results
    printf("%s:     ", name);
    if (time > (double)0.0) {
        printfcomma((uint64_t)((double)globalCLength/time));
    } else if (time == 0.0) {
        printf("∞");
    } else {
        printf("N/A");
    }
    printf("\n");

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

    free(ASplitters);
    free(BSplitters);
    free(arraySizes);
    free(algoArgs);
}

void MergePathSplitter(
    vec_t * A, uint32_t A_length,
    vec_t * B, uint32_t B_length,
    vec_t * C, uint32_t C_length,
    uint32_t threads, uint32_t* ASplitters, uint32_t* BSplitters)
{
    for (uint32_t i = 0; i <= threads; i++) {
        ASplitters[i] = A_length;
        BSplitters[i] = B_length;
    }

    uint64_t minLength = (uint64_t)A_length > (uint64_t)B_length ? (uint64_t)B_length : (uint64_t)A_length;

    for (uint32_t thread=0; thread<threads;thread++)
    {
      uint64_t combinedIndex = (uint64_t)thread * ((uint64_t)minLength * (uint64_t)2) / (uint64_t)threads;
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
    omp_set_dynamic(0);
    hostParseArgs(argc, argv);
    uint32_t seed = time(0);
    srand(seed);
    if (OutToFile) {
        initFilePointers();
    }

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

            printf("\n\nRunning with sizes:\n");
            printf("A Length:%u\n", globalALength);
            printf("B Length:%u\n", globalBLength);
            printf("C Length:%u\n", globalCLength);
            printf("Entropy:%u\n", entropy);
            printf("Runs:%u\n", RUNS);
            printf("\n\n");

            // Single Threaded Merge Algorithms
            printf("Single Threaded Merge Algorithms :  Elements Per Second\n");
            testAlgo<serialMerge>("Standard", false, false, Merge);
            testAlgo<bitonicMergeReal>("Bitonic", false, false, Merge);
            testAlgo<avx512Merge>("AVX-512 MP", false, false, Merge);
            printf("\n");

            // Single Threaded Sort Algorithms
            printf("Single Threaded Sort Algorithms  :  Elements Per Second\n");
            testAlgo<iterativeMergeSort<serialMerge>>("Standard", false, false, Sort);
            testAlgo<iterativeMergeSort<bitonicMergeReal>>("Bitonic", false, false, Sort);
            testAlgo<avx512SortNoMergePathV2<avx512Merge>>("AVX-512 Optimized", false, false, Sort);
            testAlgo<ippSort>("IPP", false, true, Sort);
            // // testAlgo<ippSort>("IPP Radix", false, true);
            testAlgo<quickSort>("Quick Sort", false, false, Sort);
            printf("\n");

            for (uint32_t j = 0; j < testingThreadsLength; j++) {
                omp_set_num_threads(testingThreads[j]);

                // Parallel Merge Algorithms
                printf("Parallel Merge Algorithms        :  Elements Per Second\n");
                testAlgo<parallelMerge<serialMerge>>("Standard", false, false, ParallelMerge);
                testAlgo<parallelMerge<bitonicMergeReal>>("Bitonic", false, false, ParallelMerge);
                testAlgo<parallelMerge<avx512Merge>>("AVX-512 MP", false, false, ParallelMerge);
                printf("\n");

                // Parallel Sort Algorithms
                printf("Parallel Sort Algorithms         :  Elements Per Second\n");
                testAlgo<parallelIterativeMergeSort<iterativeMergeSort<serialMerge>, serialMerge>>("Standard", false, false, ParallelSort);
                testAlgo<parallelIterativeMergeSort<iterativeMergeSort<bitonicMergeReal>, bitonicMergeReal>>("Bitonic", false, false, ParallelSort);
                testAlgo<parallelIterativeMergeSort<avx512SortNoMergePathV2<avx512Merge>, avx512Merge>>("AVX-512 Optimized", false, false, ParallelSort);
                printf("\n");
            }
        }
        freeGlobalData();
    }
}
