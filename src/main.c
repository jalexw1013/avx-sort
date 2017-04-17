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

#define ANSI_COLOR_RED     "\x1b[31m"
#define ANSI_COLOR_GREEN   "\x1b[32m"
#define ANSI_COLOR_YELLOW  "\x1b[33m"
#define ANSI_COLOR_BLUE    "\x1b[34m"
#define ANSI_COLOR_MAGENTA "\x1b[35m"
#define ANSI_COLOR_CYAN    "\x1b[36m"
#define ANSI_COLOR_RESET   "\x1b[0m"

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

// Global Host Variables
////////////////////////////
vec_t*    globalA;
vec_t*    globalB;
vec_t*    globalC;
vec_t*    CSorted;
vec_t*    CUnsorted;
uint32_t  h_ui_A_length                = 500000;
uint32_t  h_ui_B_length                = 500000;
uint32_t  h_ui_C_length                = 1000000; //array to put values in
uint32_t  h_ui_Ct_length               = 1000000; //for unsorted and sorted
uint32_t  RUNS                         = 50;
uint32_t  entropy                      = 28;

// Host Functions
////////////////////////////
int main(int argc, char** argv)
{
    // parse langths of A and B if user entered
    hostParseArgs(argc, argv);

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
    uint32_t seed = time(0);
    srand(seed);

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
}

void freeGlobalData() {
      free(globalA);
      free(globalB);
      free(globalC);
      free(CSorted);
      free(CUnsorted);
}

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

template <void (*T)(vec_t*,uint32_t,vec_t*,uint32_t,vec_t*,uint32_t)>
void testMerge(
    vec_t** A, uint32_t A_length,
    vec_t** B, uint32_t B_length,
    vec_t** C, uint32_t C_length,
    vec_t** CSorted,
    uint32_t runs, uint32_t algoID,
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

    //loop the number of runs
    for (uint32_t i = 0; i < runs; i++) {
        //reset timer
        tic_reset();

        //perform actual merge
        T((*A), A_length, (*B), B_length, (*C), C_length);

        //get timing
        time += tic_sincelast();

        //verify output is valid
        verifyOutput((*C), (*CSorted), C_length, algoName);

        //restore original values
        clearArray((*C), C_length);
        memcpy( (*A), ACopy, A_length * sizeof(vec_t));
        memcpy( (*B), BCopy, B_length * sizeof(vec_t));
    }
    printf("%s%i:  ", algoName, algoID);
    printf("%18.10f\n", 1e9*((time/runs) / (float)(C_length)));
}

template <SortTemplate Sort>
void testSort(
    vec_t** CUnsorted, uint32_t C_length,
    vec_t** CSorted, uint32_t Ct_length,
    uint32_t runs, const uint32_t splitNumber,
    const char* algoName) {

    //setup timing mechanism
    float time = 0.0;

    //store old values
    vec_t* unsortedCopy = (vec_t*)xmalloc(Ct_length * sizeof(vec_t));
    memcpy(unsortedCopy, (*CUnsorted), Ct_length * sizeof(vec_t));

    //loop the number of runs
    for (uint32_t i = 0; i < runs; i++) {
        //reset timer
        tic_reset();

        //perform actual sort
        Sort(CUnsorted, C_length, splitNumber);

        //get timing
        time += tic_sincelast();

        //verify output is valid
        verifyOutput((*CUnsorted), (*CSorted), C_length, algoName);

        //restore original values
        memcpy((*CUnsorted), unsortedCopy, Ct_length * sizeof(vec_t));
    }
    printf("%s:  ", algoName);
    printf("   %14.6f", 1000*(time/runs));
    printf("   %16.6f", 1e9*((time/runs) / (float)(Ct_length)));
    printf("   %20f", (float)(Ct_length)/(time/runs));
    printf("\n");
}


void tester(
    vec_t** A, uint32_t A_length,
    vec_t** B, uint32_t B_length,
    vec_t** C, uint32_t C_length,
    vec_t** CSorted, uint32_t Ct_length,
    vec_t** CUnsorted, uint32_t runs)
{
    printf("Parameters\n");
    printf("Entropy: %d\n", entropy);
    #ifdef MERGE
    printf("A Size: %" PRIu32 "\n", A_length);
    printf("B Size: %" PRIu32 "\n", B_length);
    #endif
    #ifdef SORT
    printf("Array Size: %" PRIu32 "\n", C_length);
    #endif
    printf("\n");

    #ifdef MERGE

        serialMerge((*A), A_length, (*B), B_length, (*C), Ct_length);

        testMerge<serialMerge>(
            A, A_length, B, B_length,
            C, Ct_length, CSorted,
            runs, 0, "Serial Merge");

        testMerge<serialMergeNoBranch>(
            A, A_length, B, B_length,
            C, Ct_length, CSorted,
            runs, 0, "Branchless Merge");

        testMerge<bitonicMergeReal>(
            A, A_length, B, B_length,
            C, Ct_length, CSorted,
            runs, 0, "Bitonic Merge");

        #ifdef AVX512
        testMerge<avx512Merge>(
            A, A_length, B, B_length,
            C, Ct_length, CSorted,
            runs, 0, "AVX-512 Merge");
        #endif
    #endif

    #ifdef SORT
        printf("\nSorting Results:                                Total Time (ms)   Per Element (ns)    Elements per Second\n");

        testSort<quickSort>(
            CUnsorted, C_length,
            CSorted, Ct_length,
            runs, 64, "Quick Sort                                 ");

        testSort<recursiveMergeSort<serialMerge>>(
            CUnsorted, C_length,
            CSorted, Ct_length,
            runs, 64, "Recursive Merge Sort Using Serial Merge    ");

        testSort<recursiveMergeSort<serialMergeNoBranch>>(
            CUnsorted, C_length,
            CSorted, Ct_length,
            runs, 64, "Recursive Merge Sort Using Branchless Merge");

        testSort<recursiveMergeSort<bitonicMergeReal>>(
            CUnsorted, C_length,
            CSorted, Ct_length,
            runs, 64, "Recursive Merge Sort Using Bitonic Merge   ");

        testSort<iterativeMergeSort<serialMerge>>(
            CUnsorted, C_length,
            CSorted, Ct_length,
            runs, 64, "Iterative Merge Sort Using Serial Merge    ");

        testSort<iterativeMergeSort<serialMergeNoBranch>>(
            CUnsorted, C_length,
            CSorted, Ct_length,
            runs, 64, "Iterative Merge Sort Using Branchless Merge");

        testSort<iterativeMergeSort<bitonicMergeReal>>(
            CUnsorted, C_length,
            CSorted, Ct_length,
            runs, 64, "Iterative Merge Sort Using Bitonic Merge   ");
    #endif
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

  uint32_t minLength = A_length > B_length ? B_length : A_length;

  for (uint32_t thread=0; thread<threads;thread++)
  {
    // uint32_t thread = omp_get_thread_num();
    uint32_t combinedIndex = thread * (minLength * 2) / threads;
    uint32_t x_top, y_top, x_bottom, current_x, current_y, offset, oldx, oldy;
    x_top = combinedIndex > minLength ? minLength : combinedIndex;
    y_top = combinedIndex > (minLength) ? combinedIndex - (minLength) : 0;
    x_bottom = y_top;

    oldx = -1;
    oldy = -1;

    vec_t Ai, Bi;
    while(1) {
      offset = (x_top - x_bottom) / 2;
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
    {"help"   , no_argument      , 0, 'h'},
    {0        , 0                , 0,  0 }
  };

  while(1) {
    int option_index = 0;
    int c = getopt_long(argc, argv, "A:B:R:E:h",
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
    }
  }
  h_ui_C_length = h_ui_A_length + h_ui_B_length;
  h_ui_Ct_length = h_ui_C_length;
}
