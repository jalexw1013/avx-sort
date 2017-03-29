//#include <ipps.h>
//#include <ippcore.h>
//#include <ippvm.h>
#include <stdio.h>
#include "xmalloc.h"
#include <sys/time.h>
#include <stdint.h>
#include <float.h>
#include <getopt.h>
#include <assert.h>
#include <errno.h>
#include "util.h"
#include <stdlib.h>
#include <omp.h>
#include <time.h>
#include <string.h>
#include <malloc.h>
#include <x86intrin.h>
#include <unistd.h>
#include <inttypes.h>
#include <immintrin.h>

#include "sorts.h"
#include "main.h"

//Functionality parametters
#define MERGE //Coment this out to not test merging functionality
#define SORT //Comment this out to not test sorting functionality

// Random Tuning Parameters
//////////////////////////////
//typedef Ipp32s vec_t;
#define INFINITY_VALUE 1073741824
#define NEGATIVE_INFINITY_VALUE 0

// Function Prototypes
//////////////////////////////
void hostParseArgs(
    int argc, char** argv);
void tester(vec_t**, uint32_t,vec_t**, uint32_t,
            vec_t**, uint32_t,vec_t**, uint32_t, vec_t**);
void initArrays(vec_t** A, uint32_t A_length,
    vec_t** B, uint32_t B_length,
    vec_t** C, uint32_t C_length,
    vec_t** CSorted, uint32_t Ct_length,
    vec_t** CUnsorted);
void insertData(vec_t** A, uint32_t A_length,
    vec_t** B, uint32_t B_length,
    vec_t** C, uint32_t C_length,
    vec_t** CSorted, uint32_t Ct_length,
    vec_t** CUnsorted);
/*void MergePathSplitter( vec_t * A, uint32_t A_length, vec_t * B, uint32_t B_length, vec_t * C, uint32_t C_length,
                uint32_t threads, uint32_t* splitters);*/
void freeGlobalData();
// Global Host Variables
////////////////////////////
vec_t*    globalA;
vec_t*    globalB;
vec_t*    globalC;
vec_t*    CSorted;
vec_t*    CUnsorted;
uint32_t  h_ui_A_length                = 100;
uint32_t  h_ui_B_length                = 100;
uint32_t  h_ui_C_length                = 200; //array to put values in
uint32_t  h_ui_Ct_length               = 200; //for unsorted and sorted
uint32_t  RUNS                         = 1;
uint32_t  entropy                      = 28;

// Host Functions
////////////////////////////
int main(int argc, char** argv)
{
    // parse langths of A and B if user entered
    hostParseArgs(argc, argv);

    initArrays(&globalA, h_ui_A_length,
        &globalB, h_ui_B_length,
        &globalC, h_ui_C_length,
        &CSorted, h_ui_Ct_length,
        &CUnsorted);

    insertData(&globalA, h_ui_A_length,
        &globalB, h_ui_B_length,
        &globalC, h_ui_C_length,
        &CSorted, h_ui_Ct_length,
        &CUnsorted);

    tester(&globalA, h_ui_A_length,
        &globalB, h_ui_B_length,
        &globalC, h_ui_C_length,
        &CSorted, h_ui_Ct_length,
        &CUnsorted);

    freeGlobalData();
    #if defined(SIMD_AVX512)
    printf("TESTTEST\n");
    #endif
}

//---------------------------------------------------------------------
//
// Begin section for feature detection
//
//---------------------------------------------------------------------

/*
 * The following code is taken directly from:
 * https://software.intel.com/en-us/articles/how-to-detect-knl-instruction-support
 * Used for educational/research purposes
 */

 #if defined(__INTEL_COMPILER) && (__INTEL_COMPILER >= 1300)

 #include <immintrin.h>

 int has_intel_knl_features()
 {
   const unsigned long knl_features =
       (_FEATURE_AVX512F | _FEATURE_AVX512ER |
        _FEATURE_AVX512PF | _FEATURE_AVX512CD );
   return _may_i_use_cpu_feature( knl_features );
 }

 #else /* non-Intel compiler */

 #include <stdint.h>
 #if defined(_MSC_VER)
 #include <intrin.h>
 #endif

 void run_cpuid(uint32_t eax, uint32_t ecx, uint32_t* abcd)
 {
 #if defined(_MSC_VER)
   __cpuidex(abcd, eax, ecx);
 #else
   uint32_t ebx, edx;
  #if defined( __i386__ ) && defined ( __PIC__ )
   /* in case of PIC under 32-bit EBX cannot be clobbered */
   __asm__ ( "movl %%ebx, %%edi \n\t cpuid \n\t xchgl %%ebx, %%edi" : "=D" (ebx),
  # else
   __asm__ ( "cpuid" : "+b" (ebx),
  # endif
 		      "+a" (eax), "+c" (ecx), "=d" (edx) );
 	    abcd[0] = eax; abcd[1] = ebx; abcd[2] = ecx; abcd[3] = edx;
 #endif
 }

 int check_xcr0_zmm() {
   uint32_t xcr0;
   uint32_t zmm_ymm_xmm = (7 << 5) | (1 << 2) | (1 << 1);
 #if defined(_MSC_VER)
   xcr0 = (uint32_t)_xgetbv(0);  /* min VS2010 SP1 compiler is required */
 #else
   __asm__ ("xgetbv" : "=a" (xcr0) : "c" (0) : "%edx" );
 #endif
   return ((xcr0 & zmm_ymm_xmm) == zmm_ymm_xmm); /* check if xmm, zmm and zmm state are enabled in XCR0 */
 }

 int has_intel_knl_features() {
   uint32_t abcd[4];
   uint32_t osxsave_mask = (1 << 27); // OSX.
   uint32_t avx2_bmi12_mask = (1 << 16) | // AVX-512F
                              (1 << 26) | // AVX-512PF
                              (1 << 27) | // AVX-512ER
                              (1 << 28);  // AVX-512CD
   run_cpuid( 1, 0, abcd );
   // step 1 - must ensure OS supports extended processor state management
   if ( (abcd[2] & osxsave_mask) != osxsave_mask )
     return 0;
   // step 2 - must ensure OS supports ZMM registers (and YMM, and XMM)
   if ( ! check_xcr0_zmm() )
     return 0;

   return 1;
 }
 #endif /* non-Intel compiler */

 static int can_use_intel_knl_features() {
   static int knl_features_available = -1;
   /* test is performed once */
   if (knl_features_available < 0 )
     knl_features_available = has_intel_knl_features();
   return knl_features_available;
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
    for(int i = 0; i < length; i++) {
        if(output[i] != sortedData[i]) {
            printf("Error: %s Failed To Produce Correct Results.\n", name);
            printf("Index:%d, Given Value:%d, Correct "
            "Value:%d\n", i, output[i], sortedData[i]);
            return 0;
        }
    }
    return 1;
}

void clearArray(vec_t* array, uint32_t length) {
    for (int i = 0; i < length; i++) {
        array[i] = 0;
    }
}

void tester(
    vec_t** A, uint32_t A_length,
    vec_t** B, uint32_t B_length,
    vec_t** C, uint32_t C_length,
    vec_t** CSorted, uint32_t Ct_length,
    vec_t** CUnsorted)
{

    //float qsortTime = 0.0;
    //float singleCoreSort = 0.0;
    //float multiCoreMergeSort = 0.0;

    //int cpus = sysconf(_SC_NPROCESSORS_ONLN);

    printf("Parameters\n");
    printf("Entropy: %d\n", entropy);
    printf("A Size: %" PRIu32 "\n", A_length);
    printf("B Size: %" PRIu32 "\n", B_length);
    printf("C Size: %" PRIu32 "\n", C_length);
    printf("\n");

    //---------------------------------------------------------------------
    //
    // Begin section for merging algorithms
    // Change the type define at the top to easily switch between whether
    // to run this or not.
    //
    //---------------------------------------------------------------------
    #ifdef MERGE

        clearArray((*C), C_length);
        printf("Merging Results:\n");

        /*//Serial Merge
        float* serial = (float*)xcalloc(1, sizeof(float));
        tic_reset();
        serialMerge((*A), A_length, (*B), B_length, (*C), C_length);
        *serial = tic_sincelast();
        verifyOutput((*C), (*CSorted), C_length, "Serial Merge");
        clearArray((*C), C_length);
        printf("Serial Merge:              ");
        printf("%18.10f\n", 1e9*(*serial / (float)(Ct_length)));
        free(serial);

        //Branchless serial merge
        float* serialNoBranch = (float*)xcalloc(1, sizeof(float));
        tic_reset();
        serialMergeNoBranch((*A), A_length, (*B), B_length, (*C), Ct_length);
        *serialNoBranch = tic_sincelast();
        verifyOutput((*C), (*CSorted), C_length, "Serial Merge Branchless");
        clearArray((*C), C_length);
        printf("Serial Merge no Branch:    ");
        printf("%18.10f\n", 1e9*(*serialNoBranch / (float)(Ct_length)));
        free(serialNoBranch);

        //Bitonic
        float* bitonicReal = (float*)xcalloc(1, sizeof(float));
        tic_reset();
        bitonicMergeReal((*A), A_length, (*B), B_length, (*C), Ct_length);
        *bitonicReal = tic_sincelast();
        verifyOutput((*C), (*CSorted), C_length, "Bitonic");
        clearArray((*C), C_length);
        printf("Bitonic Merge Real:        ");
        printf("%18.10f\n", 1e9*(*bitonicReal / (float)(Ct_length)));
        free(bitonicReal);*/

        //#ifdef __INTEL_COMPILER
        //AVX512 Merge (our algorithm)
        if ( can_use_intel_knl_features() ) {
            float* avx512 = (float*)xcalloc(1, sizeof(float));
            tic_reset();
            uint32_t ASplitters[17];
            uint32_t BSplitters[17];
            MergePathSplitter((*A), A_length, (*B), B_length, (*C),
                Ct_length, 16, ASplitters, BSplitters);
            serialMergeAVX512((*A), A_length, (*B), B_length, (*C), Ct_length,
                ASplitters, BSplitters);
            *avx512 = tic_sincelast();
            verifyOutput((*C), (*CSorted), C_length, "AVX512 Merge");
            clearArray((*C), C_length);
            printf("Serial Merge AVX-512:  ");
            printf("%18.10f\n", 1e9*(*avx512 / (float)(Ct_length)));
            free(avx512);
        }
        //#endif

    #endif

    //---------------------------------------------------------------------
    //
    // Begin section for sorting algorithms
    // Change the type define at the top to easily switch between whether
    // to run this or not.
    //
    //---------------------------------------------------------------------
    #ifdef SORT
        printf("\nSorting Results:      Total Time (ms)   Per Element (ns)    Elements per Second\n");

        vec_t* unsortedCopy = xmalloc(Ct_length * sizeof(vec_t));
        memcpy(unsortedCopy, (*CUnsorted), Ct_length * sizeof(vec_t));


        /*#include <unistd.h>
        int threads = sysconf(_SC_NPROCESSORS_ONLN);
        //printf("Number of Threads:%i\n", threads);

        //free(parallelCombo);

        //serialComboSort
        float serialCombo = 0.0;
        tic_reset();
        iterativeNonParallelComboMergeSort(*CUnsorted, Ct_length, threads);
        serialCombo = tic_sincelast();
        verifyOutput((*CUnsorted), (*CSorted), Ct_length, "Serial Combo Sort");
        memcpy(unsortedCopy, (*CUnsorted), Ct_length * sizeof(vec_t));
        printf("Serial Combo Sort:  ");
        printf("   %14.6f", 1000*serialCombo);
        printf("   %16.6f", 1e9*(serialCombo / (float)(Ct_length)));
        printf("   %20.6f", (float)(Ct_length)/serialCombo);
        printf("\n");

        //parallelComboSort
        float parallelCombo = 0.0;
        tic_reset();
        iterativeComboMergeSort(unsortedCopy, Ct_length);
        parallelCombo = tic_sincelast();
        verifyOutput(unsortedCopy, (*CSorted), Ct_length, "Parallel Combo Sort");
        memcpy(unsortedCopy, (*CUnsorted), Ct_length * sizeof(vec_t));
        printf("Parallel Combo Sort:");
        printf("   %14.6f", 1000*parallelCombo);
        printf("   %16.6f", 1e9*(parallelCombo / (float)(Ct_length)));
        printf("   %20.6f", (float)(Ct_length)/parallelCombo);
        printf("\n");

        #ifdef __INTEL_COMPILER
        if ( can_use_intel_knl_features() ) {
            //parallelComboSort
            float parallelCombo512 = 0.0;
            tic_reset();
            iterativeComboMergeSortAVX512(unsortedCopy, Ct_length);
            parallelCombo512 = tic_sincelast();
            verifyOutput(unsortedCopy, (*CSorted), Ct_length, "Parallel Combo Sort AVX512");
            memcpy(unsortedCopy, (*CUnsorted), Ct_length * sizeof(vec_t));
            printf("Parallel AVX512:    ");
            printf("   %14.6f", 1000*parallelCombo512);
            printf("   %16.6f", 1e9*(parallelCombo512 / (float)(Ct_length)));
            printf("   %20.6f", (float)(Ct_length)/parallelCombo512);
            printf("\n");
        }
        #endif*/

        //qsort
        float qsortTime = 0.0;
        tic_reset();
        qsort(*CUnsorted, Ct_length, sizeof(vec_t), hostBasicCompare);
        qsortTime = tic_sincelast();
        verifyOutput((*CUnsorted), (*CSorted), Ct_length, "qsort");
        memcpy( (*CUnsorted), unsortedCopy, Ct_length * sizeof(vec_t));
        printf("qsort:              ");
        printf("   %14.6f", 1000*qsortTime);
        printf("   %16.6f", 1e9*(qsortTime / (float)(Ct_length)));
        printf("   %20.6f", (float)(Ct_length)/qsortTime);
        printf("\n");
        //free(qsortTime);

        /*//paralel quick sort
        float parallelQuickSortTime = 0.0;
        tic_reset();
        quicksort(*CUnsorted, 0, Ct_length - 1);
        parallelQuickSortTime = tic_sincelast();
        verifyOutput((*CUnsorted), (*CSorted), Ct_length, "Parallel Quick Sort");
        memcpy(unsortedCopy, (*CUnsorted), Ct_length * sizeof(vec_t));
        printf("Parallel quicksort:   ");
        printf("%18.10f\n", 1e9*(parallelQuickSortTime / (float)(Ct_length)));
        //free(parallelQuickSortTime);*/

        //#ifdef __INTEL_COMPILER
        if ( can_use_intel_knl_features() ) {
            //simpleIterativeMergeSort
            float simpleIterativeMergeSortTime = 0.0;
            tic_reset();
            simpleIterativeMergeSort(CUnsorted, Ct_length);
            simpleIterativeMergeSortTime = tic_sincelast();
            verifyOutput((*CUnsorted), (*CSorted), Ct_length, "Simple Iterative Merge Sort");
            memcpy( (*CUnsorted), unsortedCopy, Ct_length * sizeof(vec_t));
            printf("Iterative Mergesort:");
            printf("   %14.6f", 1000*simpleIterativeMergeSortTime);
            printf("   %16.6f", 1e9*(simpleIterativeMergeSortTime / (float)(Ct_length)));
            printf("   %20.6f", (float)(Ct_length)/simpleIterativeMergeSortTime);
            printf("\n");
        }
        //#endif

    #endif
}


void MergePathSplitter( vec_t * A, uint32_t A_length, vec_t * B, uint32_t B_length, vec_t * C, uint32_t C_length,
    uint32_t threads, uint32_t* ASplitters, uint32_t* BSplitters) {
  ASplitters[threads] = A_length;
  BSplitters[threads] = B_length;

  for (int thread=0; thread<threads;thread++)
  {
    // uint32_t thread = omp_get_thread_num();
    int32_t combinedIndex = thread * (A_length + B_length) / threads;
    int32_t x_top, y_top, x_bottom, current_x, current_y, offset;
    x_top = combinedIndex > A_length ? A_length : combinedIndex;
    y_top = combinedIndex > (A_length) ? combinedIndex - (A_length) : 0;
    x_bottom = y_top;

    vec_t Ai, Bi;
    while(1) {
        // printf("test\n");
      offset = (x_top - x_bottom) / 2;
      current_y = y_top + offset;
      current_x = x_top - offset;
      if(current_x > A_length - 1 || current_y < 1) {
        Ai = 1;Bi = 0;
      } else {
        //printf("current_x: %i\n", current_x);
        //printf("current_y: %i\n", current_y);
        Ai = A[current_x];Bi = B[current_y - 1];
      }
      //printf("One\n");
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
    //printf("Out\n");
//    #pragma omp barrier

    // uint32_t astop = uip_diagonal_intersections[thread*2+2];
    // uint32_t bstop = uip_diagonal_intersections[thread*2+3];
    // uint32_t ci = current_x + current_y;

    // while(current_x < astop && current_y < bstop) {
    //   C[ci++] = A[current_x] < B[current_y] ? A[current_x++] : B[current_y++];
    // }
    // while(current_x < astop) {
    //   C[ci++] = A[current_x++];
    // }
    // while(current_y < bstop) {
    //   C[ci++] = B[current_y++];
    // }
    //printf("Done With Thread %i\n", thread);
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
    extern int    optind, opterr, optopt;
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
        //        printf("entropy %d\n", entropy);
        //               fprintf(stderr, "Entropy = %d.\n", entropy);
        break;
    }
  }
  h_ui_C_length = h_ui_A_length + h_ui_B_length;
  h_ui_Ct_length = h_ui_C_length;
}
