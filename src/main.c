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

#include "sorts.h"
#include "main.h"

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
uint32_t  h_ui_A_length                = 9000000;
uint32_t  h_ui_B_length                = 9000000;
uint32_t  h_ui_C_length                = 18000000;
uint32_t  h_ui_Ct_length               = 18000000;
uint32_t  RUNS                         = 1;
uint32_t  entropy                      = 28;

uint32_t lengths[] = {9000000};
uint32_t lengthOfLengths = 1;


#define min(a,b) (a <= b)? a : b
#define max(a,b) (a <  b)? b : a

// Host Functions
////////////////////////////
int main(int argc, char** argv)
{
  // parse langths of A and B if user entered
  hostParseArgs(argc, argv);

  tester(&globalA, h_ui_A_length,
      &globalB, h_ui_B_length,
      &globalC, h_ui_C_length,
      &CSorted, h_ui_Ct_length,
      &CUnsorted);

   freeGlobalData();

}


int hostBasicCompare(const void * a, const void * b) {
  return (int) (*(vec_t *)a - *(vec_t *)b);
}

void Init(vec_t** A, uint32_t A_length,
    vec_t** B, uint32_t B_length,
    vec_t** C, uint32_t C_length,
    vec_t** CSorted, uint32_t Ct_length,
    vec_t** CUnsorted)
{
  *A  = (vec_t*) xmalloc((A_length  + 8) * (sizeof(vec_t)));
  *B  = (vec_t*) xmalloc((B_length  + 8) * (sizeof(vec_t)));
  *C  = (vec_t*) xmalloc((C_length  + 32) * (sizeof(vec_t)));
  *CSorted = (vec_t*) xmalloc((Ct_length + 32) * (sizeof(vec_t)));
  *CUnsorted = (vec_t*) xmalloc((Ct_length + 32) * (sizeof(vec_t)));

   uint32_t seed = time(0);// % 100000000;
//  uint32_t seed = 13503;
  srand(seed);

  for(uint32_t i = 0; i < A_length; ++i) {
    (*A)[i] = rand() % (1 << (entropy - 1));
    (*CUnsorted)[i] = (*A)[i];
  }

  for(uint32_t i = 0; i < B_length; ++i) {
    (*B)[i] = rand() % (1 << (entropy - 1));
   (*CUnsorted)[i+A_length] = (*B)[i];
  }

  qsort(*A, A_length, sizeof(vec_t), hostBasicCompare);
  qsort(*B, B_length, sizeof(vec_t), hostBasicCompare);

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

void reshuffleData(
    vec_t** A, uint32_t A_length,
    vec_t** B, uint32_t B_length,
    vec_t** C, uint32_t C_length,
    vec_t** CSorted, uint32_t Ct_length,
    vec_t** CUnsorted)
{
    freeGlobalData();
    Init(A,A_length,B,B_length,C, C_length,CSorted, Ct_length,CUnsorted);
}

void freeGlobalData() {
      //free(globalA);
      //free(globalB);
      //free(globalC);
      //free(CSorted);
      //free(CUnsorted);
}

void tester(
    vec_t** A, uint32_t A_length,
    vec_t** B, uint32_t B_length,
    vec_t** C, uint32_t C_length,
    vec_t** CSorted, uint32_t Ct_length,
    vec_t** CUnsorted)
{
  Init(A, A_length,
      B, B_length,
      C, C_length,
      CSorted, Ct_length,
      CUnsorted);

    float* serial = xmalloc(lengthOfLengths*sizeof(float));
    float* serialNoBranch = xmalloc(lengthOfLengths*sizeof(float));
    float* bitonicReal = xmalloc(lengthOfLengths*sizeof(float));
    float* intrinsic = xmalloc(lengthOfLengths*sizeof(float));
    float* avx512 = xmalloc(lengthOfLengths*sizeof(float));
    float* avx2 = xmalloc(lengthOfLengths*sizeof(float));
    float* mergenet = xmalloc(lengthOfLengths*sizeof(float));

    float* qsortTime = xmalloc(lengthOfLengths*sizeof(float));
    float* singleCoreSort = xmalloc(lengthOfLengths*sizeof(float));
    float* multiCoreMergeSort = xmalloc(lengthOfLengths*sizeof(float));

    vec_t* Cptr=*C;
  //int j = 0;
  for (int j = 0; j < lengthOfLengths; j++) {

      int cpus = sysconf(_SC_NPROCESSORS_ONLN);
      lengths[j] = cpus * 50 / 2;

      //init the arrays of new lengths
      A_length = lengths[j];
      B_length = lengths[j];
      C_length = A_length + B_length;
      Ct_length = C_length;
      reshuffleData(
          A, A_length,
          B, B_length,
          C, C_length,
          CSorted, Ct_length,
          CUnsorted);
      Cptr=*C;

      /*for (int i = 0; i < RUNS; i++) {

        tic_reset();
        serialMerge(*A, A_length, *B, B_length, *C, C_length);
        serial[j] += tic_sincelast();
        for(int ci=0; ci<C_length; ci++) {Cptr[ci]=0;}

        tic_reset();
        serialMergeNoBranch(*A, A_length, *B, B_length, *C, Ct_length);
        serialNoBranch[j] += tic_sincelast();
        for(int ci=0; ci<C_length; ci++) {Cptr[ci]=0;}

        tic_reset();
        bitonicMergeReal(*A, A_length, *B, B_length, *C, Ct_length);
        bitonicReal[j] +=tic_sincelast();
        for(int ci=0; ci<C_length; ci++) {Cptr[ci]=0;}

        tic_reset();
        serialMergeIntrinsic(*A, A_length, *B, B_length, *C, Ct_length);
        intrinsic[j] +=tic_sincelast();
        for(int ci=0; ci<C_length; ci++) {Cptr[ci]=0;}

        /*tic_reset();
        serialMergeAVX512(*A, A_length, *B, B_length, *C, Ct_length);
        avx512[j] +=tic_sincelast();
        for(int ci=0; ci<C_length; ci++) {Cptr[ci]=0;}

        tic_reset();
        //serialMergeAVX2(*A, A_length, *B, B_length, *C, Ct_length);
        avx2[j] +=tic_sincelast();
        for(int ci=0; ci<C_length; ci++) {Cptr[ci]=0;}

        tic_reset();
        mergeNetwork(*A, A_length, *B, B_length, *C, Ct_length);
        mergenet[j] +=tic_sincelast();
        for(int ci=0; ci<C_length; ci++) {Cptr[ci]=0;}*/

        /*for(int i = 0; i < C_length; ++i) {
            assert(C[i] == globalC[i]);
            if(C[i]!=CSorted[i]) {
                printf("\n %d,%d,%d \n", i,C[i],CSorted[i]);
                return;
            }
        }



      }*/

      /*tic_reset();
      qsort(*CUnsorted, Ct_length, sizeof(uint32_t), uint32Compare);
      qsortTime[j] +=tic_sincelast();*/
      /*for(int i = 0; i < C_length; ++i) {
          //assert((*CUnsorted)[i] == globalC[i]);
          if((*CUnsorted)[i]!=(*CSorted)[i]) {
              printf("\n %d,%d,%d \n", i,(*C)[i],(*CSorted)[i]);
              //return;
          }
      }*/
      //for(int ci=0; ci<Ct_length; ci++) {CUnsorted[ci]=0;}

      /*tic_reset();
      parallelComboSort(*CUnsorted, Ct_length, serialMergeNoBranch);
      singleCoreSort[j] +=tic_sincelast();
      for(int i = 0; i < C_length; ++i) {
          //assert((*CUnsorted)[i] == globalC[i]);
          if((*CUnsorted)[i]!=(*CSorted)[i]) {
              printf("\n %d,%d,%d \n", i,(*C)[i],(*CSorted)[i]);
              //return;
          }
      }
      for(int ci=0; ci<Ct_length; ci++) {CUnsorted[ci]=0;}

      printf("Something\n");*/

      tic_reset();
      parallelComboSort(*CUnsorted, Ct_length, serialMergeNoBranch, cpus);
      multiCoreMergeSort[j] +=tic_sincelast();/*
      for(int i = 0; i < C_length; ++i) {
          //assert((*CUnsorted)[i] == globalC[i]);
          if((*CUnsorted)[i]!=(*CSorted)[i]) {
              printf("\n %d,%d,%d \n", i,(*C)[i],(*CSorted)[i]);
              //return;
          }
      }
      for(int ci=0; ci<Ct_length; ci++) {CUnsorted[ci]=0;}*/

  }

  printf("Parameters\n");
  printf("Entropy: %d\n", entropy);
  printf("Runs: %i\n", RUNS);
  printf("\n");
  printf("Results:\n");
  printf("\n");

  printf("Algorithm              ");
  for (int i = 0; i < lengthOfLengths; i++) {
      printf("%15i", lengths[i]*2);
  }
  printf("\nQsort:                 ");
  for (int i = 0; i < lengthOfLengths; i++) {
      printf("%15.10f", 1e8*(qsortTime[i] / (float)(RUNS*Ct_length)));
  }
  printf("\nSerial Combo Sort:     ");
  for (int i = 0; i < lengthOfLengths; i++) {
      printf("%15.10f", 1e8*(singleCoreSort[i] / (float)(RUNS*Ct_length)));
  }
  printf("\nParallel Combo Sort:   ");
  for (int i = 0; i < lengthOfLengths; i++) {
      printf("%15.10f", 1e8*(multiCoreMergeSort[i] / (float)(RUNS*Ct_length)));
  }
  /*printf("\nSerial Merge:          ");
  for (int i = 0; i < lengthOfLengths; i++) {
      printf("%15.10f", 1e8*(serial[i] / (float)(RUNS*Ct_length)));
  }
  printf("\nSerial Merge no Branch:");
  for (int i = 0; i < lengthOfLengths; i++) {
      printf("%15.10f", 1e8*(serialNoBranch[i] / (float)(RUNS*Ct_length)));
  }
  printf("\nBitonic Merge Real:    ");
  for (int i = 0; i < lengthOfLengths; i++) {
      printf("%15.10f", 1e8*(bitonicReal[i] / (float)(RUNS*Ct_length)));
  }
  printf("\nSerial Merge Intrinsic:");
  for (int i = 0; i < lengthOfLengths; i++) {
      printf("%15.10f", 1e8*(intrinsic[i] / (float)(RUNS*Ct_length)));
  }
  printf("\nSerial Merge AVX2:     ");
  for (int i = 0; i < lengthOfLengths; i++) {
      printf("%15.10f", 1e8*(avx2[i] / (float)(RUNS*Ct_length)));
  }
  printf("\nSerial Merge AVX-512:  ");
  for (int i = 0; i < lengthOfLengths; i++) {
      printf("%15.10f", 1e8*(avx512[i] / (float)(RUNS*Ct_length)));
  }
  printf("\nMerge Network:         ");
  for (int i = 0; i < lengthOfLengths; i++) {
      printf("%15.10f", 1e8*(mergenet[i] / (float)(RUNS*Ct_length)));
  }*/


    //free(serial);
    //free(serialNoBranch);
    //free(bitonicReal);
    //free(intrinsic);
    //free(avx2);
    //free(avx512);
    //free(mergenet);


  /*printf("%7.1lf%11.0lf%10.3lf\n", 100.1, 1221.0, 9348.012);
  printf("%7.1lf%11.0lf%10.3lf\n", 2.3, 211.0, 214.0);


  printf("%d", entropy);
  printf(",%.10f", 1e8*(serial / (float)(RUNS*Ct_length)));
  printf(",%.10f", 1e8*(serialNoBranch / (float)(RUNS*Ct_length)));
  printf(",%.10f", 1e8*(bitonicReal / (float)(RUNS*Ct_length)));
  printf(",%.10f", 1e8*(intrinsic / (float)(RUNS*Ct_length)));
  printf(",%.10f", 1e8*(mergenet / (float)(RUNS*Ct_length)));*/


  printf("\n");
  int32_t swapped,missed;


  vec_t Astam[20]={1,1,1,3,4,4,7,7,8,9,100000,100000,100000,100000,100000,100000,100000,100000,100000,100000};
  vec_t Bstam[20]={2,2,3,3,4,5,6,7,8,9,100000,100000,100000,100000,100000,100000,100000,100000,100000,100000};
  vec_t Cstam[30];

// mergeNetwork(Astam,10,Bstam,10, Cstam,20);
//  mergeNetwork(*A, A_length, *B, B_length, *Ct, Ct_length);

  // for (int ci=0; ci<20; ci++)
  //   printf("%d, ", Cstam[ci]);
  //   // printf("%d, ", (*Ct)[ci]);
  // printf("\n");

//  return;

  return;


}


void MergePathSplitter( vec_t * A, uint32_t A_length, vec_t * B, uint32_t B_length, vec_t * C, uint32_t C_length,
    uint32_t threads, uint32_t* splitters) {
  splitters[threads*2] = A_length;
  splitters[threads*2+1] = B_length;

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
      offset = (x_top - x_bottom) / 2;
      current_y = y_top + offset;
      current_x = x_top - offset;
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
          splitters[thread*2]   = current_x;splitters[thread*2+1] = current_y;
          break;
        } else {//Both zeros
          x_top = current_x - 1;y_top = current_y + 1;
        }
      } else {// Both ones
        x_bottom = current_x + 1;
      }
    }
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
  }
}





#define PRINT_ARRAY(ARR) for (int t=0; t<threads;t++){printf("%10d, ",ARR[t]);}printf("\n");
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
