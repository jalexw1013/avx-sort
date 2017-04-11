#include <stdio.h>
#include "xmalloc.h"
#include <stdint.h>
#include <float.h>
#include <assert.h>
#include <errno.h>
#include "util.h"
#include <stdlib.h>
#include <omp.h>
#include <malloc.h>
#include <x86intrin.h>
#include <time.h>
#include <unistd.h>
#include <string.h>

#include "sorts.h"

// _MM_SHUFFLE (z, y, x, w)
// (z<<6) | (y<<4) | (x<<2) | w
// Note that these indices are all reverse because
// an implcit reverse happens during the store, and
// thus we reverse here to avoid the need to reverse
// after we get the results.
// const uint8_t m0110 =          (1<<4) | (1<<2);
// const uint8_t m1010 = (1<<6) |          (1<<2);
// const uint8_t m1100 = (1<<6) | (1<<4);
// const uint8_t m1221 = (1<<6) | (2<<4) | (2<<2) | 1;
// const uint8_t m2121 = (2<<6) | (1<<4) | (2<<2) | 1;
// const uint8_t m2332 = (2<<6) | (3<<4) | (3<<2) | 2;
// const uint8_t m3120 = (3<<6) | (1<<4) | (2<<2) | 0;
// const uint8_t m3232 = (3<<6) | (2<<4) | (3<<2) | 2;
//
// const uint8_t m0123 = (0<<6) | (1<<4) | (2<<2) | 3;
// const uint8_t m0321 = (0<<6) | (3<<4) | (2<<2) | 1;
// const uint8_t m2103 = (2<<6) | (1<<4) | (0<<2) | 3;
// const uint8_t m0213 = (0<<6) | (2<<4) | (1<<2) | 3;
// const uint8_t m1001 = (1<<6)                   | 1;

void serialMerge(
    vec_t* A, uint32_t A_length,
    vec_t* B, uint32_t B_length,
    vec_t* C, uint32_t C_length)
{
  uint32_t Aindex = 0;
  uint32_t Bindex = 0;
  uint32_t Cindex = 0;
  int32_t flag;

  while(Aindex < A_length && Bindex < B_length) {
    C[Cindex++] = A[Aindex] < B[Bindex] ? A[Aindex++] : B[Bindex++];
  }
  while(Aindex < A_length) C[Cindex++] = A[Aindex++];
  while(Bindex < B_length) C[Cindex++] = B[Bindex++];
}



void serialMergeNoBranch(
      vec_t* A, uint32_t A_length,
      vec_t* B, uint32_t B_length,
      vec_t* C, uint32_t C_length)
  {
    uint32_t Aindex = 0;
    uint32_t Bindex = 0;
    uint32_t Cindex = 0;
    int32_t flag;
    //    uint32_t mask,notMask;
    while(Aindex < A_length && Bindex < B_length) {
      flag = ((unsigned int)(A[Aindex] - B[Bindex]) >> 31 ) ;
      C[Cindex++] = (flag)*A[Aindex] + (1-flag)*B[Bindex];
      Aindex +=flag;
      Bindex +=1-flag;
    }
    while(Aindex < A_length) C[Cindex++] = A[Aindex++];
    while(Bindex < B_length) C[Cindex++] = B[Bindex++];
}

/*
 * See Merge Sort From Srinivas's code
 * https://github.com/psombe/sorting
 */

const uint8_t m0110 =          (1<<4) | (1<<2);
const uint8_t m1010 = (1<<6) |          (1<<2);
const uint8_t m1100 = (1<<6) | (1<<4);
const uint8_t m1221 = (1<<6) | (2<<4) | (2<<2) | 1;
const uint8_t m2121 = (2<<6) | (1<<4) | (2<<2) | 1;
const uint8_t m2332 = (2<<6) | (3<<4) | (3<<2) | 2;
const uint8_t m3120 = (3<<6) | (1<<4) | (2<<2) | 0;
const uint8_t m3232 = (3<<6) | (2<<4) | (3<<2) | 2;

const uint8_t m0123 = (0<<6) | (1<<4) | (2<<2) | 3;
const uint8_t m0321 = (0<<6) | (3<<4) | (2<<2) | 1;
const uint8_t m2103 = (2<<6) | (1<<4) | (0<<2) | 3;
const uint8_t m0213 = (0<<6) | (2<<4) | (1<<2) | 3;
const uint8_t m1001 = (1<<6)                   | 1;

#define min(a,b) (a <= b)? a : b
#define max(a,b) (a <  b)? b : a

void quickSort (uint32_t N, vec_t* A)
{
    qsort (A, N, sizeof(vec_t), hostBasicCompare);
}

void bitonicMergeReal(vec_t* A, uint32_t A_length,
                      vec_t* B, uint32_t B_length,
                      vec_t* C, uint32_t C_length){
    long Aindex = 0,Bindex = 0, Cindex = 0;
    int isA, isB;

    __m128i sA = _mm_loadu_si128((const __m128i*)&(A[Aindex]));
    __m128i sB = _mm_loadu_si128((const __m128i*)&(B[Bindex]));
    while ((Aindex < (A_length-4)) && (Bindex < (B_length-4)))
    {
        // load SIMD registers from A and B
        isA = 0;
        isB = 0;
        // reverse B
        sB = _mm_shuffle_epi32(sB, m0123);
        // level 1
        __m128i sL1 = _mm_min_epu32(sA, sB);
        __m128i sH1 = _mm_max_epu32(sA, sB);
        __m128i sL1p = _mm_unpackhi_epi64(sH1, sL1);
        __m128i sH1p = _mm_unpacklo_epi64(sH1, sL1);
        // level 2
        __m128i sL2 = _mm_min_epu32(sH1p, sL1p);
        __m128i sH2 = _mm_max_epu32(sH1p, sL1p);
        __m128i c1010 = _mm_set_epi32(-1, 0, -1, 0);
        __m128i c0101 = _mm_set_epi32(0, -1, 0, -1);
        // use blend
        __m128i sL2p = _mm_or_si128(_mm_and_si128(sL2, c1010), _mm_and_si128(_mm_shuffle_epi32(sH2, m0321), c0101));
        __m128i sH2p = _mm_or_si128(_mm_and_si128(_mm_shuffle_epi32(sL2, m2103), c1010), _mm_and_si128(sH2, c0101));
        // level 3
        __m128i sL3 = _mm_min_epu32(sL2p, sH2p);
        __m128i sH3 = _mm_max_epu32(sL2p, sH2p);
        __m128i sL3p = _mm_shuffle_epi32(_mm_unpackhi_epi64(sH3, sL3), m0213);
        __m128i sH3p = _mm_shuffle_epi32(_mm_unpacklo_epi64(sH3, sL3), m0213);
        // store back data into C from SIMD registers
        _mm_storeu_si128((__m128i*)&(C[Cindex]), sL3p);
        // calculate index for the next run
        sB=sH3p;
        Cindex+=4;
        #if PRINTCMP
        global_count += 24;
        #endif
        if (A[Aindex+4]<B[Bindex+4]){
            Aindex+=4;
            isA = 1;
            sA = _mm_loadu_si128((const __m128i*)&(A[Aindex]));
        }
        else {
            Bindex+=4;
            isB = 1;
            sA = _mm_loadu_si128((const __m128i*)&(B[Bindex]));
        }
    }

    if( isA ) Bindex += 4;
    else Aindex += 4;

    int tempindex = 0;
    int temp_length = 4;
    vec_t temp[4];
    _mm_storeu_si128((__m128i*)temp, sB);

    #if PRINTCMP
    global_count++;
    #endif
    if (temp[3] <= A[Aindex])
    {
        Aindex -= 4;
        _mm_storeu_si128((__m128i*)&(A[Aindex]), sB);
    }
    else
    {
        Bindex -= 4;
        _mm_storeu_si128((__m128i*)&(B[Bindex]), sB);
    }

    while (Cindex < C_length)
    {
        if (Aindex < A_length && Bindex < B_length)
        {
            #if PRINTCMP
            global_count++;
            #endif
            if (A[Aindex] < B[Bindex])
            {
                C[Cindex++] = A[Aindex++];
            }
            else
            {
                C[Cindex++] = B[Bindex++];
            }
        }
        else
        {
            while (Aindex < A_length)
            {
                C[Cindex++] = A[Aindex++];
            }
            while (Bindex < B_length)
            {
                C[Cindex++] = B[Bindex++];
            }
        }
    }

    return;
}

// Removing unnecessary copy of output to input
int ossemergesort(uint32_t N, vec_t* A, vec_t* O)
{
    if(N < 64)
    {
        quickSort(N,A);
        return 0; // 0 - A contains the sorted lists 1 - O contains the sorted list
    }

    long d = N >> 1 ;

    // Recursively sort them
    int first_ret, last_ret;
    first_ret = ossemergesort(d, A, O);
    last_ret = ossemergesort(N - d, A + d, O + d);

    vec_t *ip1,*ip2;

    ip1 = (first_ret == 0)? A : O;
    ip2 = (last_ret == 0)? A : O;

    long i,i1,i2;
    i = 0;
    i1 = 0;
    i2 = d;

    vec_t* op;
    op = (first_ret == 0)? O : A;

    // SSE Merge
    bitonicMergeReal(ip1, d, ip2 + d, N - d, op, N);

    return (first_ret + 1)%2;
}

void sseMergeSortO(uint32_t N, vec_t* A)
{
    int ret;

    vec_t* Aaux = (vec_t *) malloc(sizeof(vec_t)*N);

    ret = ossemergesort(N,A,Aaux);

    if(ret == 1)
    {
        for(long i=0; i < N; i++)
        {
            A[i] = Aaux[i];
        }
    }

    free(Aaux);

    return;
}

void sseMergeSort(uint32_t N, vec_t* A)
{
    sseMergeSortO(N,A);
}

int compare (const void* a, const void* b)
{
    vec_t ka = *(const vec_t *)a;
    vec_t kb = *(const vec_t *)b;
    if (ka < kb)
        return -1;
    else if (ka == kb)
        return 0;
    else
        return 1;
}

// Removing unnecessary copy of output to input
int omergesort(uint32_t N, vec_t* A, vec_t* O)
{
    if(N < 64)
    {
        quickSort(N,A);
        return 0; // 0 - A contains the sorted lists 1 - O contains the sorted list
    }

    uint32_t d = N >> 1 ;

    // Recursively sort them
    int first_ret, last_ret;
    first_ret = omergesort(d, A, O);
    last_ret = omergesort(N - d, A + d, O + d);

    vec_t *ip1,*ip2;

    ip1 = (first_ret == 0)? A : O;
    ip2 = (last_ret == 0)? A : O;

    uint32_t i,i1,i2;
    i = 0;
    i1 = 0;
    i2 = d;

    vec_t* op;
    op = (first_ret == 0)? O : A;

    // Merge them
    for( i = 0; i < N; i++)
    {
        if( i1 < d && i2 < N)
        {
            if(compare((void*)&(ip1[i1]),(void*)&(ip2[i2])) > 0)
            {
                op[i] = ip2[i2];
                i2++;
            }
            else
            {
                op[i] = ip1[i1];
                i1++;
            }
        }
        else if( i1 >= d)
        {
            while(i2 < N)
            {
                op[i] = ip2[i2];
                i2++;
                i++;
            }
        }
        else
        {
            while( i1 < d)
            {
                op[i] = ip1[i1];
                i1++;
                i++;
            }
        }
    }

    return (first_ret + 1)%2;
}

void mergeSortO(uint32_t N, vec_t* A)
{
    int ret;

    vec_t* Aaux = (vec_t *) malloc(sizeof(vec_t)*N);

    ret = omergesort(N,A,Aaux);

    if(ret == 1)
    {
        for(long i=0; i < N; i++)
        {
            A[i] = Aaux[i];
        }
    }

    free(Aaux);

    return;
}

void mergeSortP(uint32_t N, vec_t* A)
{
    int ret;

    vec_t* Aaux = (vec_t *) malloc(sizeof(vec_t)*N);
    omergesort(N,A,Aaux);
    free(Aaux);

    return;
}

void mergeSort(uint32_t N, vec_t* A)
{
    mergeSortO(N,A);
}

// void bitonicMergeReal(vec_t* A, uint32_t A_length,
//                       vec_t* B, uint32_t B_length,
//                       vec_t* C, uint32_t C_length){
//     /*uint32_t Aindex = 0,Bindex = 0, Cindex = 0;
//     int isA, isB;
//     __m128i sA = _mm_loadu_si128((const __m128i*)&(A[Aindex]));
//     __m128i sB = _mm_loadu_si128((const __m128i*)&(B[Bindex]));
//     while ((Aindex < (A_length-4)) && (Bindex < (B_length-4)))
//     {
//         // load SIMD registers from A and B
//         isA = 0;
//         isB = 0;
//         // reverse B
//         sB = _mm_shuffle_epi32(sB, m0123);
//         // level 1
//         __m128i sL1 = _mm_min_epu32(sA, sB);
//         __m128i sH1 = _mm_max_epu32(sA, sB);
//         __m128i sL1p = _mm_unpackhi_epi64(sH1, sL1);
//         __m128i sH1p = _mm_unpacklo_epi64(sH1, sL1);
//         // level 2
//         __m128i sL2 = _mm_min_epu32(sH1p, sL1p);
//         __m128i sH2 = _mm_max_epu32(sH1p, sL1p);
//         __m128i c1010 = _mm_set_epi32(-1, 0, -1, 0);
//         __m128i c0101 = _mm_set_epi32(0, -1, 0, -1);
//         // use blend
//         __m128i sL2p = _mm_or_si128(_mm_and_si128(sL2, c1010), _mm_and_si128(_mm_shuffle_epi32(sH2, m0321), c0101));
//         __m128i sH2p = _mm_or_si128(_mm_and_si128(_mm_shuffle_epi32(sL2, m2103), c1010), _mm_and_si128(sH2, c0101));
//         // level 3
//         __m128i sL3 = _mm_min_epu32(sL2p, sH2p);
//         __m128i sH3 = _mm_max_epu32(sL2p, sH2p);
//         __m128i sL3p = _mm_shuffle_epi32(_mm_unpackhi_epi64(sH3, sL3), m0213);
//         __m128i sH3p = _mm_shuffle_epi32(_mm_unpacklo_epi64(sH3, sL3), m0213);
//         // store back data into C from SIMD registers
//         _mm_storeu_si128((__m128i*)&(C[Cindex]), sL3p);
//         // calculate index for the next run
//         sB=sH3p;
//         Cindex+=4;
//         if (A[Aindex+4]<B[Bindex+4]){
//             Aindex+=4;
//             isA = 1;
//             sA = _mm_loadu_si128((const __m128i*)&(A[Aindex]));
//         }
//         else {
//             Bindex+=4;
//             isB = 1;
//             sA = _mm_loadu_si128((const __m128i*)&(B[Bindex]));
//          }
//       }
//      if( isA ) Bindex += 4;
//     else Aindex += 4;
//
//     int tempindex = 0;
//     int temp_length = 4;
//     vec_t temp[4];
//     _mm_storeu_si128((__m128i*)temp, sB);
//     if (temp[3] <= A[Aindex])
//     {
//         Aindex -= 4;
//         for(int ii=0; ii < 4; ii++)
//         {
//             A[Aindex + ii] = temp[ii];
//         }
//     }
//     else
//     {
//         Bindex -= 4;
//         for(int ii=0; ii < 4; ii++)
//         {
//             B[Bindex + ii] = temp[ii];
//         }
//     }
//     for (Cindex; Cindex < C_length; Cindex++)
//     {
//         if (Aindex < A_length && Bindex < B_length)
//         {
//             if (A[Aindex] < B[Bindex])
//             {
//                 C[Cindex] = A[Aindex];
//                 Aindex++;
//             }
//             else
//             {
//                 C[Cindex] = B[Bindex];
//                 Aindex++;
//             }
//         }
//         else
//         {
//             while (Aindex < A_length)
//             {
//                 C[Cindex] = A[Aindex];
//                 Aindex++;
//                 Cindex++;
//             }
//             while (Bindex < B_length)
//             {
//                 C[Cindex] = B[Bindex];
//                 Bindex++;
//                 Cindex++;
//             }
//         }
//     }
//     return;*/
//       uint32_t Aindex = 0,Bindex = 0, Cindex = 0;
//
//       __m128i sA = _mm_loadu_si128((const __m128i*)&(A[Aindex]));
//       __m128i sB = _mm_loadu_si128((const __m128i*)&(B[Bindex]));
//       while ((Aindex < (A_length-4)) || (Bindex < (B_length-4)))
//       {
//         // load SIMD registers from A and B
//        // reverse B
//         sB = _mm_shuffle_epi32(sB, m0123);
//         // level 1
//         __m128i sL1 = _mm_min_epu32(sA, sB);
//         __m128i sH1 = _mm_max_epu32(sA, sB);
//         __m128i sL1p = _mm_unpackhi_epi64(sH1, sL1);
//         __m128i sH1p = _mm_unpacklo_epi64(sH1, sL1);
//         // level 2
//         __m128i sL2 = _mm_min_epu32(sH1p, sL1p);
//         __m128i sH2 = _mm_max_epu32(sH1p, sL1p);
//         __m128i c1010 = _mm_set_epi32(-1, 0, -1, 0);
//         __m128i c0101 = _mm_set_epi32(0, -1, 0, -1);
//         // use blend
//         __m128i sL2p = _mm_or_si128(_mm_and_si128(sL2, c1010), _mm_and_si128(_mm_shuffle_epi32(sH2, m0321), c0101));
//         __m128i sH2p = _mm_or_si128(_mm_and_si128(_mm_shuffle_epi32(sL2, m2103), c1010), _mm_and_si128(sH2, c0101));
//         // level 3
//         __m128i sL3 = _mm_min_epu32(sL2p, sH2p);
//         __m128i sH3 = _mm_max_epu32(sL2p, sH2p);
//         __m128i sL3p = _mm_shuffle_epi32(_mm_unpackhi_epi64(sH3, sL3), m0213);
//         __m128i sH3p = _mm_shuffle_epi32(_mm_unpacklo_epi64(sH3, sL3), m0213);
//         // store back data into C from SIMD registers
//         _mm_storeu_si128((__m128i*)&(C[Cindex]), sL3p);
//         // calculate index for the next run
//         sB=sH3p;
//         Cindex+=4;
//         if (A[Aindex+4]<B[Bindex+4]){
//         	Aindex+=4;
//         	sA = _mm_loadu_si128((const __m128i*)&(A[Aindex]));
//         }
//         else {
//         	Bindex+=4;
//         	sA = _mm_loadu_si128((const __m128i*)&(B[Bindex]));
//         }
//      }
// }

#define BYTE_TO_BINARY_PATTERN "%c%c%c%c%c%c%c%c"
#define BYTE_TO_BINARY(byte)  \
  (byte & 0x80 ? '1' : '0'), \
  (byte & 0x40 ? '1' : '0'), \
  (byte & 0x20 ? '1' : '0'), \
  (byte & 0x10 ? '1' : '0'), \
  (byte & 0x08 ? '1' : '0'), \
  (byte & 0x04 ? '1' : '0'), \
  (byte & 0x02 ? '1' : '0'), \
  (byte & 0x01 ? '1' : '0')

void print256_num(__m256i var)
{
    uint32_t *val = (uint32_t*) &var;
    printf("Numerical: %i %i %i %i %i %i %i %i \n",
           val[0], val[1], val[2], val[3], val[4], val[5],
           val[6], val[7]);
}
void print512_num(char *text, __m512i var)
{
    uint32_t *val = (uint32_t*) &var;
    printf("%s: %i %i %i %i %i %i %i %i %i %i %i %i %i %i %i %i\n", text,
           val[0], val[1], val[2], val[3], val[4], val[5], val[6], val[7],
           val[8], val[9], val[10], val[11], val[12], val[13], val[14], val[15]);
}
void print16intarray(char *text, int *val) {
    printf("%s: %i %i %i %i %i %i %i %i %i %i %i %i %i %i %i %i\n", text,
           val[0], val[1], val[2], val[3], val[4], val[5], val[6], val[7],
           val[8], val[9], val[10], val[11], val[12], val[13], val[14], val[15]);
}
void printmmask16(char *text, __mmask16 mask) {
    //uint16_t *val = (uint16_t*) &mask;
    //printf("%s: %i %i %i %i %i %i %i %i %i %i %i %i %i %i %i %i\n", text,
           //val[0], val[1], val[2], val[3], val[4], val[5], val[6], val[7],
           //val[8], val[9], val[10], val[11], val[12], val[13], val[14], val[15]);
    //uint16_t num = (uint16_t)mask;
    printf("%s: "BYTE_TO_BINARY_PATTERN" "BYTE_TO_BINARY_PATTERN"\n",text, BYTE_TO_BINARY(mask>>8), BYTE_TO_BINARY(mask));
    //printf("%s: %i\n", text, num);
}

#ifdef AVX512
void serialMergeAVX512(
    vec_t* A, uint32_t A_length,
    vec_t* B, uint32_t B_length,
    vec_t* C, uint32_t C_length)
{

        uint32_t ASplitters[17];
        uint32_t BSplitters[17];
        MergePathSplitter(A, A_length, B, B_length, C,
            C_length, 16, ASplitters, BSplitters);

        //start indexes
        __m512i vindexA = _mm512_set_epi32(ASplitters[15], ASplitters[14],
                                           ASplitters[13], ASplitters[12],
                                           ASplitters[11], ASplitters[10],
                                           ASplitters[9], ASplitters[8],
                                           ASplitters[7], ASplitters[6],
                                           ASplitters[5], ASplitters[4],
                                           ASplitters[3], ASplitters[2],
                                           ASplitters[1], ASplitters[0]);
        __m512i vindexB = _mm512_set_epi32(BSplitters[15], BSplitters[14],
                                           BSplitters[13], BSplitters[12],
                                           BSplitters[11], BSplitters[10],
                                           BSplitters[9], BSplitters[8],
                                           BSplitters[7], BSplitters[6],
                                           BSplitters[5], BSplitters[4],
                                           BSplitters[3], BSplitters[2],
                                           BSplitters[1], BSplitters[0]);
        //stop indexes
        __m512i vindexAStop = _mm512_set_epi32(ASplitters[16],
                                           ASplitters[15], ASplitters[14],
                                           ASplitters[13], ASplitters[12],
                                           ASplitters[11], ASplitters[10],
                                           ASplitters[9], ASplitters[8],
                                           ASplitters[7], ASplitters[6],
                                           ASplitters[5], ASplitters[4],
                                           ASplitters[3], ASplitters[2],
                                           ASplitters[1]);
        __m512i vindexBStop = _mm512_set_epi32(BSplitters[16],
                                           BSplitters[15], BSplitters[14],
                                           BSplitters[13], BSplitters[12],
                                           BSplitters[11], BSplitters[10],
                                           BSplitters[9], BSplitters[8],
                                           BSplitters[7], BSplitters[6],
                                           BSplitters[5], BSplitters[4],
                                           BSplitters[3], BSplitters[2],
                                           BSplitters[1]);
        //vindex start
        __m512i vindexC = _mm512_add_epi32(vindexA, vindexB);

        //other Variables
        const __m512i mizero = _mm512_set_epi32(0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0);
        const __m512i mione = _mm512_set_epi32(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1);
        const __m512i minegone = _mm512_set_epi32(-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1);

        __mmask16 exceededAStop = _mm512_cmpgt_epi32_mask(vindexAStop, vindexA);
        __mmask16 exceededBStop = _mm512_cmpgt_epi32_mask(vindexBStop, vindexB);

        while ((exceededAStop | exceededBStop) != 0) {
            //get the current elements
            __m512i miAelems = _mm512_mask_i32gather_epi32(mizero, exceededAStop, vindexA, (const int *)A, 4);
            __m512i miBelems = _mm512_mask_i32gather_epi32(mizero, exceededBStop, vindexB, (const int *)B, 4);

            //compare the elements
            __mmask16 micmp = _mm512_cmple_epi32_mask(miAelems, miBelems);
            micmp = (~exceededBStop | (micmp & exceededAStop));

            //copy the elements to the final elements
            __m512i miCelems = _mm512_mask_add_epi32(miBelems, micmp, miAelems, mizero);
            _mm512_mask_i32scatter_epi32((int *)C, exceededAStop | exceededBStop, vindexC, miCelems, 4);

            //increase indexes
            vindexA = _mm512_mask_add_epi32(vindexA, exceededAStop & micmp, vindexA, mione);
            vindexB = _mm512_mask_add_epi32(vindexB, exceededBStop & ~micmp, vindexB, mione);
            exceededAStop = _mm512_cmpgt_epi32_mask(vindexAStop, vindexA);
            exceededBStop = _mm512_cmpgt_epi32_mask(vindexBStop, vindexB);
            vindexC = _mm512_mask_add_epi32(vindexC, exceededAStop | exceededBStop, vindexC, mione);
        }
}
#endif

// inline void iterativeComboMergeSort(vec_t* array, uint32_t array_length/*, void(*mergeFunction)(vec_t*,int32_t,vec_t*,int32_t,vec_t*,uint32_t)*/)
// {
//         vec_t* C = (vec_t*)xcalloc((array_length), sizeof(vec_t));
//         //uint32_t initialSubArraySize = (array_length % omp_get_num_threads()) ? (array_length / omp_get_num_threads()) + 1 : (array_length / omp_get_num_threads());
//         #pragma omp parallel
//         {
//             //parallelComboMergeSortParallelHelper(array, array_length, omp_get_num_threads(), , , C/*, mergeFunction*/);
//             uint32_t threadNum = omp_get_thread_num();
//             uint32_t initialSubArraySize = (array_length % omp_get_num_threads()) ? (array_length / omp_get_num_threads()) + 1 : (array_length / omp_get_num_threads());
//             uint32_t start,stop;
//             uint32_t i = threadNum*initialSubArraySize;
//             start=i;
//             stop=start+(i + initialSubArraySize < array_length)?initialSubArraySize:(array_length - i);
//             // printf("%d %d %d %d\n",threadNum,start, stop, initialSubArraySize);
//             //return;
//
//             qsort(array + start,   stop, sizeof(vec_t), hostBasicCompare);
//
//             #pragma omp barrier
//             uint32_t currentSubArraySize = initialSubArraySize;
//             while (currentSubArraySize < array_length) {
//                 if(threadNum==0)
//                     printf("*");
//                 //merge one
//                 uint32_t A_start = threadNum * 2 * currentSubArraySize;
//                 if (A_start < array_length - 1) {
//                     uint32_t B_start = min(A_start + currentSubArraySize - 1, array_length - 1);
//                     uint32_t B_end = min(A_start + 2 * currentSubArraySize - 1, array_length - 1);
//                     uint32_t A_length = B_start - A_start + 1;
//                     uint32_t B_length = B_end - B_start;
//                     bitonicMergeReal(array + A_start, A_length, array + B_start + 1, B_length, C + A_start, A_length + B_length);
//                 }
//                 currentSubArraySize = 2 * currentSubArraySize;
//                 #pragma omp barrier
//                 #pragma omp single
//                 {
//                     if (currentSubArraySize >= array_length) {
//                         memcpy(array, C, array_length * sizeof(vec_t));
//                     }
//                 }
//                 #pragma omp barrier
//                 if (currentSubArraySize >= array_length) {
//                     break;
//                 }
//                 A_start = threadNum * 2 * currentSubArraySize;
//                 if (A_start < array_length - 1) {
//                     uint32_t B_start = min(A_start + currentSubArraySize - 1, array_length - 1);
//                     uint32_t B_end = min(A_start + 2 * currentSubArraySize - 1, array_length - 1);
//                     uint32_t A_length = B_start - A_start + 1;
//                     uint32_t B_length = B_end - B_start;
//                     bitonicMergeReal(C + A_start, A_length, C + B_start + 1, B_length, array + A_start, A_length + B_length);
//                 }
//                 currentSubArraySize = 2 * currentSubArraySize;
//                 #pragma omp barrier
//             }
//         }
//
//         free(C);
// }

#ifdef __INTEL_COMPILER
#ifdef AVX512
inline void iterativeComboMergeSortAVX512(vec_t* array, uint32_t array_length/*, void(*mergeFunction)(vec_t*,int32_t,vec_t*,int32_t,vec_t*,uint32_t)*/)
{
        vec_t* C = (vec_t*)xcalloc((array_length), sizeof(vec_t));
        //uint32_t initialSubArraySize = (array_length % omp_get_num_threads()) ? (array_length / omp_get_num_threads()) + 1 : (array_length / omp_get_num_threads());
        #pragma omp parallel
        {
            //parallelComboMergeSortParallelHelper(array, array_length, omp_get_num_threads(), , , C/*, mergeFunction*/);
            uint32_t threadNum = omp_get_thread_num();
            uint32_t initialSubArraySize = (array_length % omp_get_num_threads()) ? (array_length / omp_get_num_threads()) + 1 : (array_length / omp_get_num_threads());
            uint32_t i = threadNum*initialSubArraySize;
            qsort(
                array + i,
                (i + initialSubArraySize < array_length)?initialSubArraySize:(array_length - i),
                sizeof(vec_t), hostBasicCompare);
            #pragma omp barrier
            uint32_t currentSubArraySize = initialSubArraySize;
            while (currentSubArraySize < array_length) {
                //merge one
                uint32_t A_start = threadNum * 2 * currentSubArraySize;
                if (A_start < array_length - 1) {
                    uint32_t B_start = min(A_start + currentSubArraySize - 1, array_length - 1);
                    uint32_t B_end = min(A_start + 2 * currentSubArraySize - 1, array_length - 1);
                    uint32_t A_length = B_start - A_start + 1;
                    uint32_t B_length = B_end - B_start;
                    uint32_t ASplitters[17];
                    uint32_t BSplitters[17];
                    MergePathSplitter(array + A_start, A_length, array + B_start + 1, B_length, C + A_start,
                        A_length + B_length, 16, ASplitters, BSplitters);
                    serialMergeAVX512(array + A_start, A_length, array + B_start + 1, B_length, C + A_start, A_length + B_length,
                        ASplitters, BSplitters);
                }
                currentSubArraySize = 2 * currentSubArraySize;
                #pragma omp barrier
                #pragma omp single
                {
                    if (currentSubArraySize >= array_length) {
                        memcpy(array, C, array_length * sizeof(vec_t));
                    }
                }
                #pragma omp barrier
                if (currentSubArraySize >= array_length) {
                    break;
                }
                A_start = threadNum * 2 * currentSubArraySize;
                if (A_start < array_length - 1) {
                    uint32_t B_start = min(A_start + currentSubArraySize - 1, array_length - 1);
                    uint32_t B_end = min(A_start + 2 * currentSubArraySize - 1, array_length - 1);
                    uint32_t A_length = B_start - A_start + 1;
                    uint32_t B_length = B_end - B_start;
                    uint32_t ASplitters[17];
                    uint32_t BSplitters[17];
                    MergePathSplitter(C + A_start, A_length, C + B_start + 1, B_length, array + A_start,
                        A_length + B_length, 16, ASplitters, BSplitters);
                    serialMergeAVX512(C + A_start, A_length, C + B_start + 1, B_length, array + A_start, A_length + B_length,
                        ASplitters, BSplitters);                }
                currentSubArraySize = 2 * currentSubArraySize;
                #pragma omp barrier
            }
        }

        free(C);
}
#endif
#endif

inline void iterativeComboMergeSortTemp(vec_t* array, uint32_t array_length)
{
        vec_t* C = (vec_t*)xcalloc((array_length), sizeof(vec_t));
        //uint32_t initialSubArraySize = (array_length % omp_get_num_threads()) ? (array_length / omp_get_num_threads()) + 1 : (array_length / omp_get_num_threads());
        #pragma omp parallel
        {
            //parallelComboMergeSortParallelHelper(array, array_length, omp_get_num_threads(), , , C/*, mergeFunction*/);
            uint32_t threadNum = omp_get_thread_num();
            uint32_t initialSubArraySize = (array_length % omp_get_num_threads()) ? (array_length / omp_get_num_threads()) + 1 : (array_length / omp_get_num_threads());
            uint32_t i = threadNum*initialSubArraySize;
            qsort(
                array + i,
                (i + initialSubArraySize < array_length)?initialSubArraySize:(array_length - i),
                sizeof(vec_t), hostBasicCompare);
            #pragma omp barrier
            uint32_t currentSubArraySize = initialSubArraySize;
            while (currentSubArraySize < array_length) {
                //merge one
                uint32_t A_start = threadNum * 2 * currentSubArraySize;
                if (A_start < array_length - 1) {
                    uint32_t B_start = min(A_start + currentSubArraySize - 1, array_length - 1);
                    uint32_t B_end = min(A_start + 2 * currentSubArraySize - 1, array_length - 1);
                    uint32_t A_length = B_start - A_start + 1;
                    uint32_t B_length = B_end - B_start;
                    uint32_t ASplitters[17];
                    uint32_t BSplitters[17];
                    serialMerge(array + A_start, A_length, array + B_start + 1, B_length, C + A_start, A_length + B_length);
                }
                currentSubArraySize = 2 * currentSubArraySize;
                #pragma omp barrier
                #pragma omp single
                {
                    if (currentSubArraySize >= array_length) {
                        memcpy(array, C, array_length * sizeof(vec_t));
                    }
                }
                #pragma omp barrier
                if (currentSubArraySize >= array_length) {
                    break;
                }
                A_start = threadNum * 2 * currentSubArraySize;
                if (A_start < array_length - 1) {
                    uint32_t B_start = min(A_start + currentSubArraySize - 1, array_length - 1);
                    uint32_t B_end = min(A_start + 2 * currentSubArraySize - 1, array_length - 1);
                    uint32_t A_length = B_start - A_start + 1;
                    uint32_t B_length = B_end - B_start;
                    uint32_t ASplitters[17];
                    uint32_t BSplitters[17];
                    serialMerge(array + A_start, A_length, array + B_start + 1, B_length, C + A_start, A_length + B_length);
                                    }
                currentSubArraySize = 2 * currentSubArraySize;
                #pragma omp barrier
            }
        }

        free(C);
}

inline void iterativeNonParallelComboMergeSort(vec_t* array, uint32_t array_length, uint32_t numThreads)
{
    assert(numThreads != 0);
    vec_t* C = (vec_t*)xcalloc((array_length), sizeof(vec_t));

    uint32_t initialSubArraySize = array_length / numThreads;
    if (array_length % numThreads != 0) initialSubArraySize++;

    //sort one array per thread
    for (int i = 0; i < array_length; i += initialSubArraySize) {
        qsort(
            array + i,
            (i + initialSubArraySize < array_length)?initialSubArraySize:(array_length - i),
            sizeof(vec_t), hostBasicCompare);
    }

    //Now merge up the sorted arrays
    for (uint32_t currentSubArraySize = initialSubArraySize; currentSubArraySize < array_length; currentSubArraySize = 2 * currentSubArraySize)
    {
        //Merge from array into C
    	for (uint32_t A_start = 0; A_start < array_length - 1; A_start += 2 * currentSubArraySize)
    	{
    		uint32_t B_start = min(A_start + currentSubArraySize - 1, array_length - 1);
    		uint32_t B_end = min(A_start + 2 * currentSubArraySize - 1, array_length - 1);
            uint32_t A_length = B_start - A_start + 1;
            uint32_t B_length = B_end - B_start;
            serialMergeNoBranch(array + A_start, A_length, array + B_start + 1, B_length, C + A_start, A_length + B_length);
    	}

        //if done merging, copy elements from C back into Array
        currentSubArraySize = 2 * currentSubArraySize;
        if (currentSubArraySize >= array_length) {
            memcpy(array, C, array_length * sizeof(vec_t));
            break;
        }

        //merge from C back into Array
        for (uint32_t A_start = 0; A_start < array_length - 1; A_start += 2 * currentSubArraySize)
    	{
    		uint32_t B_start = min(A_start + currentSubArraySize - 1, array_length - 1);
    		uint32_t B_end = min(A_start + 2 * currentSubArraySize - 1, array_length - 1);
            uint32_t A_length = B_start - A_start + 1;
            uint32_t B_length = B_end - B_start;
            serialMergeNoBranch(C + A_start, A_length, C + B_start + 1, B_length, array + A_start, A_length + B_length);
    	}
    }
    free(C);
}

/**
 * Parallel Quicksort taken from here:
 * http://stackoverflow.com/questions/16007640/openmp-parallel-quicksort
 * Used for comparison
 */
uint32_t Parallelpartition(uint32_t * lt, uint32_t * gt, uint32_t * a, uint32_t p, uint32_t r)
{
    uint32_t i = 0;
    uint32_t j;
    uint32_t key = a[r];
    uint32_t lt_n = 0;
    uint32_t gt_n = 0;

    #pragma omp parallel for
    for(i = p; i < r; i++){
        if(a[i] < a[r]){
            lt[lt_n++] = a[i];
        }else{
            gt[gt_n++] = a[i];
        }
    }

    for(i = 0; i < lt_n; i++){
        a[p + i] = lt[i];
    }

    a[p + lt_n] = key;

    for(j = 0; j < gt_n; j++){
        a[p + lt_n + j + 1] = gt[j];
    }

    return p + lt_n;
}

void Paralelquicksort(uint32_t * a, uint32_t p, uint32_t r)
{
    uint32_t div;
    if(p < r){
        uint32_t * lt = (uint32_t*)xcalloc(r-p, sizeof(uint32_t));
        uint32_t * gt = (uint32_t*)xcalloc(r-p, sizeof(uint32_t));
        div = Parallelpartition(lt, gt, a, p, r);
        free(lt);
        free(gt);
        #pragma omp parallel sections
        {
            #pragma omp section
            Paralelquicksort(a, p, div - 1);
            #pragma omp section
            Paralelquicksort(a, div + 1, r);

        }
    }
}

////////////////////////////////////////////////////////////////////////////////
//
// New section for single core sorting algorithms
//
////////////////////////////////////////////////////////////////////////////////

void simpleIterativeMergeSort(vec_t** array, uint32_t array_length) {
    vec_t* C = (vec_t*)xcalloc((array_length + 8), sizeof(vec_t));
    uint32_t * ASplitters = (vec_t*)xcalloc((17), sizeof(vec_t));
    uint32_t * BSplitters = (vec_t*)xcalloc((17), sizeof(vec_t));

    for (uint32_t currentSubArraySize = 1; currentSubArraySize < array_length; currentSubArraySize = 2 * currentSubArraySize)
    {
        //tic_reset();
    	for (uint32_t A_start = 0; A_start < array_length - 1; A_start += 2 * currentSubArraySize)
    	{
    		uint32_t B_start = min(A_start + currentSubArraySize - 1, array_length - 1);
    		uint32_t B_end = min(A_start + 2 * currentSubArraySize - 1, array_length - 1);
            uint32_t A_length = B_start - A_start + 1;
            uint32_t B_length = B_end - B_start;

            serialMerge((*array) + A_start, A_length, (*array) + B_start + 1, B_length, C + A_start, A_length + B_length);
    	}
        //float tmpF = tic_sincelast();
        //printf("Time at Size %i : %i\n", currentSubArraySize, tmpF);

        //pointer swap for C
        vec_t* tmp = *array;
        *array = C;
        C = tmp;
    }
    free(C);
    free(ASplitters);
    free(BSplitters);
}

void iterativeMergeSortAVX512(vec_t** array, uint32_t array_length) {
    vec_t* C = (vec_t*)xcalloc((array_length + 8), sizeof(vec_t));
    uint32_t * ASplitters = (vec_t*)xcalloc((17), sizeof(vec_t));
    uint32_t * BSplitters = (vec_t*)xcalloc((17), sizeof(vec_t));

    for (uint32_t currentSubArraySize = 1; currentSubArraySize < array_length; currentSubArraySize = 2 * currentSubArraySize)
    {
        //tic_reset();
    	for (uint32_t A_start = 0; A_start < array_length - 1; A_start += 2 * currentSubArraySize)
    	{
    		uint32_t B_start = min(A_start + currentSubArraySize - 1, array_length - 1);
    		uint32_t B_end = min(A_start + 2 * currentSubArraySize - 1, array_length - 1);
            uint32_t A_length = B_start - A_start + 1;
            uint32_t B_length = B_end - B_start;

            // MergePathSplitter((*array) + A_start, A_length, (*array) + B_start + 1, B_length, C + A_start, A_length + B_length, 16, ASplitters, BSplitters);
            // serialMergeAVX512((*array) + A_start, A_length, (*array) + B_start + 1, B_length, C + A_start, A_length + B_length, ASplitters, BSplitters);
    	}
        //float tmpF = tic_sincelast();
        //printf("Time at Size %i : %i\n", currentSubArraySize, tmpF);

        //pointer swap for C
        vec_t* tmp = *array;
        *array = C;
        C = tmp;
    }
    free(C);
    free(ASplitters);
    free(BSplitters);
}

void iterativeMergeSortAVX512Modified(vec_t** array, uint32_t array_length) {
    vec_t* C = (vec_t*)xcalloc((array_length + 8), sizeof(vec_t));
    uint32_t * ASplitters = (vec_t*)xcalloc((17), sizeof(vec_t));
    uint32_t * BSplitters = (vec_t*)xcalloc((17), sizeof(vec_t));

    for (uint32_t currentSubArraySize = 1; currentSubArraySize < array_length; currentSubArraySize = 2 * currentSubArraySize)
    {
    	for (uint32_t A_start = 0; A_start < array_length - 1; A_start += 2 * currentSubArraySize)
    	{
    		uint32_t B_start = min(A_start + currentSubArraySize - 1, array_length - 1);
    		uint32_t B_end = min(A_start + 2 * currentSubArraySize - 1, array_length - 1);
            uint32_t A_length = B_start - A_start + 1;
            uint32_t B_length = B_end - B_start;

            if (currentSubArraySize > 64) {
                // MergePathSplitter((*array) + A_start, A_length, (*array) + B_start + 1, B_length, C + A_start, A_length + B_length, 16, ASplitters, BSplitters);
                // serialMergeAVX512((*array) + A_start, A_length, (*array) + B_start + 1, B_length, C + A_start, A_length + B_length, ASplitters, BSplitters);
            } else {
                serialMerge((*array) + A_start, A_length, (*array) + B_start + 1, B_length, C + A_start, A_length + B_length);
            }
    	}

        //pointer swap for C
        vec_t* tmp = *array;
        *array = C;
        C = tmp;
    }
    free(C);
    free(ASplitters);
    free(BSplitters);
}

void iterativeMergeSortAVX512Modified2(vec_t** array, uint32_t array_length) {
    vec_t* C = (vec_t*)xcalloc((array_length + 8), sizeof(vec_t));
    uint32_t * ASplitters = (vec_t*)xcalloc((17), sizeof(vec_t));
    uint32_t * BSplitters = (vec_t*)xcalloc((17), sizeof(vec_t));

    for (uint32_t currentSubArraySize = 1; currentSubArraySize < array_length; currentSubArraySize = 2 * currentSubArraySize)
    {
    	for (uint32_t A_start = 0; A_start < array_length - 1; A_start += 2 * currentSubArraySize)
    	{
    		uint32_t B_start = min(A_start + currentSubArraySize - 1, array_length - 1);
    		uint32_t B_end = min(A_start + 2 * currentSubArraySize - 1, array_length - 1);
            uint32_t A_length = B_start - A_start + 1;
            uint32_t B_length = B_end - B_start;

            if (currentSubArraySize > 64 && A_length == B_length) {
                // MergePathSplitter((*array) + A_start, A_length, (*array) + B_start + 1, B_length, C + A_start, A_length + B_length, 16, ASplitters, BSplitters);
                // serialMergeAVX512((*array) + A_start, A_length, (*array) + B_start + 1, B_length, C + A_start, A_length + B_length, ASplitters, BSplitters);
            } else {
                serialMerge((*array) + A_start, A_length, (*array) + B_start + 1, B_length, C + A_start, A_length + B_length);
            }
    	}

        //pointer swap for C
        vec_t* tmp = *array;
        *array = C;
        C = tmp;
    }

    free(C);
    free(ASplitters);
    free(BSplitters);
}

void iterativeMergeSortAVX512Modified3(vec_t** array, uint32_t array_length) {
    vec_t* C = (vec_t*)xcalloc((array_length + 8), sizeof(vec_t));
    uint32_t * ASplitters = (vec_t*)xcalloc((17), sizeof(vec_t));
    uint32_t * BSplitters = (vec_t*)xcalloc((17), sizeof(vec_t));

    uint32_t initialSubArraySize = 128;//array_length / numThreads;
    //if (array_length % 128 != 0) initialSubArraySize++;

    //sort one array per thread
    for (int i = 0; i < array_length; i += initialSubArraySize) {
        qsort(
            (*array) + i,
            (i + initialSubArraySize < array_length)?initialSubArraySize:(array_length - i),
            sizeof(vec_t), hostBasicCompare);
    }

    for (uint32_t currentSubArraySize = initialSubArraySize; currentSubArraySize < array_length; currentSubArraySize = 2 * currentSubArraySize)
    {
    	for (uint32_t A_start = 0; A_start < array_length - 1; A_start += 2 * currentSubArraySize)
    	{
    		uint32_t B_start = min(A_start + currentSubArraySize - 1, array_length - 1);
    		uint32_t B_end = min(A_start + 2 * currentSubArraySize - 1, array_length - 1);
            uint32_t A_length = B_start - A_start + 1;
            uint32_t B_length = B_end - B_start;

            if (currentSubArraySize > 64 && A_length == B_length) {
                // MergePathSplitter((*array) + A_start, A_length, (*array) + B_start + 1, B_length, C + A_start, A_length + B_length, 16, ASplitters, BSplitters);
                // serialMergeAVX512((*array) + A_start, A_length, (*array) + B_start + 1, B_length, C + A_start, A_length + B_length, ASplitters, BSplitters);
            } else {
                serialMerge((*array) + A_start, A_length, (*array) + B_start + 1, B_length, C + A_start, A_length + B_length);
            }
    	}

        //pointer swap for C
        vec_t* tmp = *array;
        *array = C;
        C = tmp;
    }

    free(C);
    free(ASplitters);
    free(BSplitters);
}

void parallelIMergeSort(vec_t** array, uint32_t array_length)
{
        vec_t* C = (vec_t*)xcalloc((array_length + 32), sizeof(vec_t));
        //uint32_t initialSubArraySize = (array_length % omp_get_num_threads()) ? (array_length / omp_get_num_threads()) + 1 : (array_length / omp_get_num_threads());
        #pragma omp parallel
        {
            //Calculate indicies
            uint32_t threadNum = omp_get_thread_num();
            uint32_t initialSubArraySize = (array_length % omp_get_num_threads()) ? (array_length / omp_get_num_threads()) + 1 : (array_length / omp_get_num_threads());
            uint32_t start,stop;
            uint32_t i = threadNum*initialSubArraySize;
            start=i;
            stop=start+(i + initialSubArraySize < array_length)?initialSubArraySize:(array_length - i);

            //in core sort
            qsort((*array) + start, stop, sizeof(vec_t), hostBasicCompare);

            //begin merging
            #pragma omp barrier
            uint32_t currentSubArraySize = initialSubArraySize;
            while (currentSubArraySize < array_length) {
                uint32_t A_start = threadNum * 2 * currentSubArraySize;
                if (A_start < array_length - 1) {
                    uint32_t B_start = min(A_start + currentSubArraySize - 1, array_length - 1);
                    uint32_t B_end = min(A_start + 2 * currentSubArraySize - 1, array_length - 1);
                    uint32_t A_length = B_start - A_start + 1;
                    uint32_t B_length = B_end - B_start;
                    serialMerge((*array) + A_start, A_length, (*array) + B_start + 1, B_length, C + A_start, A_length + B_length);
                }
                currentSubArraySize = 2 * currentSubArraySize;
                #pragma omp barrier
                #pragma omp single
                {
                    //pointer swap for C
                    vec_t* tmp = *array;
                    *array = C;
                    C = tmp;
                }
            }
        }

        free(C);
}
