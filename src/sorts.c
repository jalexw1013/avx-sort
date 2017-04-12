#include <stdio.h>
#include <stdint.h>
#include <float.h>
#include <assert.h>
#include <errno.h>
#include <stdlib.h>
#include <omp.h>
#include <malloc.h>
#include <x86intrin.h>
#include <time.h>
#include <unistd.h>
#include <string.h>

#include "utils/util.h"
#include "utils/xmalloc.h"
#include "sorts.h"

////////////////////////////////////////////////////////////////////////////////
//
// Merging algorithms
//
////////////////////////////////////////////////////////////////////////////////

void serialMerge(
    vec_t* A, uint32_t A_length,
    vec_t* B, uint32_t B_length,
    vec_t* C, uint32_t C_length)
{
    uint32_t Aindex = 0;
    uint32_t Bindex = 0;
    uint32_t Cindex = 0;

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
 * SSE Merge Sort From Srinivas's code
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

void bitonicMergeReal(vec_t* A, uint32_t A_length,
                      vec_t* B, uint32_t B_length,
                      vec_t* C, uint32_t C_length)
{
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

#ifdef AVX512
void avx512Merge(
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

////////////////////////////////////////////////////////////////////////////////
//
// Sorting algorithms
//
////////////////////////////////////////////////////////////////////////////////

void quickSort(vec_t** array, uint32_t array_length)
{
    qsort((void*)(*array), array_length, sizeof(vec_t), hostBasicCompare);
}

void iterativeMergeSort(vec_t** array, uint32_t array_length)
{
    vec_t* C = (vec_t*)xcalloc((array_length + 32), sizeof(vec_t));

    for (uint32_t currentSubArraySize = 1; currentSubArraySize < array_length; currentSubArraySize = 2 * currentSubArraySize)
    {
    	for (uint32_t A_start = 0; A_start < array_length - 1; A_start += 2 * currentSubArraySize)
    	{
    		uint32_t B_start = min(A_start + currentSubArraySize - 1, array_length - 1);
    		uint32_t B_end = min(A_start + 2 * currentSubArraySize - 1, array_length - 1);
            uint32_t A_length = B_start - A_start + 1;
            uint32_t B_length = B_end - B_start;

            serialMerge((*array) + A_start, A_length, (*array) + B_start + 1, B_length, C + A_start, A_length + B_length);
    	}
        //pointer swap for C
        vec_t* tmp = *array;
        *array = C;
        C = tmp;
    }
    free(C);
}

void parallelIterativeMergeSort(vec_t** array, uint32_t array_length)
{
        vec_t* C = (vec_t*)xcalloc((array_length + 32), sizeof(vec_t));
        int earlyEnd = 1; //Set to zero if small sub array

        #pragma omp parallel
        {
            //just quick sort for small array sizes (less than num threads)
            #pragma omp single
            {
                if (array_length < omp_get_num_threads()) {
                    qsort((*array), array_length, sizeof(vec_t), hostBasicCompare);
                    earlyEnd = 0;
                }
            }

            //end if already sorted above
            if (earlyEnd) {
                //Calculate indicies
                uint32_t threadNum = omp_get_thread_num();
                uint32_t initialSubArraySize = (array_length % omp_get_num_threads()) ? (array_length / omp_get_num_threads()) + 1 : (array_length / omp_get_num_threads());
                uint32_t start = threadNum*initialSubArraySize;
                uint32_t size = (start + initialSubArraySize < array_length)?initialSubArraySize:(array_length - start);

                //in core sort
                qsort((*array) + start, size, sizeof(vec_t), hostBasicCompare);

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
        }

        free(C);
}
