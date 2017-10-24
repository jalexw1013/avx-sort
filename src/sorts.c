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
#include <immintrin.h>
#include <sys/time.h>

#include "utils/util.h"
#include "utils/xmalloc.h"
#include "sorts.h"

////////////////////////////////////////////////////////////////////////////////
//
// Merging algorithms
//
////////////////////////////////////////////////////////////////////////////////

inline void serialMerge(
    vec_t* A, uint32_t A_length,
    vec_t* B, uint32_t B_length,
    vec_t* C, uint32_t C_length,
    struct memPointers* pointers)
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

inline void serialMergeNoBranch(
      vec_t* A, uint32_t A_length,
      vec_t* B, uint32_t B_length,
      vec_t* C, uint32_t C_length,
      struct memPointers* pointers)
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

inline void bitonicMergeReal(vec_t* A, uint32_t A_length,
                      vec_t* B, uint32_t B_length,
                      vec_t* C, uint32_t C_length,
                      struct memPointers* pointers)
{
    // TODO i think these can be 4s
    if (A_length < 5 || B_length < 5 || C_length < 5) {
        serialMerge(A,A_length,B,B_length,C,C_length,NULL);
        return;
    }

    long Aindex = 0,Bindex = 0, Cindex = 0;
    int isA = 0;//, isB;

    __m128i sA = _mm_loadu_si128((const __m128i*)&(A[Aindex]));
    __m128i sB = _mm_loadu_si128((const __m128i*)&(B[Bindex]));
    while ((Aindex < (A_length-4)) && (Bindex < (B_length-4)))
    {
        // load SIMD registers from A and B
        isA = 0;
        //isB = 0;
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
        if (A[Aindex+4]<B[Bindex+4]){
            Aindex+=4;
            isA = 1;
            sA = _mm_loadu_si128((const __m128i*)&(A[Aindex]));
        }
        else {
            Bindex+=4;
            //isB = 1;
            sA = _mm_loadu_si128((const __m128i*)&(B[Bindex]));
        }
    }
    if( isA ) Bindex += 4;
    else Aindex += 4;

    //int tempindex = 0;
    //int temp_length = 4;
    vec_t temp[4];
    _mm_storeu_si128((__m128i*)temp, sB);

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

inline void avx512MergeNoMergePath(
    vec_t* A, uint32_t A_length,
    vec_t* B, uint32_t B_length,
    vec_t* C, uint32_t C_length,
    struct memPointers* pointers)
{
    uint32_t* ASplitters = pointers->ASplitters;
    uint32_t* BSplitters = pointers->BSplitters;

    //start indexes
    __m512i vindexA = _mm512_load_epi32(ASplitters);
    __m512i vindexB = _mm512_load_epi32(BSplitters);
    __m512i vindexC = _mm512_add_epi32(vindexA, vindexB);

    //stop indexes
    const __m512i vindexAStop = _mm512_load_epi32(ASplitters + 1);
    const __m512i vindexBStop = _mm512_load_epi32(BSplitters + 1);

    //other Variables
    static const __m512i mizero = _mm512_set_epi32(0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0);
    static const __m512i mione = _mm512_set_epi32(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1);

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
        __m512i miCelems = _mm512_mask_blend_epi32(micmp, miBelems, miAelems);
        _mm512_mask_i32scatter_epi32((int *)C, exceededAStop | exceededBStop, vindexC, miCelems, 4);

        //increase indexes
        vindexA = _mm512_mask_add_epi32(vindexA, exceededAStop & micmp, vindexA, mione);
        vindexB = _mm512_mask_add_epi32(vindexB, exceededBStop & ~micmp, vindexB, mione);
        exceededAStop = _mm512_cmpgt_epi32_mask(vindexAStop, vindexA);
        exceededBStop = _mm512_cmpgt_epi32_mask(vindexBStop, vindexB);
        vindexC = _mm512_mask_add_epi32(vindexC, exceededAStop | exceededBStop, vindexC, mione);
    }
}

inline void avx512Merge(
    vec_t* A, uint32_t A_length,
    vec_t* B, uint32_t B_length,
    vec_t* C, uint32_t C_length,
    struct memPointers* pointers)
{
    MergePathSplitter(A, A_length, B, B_length, C,
        C_length, 16, pointers->ASplitters, pointers->BSplitters);

    avx512MergeNoMergePath(
        A, A_length,
        B, B_length,
        C, C_length,
        pointers);
}

inline void avx512ParallelMerge(
    vec_t* A, uint32_t A_length,
    vec_t* B, uint32_t B_length,
    vec_t* C, uint32_t C_length,
    struct memPointers* pointers)
{
    #pragma omp parallel
    {
        uint32_t numThreads = omp_get_num_threads();
        uint32_t threadNum = omp_get_thread_num();
        uint32_t* ASplitters = pointers->ASplitters + (numThreads + 1) * threadNum;
        uint32_t* BSplitters = pointers->BSplitters + (numThreads + 1) * threadNum;
        MergePathSplitter(A, A_length, B, B_length, C,
            C_length, numThreads, ASplitters, BSplitters);
        uint32_t A_length = ASplitters[threadNum + 1] - ASplitters[threadNum];
        uint32_t B_length = BSplitters[threadNum + 1] - BSplitters[threadNum];
        avx512Merge(A + ASplitters[threadNum], A_length,
            B + BSplitters[threadNum], B_length,
            C + ASplitters[threadNum] + BSplitters[threadNum], A_length + B_length, NULL);
    }
}
#endif

////////////////////////////////////////////////////////////////////////////////
//
// Sorting algorithms
//
////////////////////////////////////////////////////////////////////////////////
#ifdef SORT
void quickSort(
    vec_t* array, vec_t* C, uint32_t array_length, const uint32_t splitNumber, struct memPointers* pointers)
{
    qsort((void*)array, array_length, sizeof(vec_t), hostBasicCompare);
}

template <MergeTemplate Merge>
void avx512SortNoMergePath(
    vec_t* array, vec_t* C, uint32_t array_length, const uint32_t splitNumber, struct memPointers* pointers)
{
    // Checks, can be removed for performance testing
    if (array_length % 32 != 0) {
        printf("Segment Sort Array must be divisible by 32");
    }

    __m512i vindexA = _mm512_set_epi32(30, 28, 26, 24, 22, 20, 18, 16, 14, 12, 10, 8, 6, 4, 2, 0);
    __m512i vindexB = _mm512_set_epi32(31, 29, 27, 25, 23, 21, 19, 17, 15, 13, 11, 9, 7, 5, 3, 1);
    __m512i vindexBStop = _mm512_set_epi32(32, 30, 28, 26, 24, 22, 20, 18, 16, 14, 12, 10, 8, 6, 4, 2);

    // Round one, take unsorted array into sub arrays of size 2
    for (uint32_t index = 0; index < array_length; index += 32) {
        // Get Elements
        __m512i miAelems = _mm512_load_epi32(array + index);
        __m512i miBelems = _mm512_load_epi32(array + index + 16);

        //compare the elements
        __mmask16 micmp = _mm512_cmple_epi32_mask(miAelems, miBelems);

        //copy the elements to the final elements
        __m512i miC1elems = _mm512_mask_blend_epi32(micmp, miBelems, miAelems);
        __m512i miC2elems = _mm512_mask_blend_epi32(micmp, miAelems, miBelems);

        _mm512_i32scatter_epi32((int *)array + index, vindexA, miC1elems, 4);
        _mm512_i32scatter_epi32((int *)array + index, vindexB, miC2elems, 4);
    }

    static const __m512i mione = _mm512_set_epi32(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1);
    __m512i vindexAInner, vindexBInner, vindexCInner;
    __m512i miAelems, miBelems, miCelems;
    __mmask16 maskA = (__mmask16)0xFFFF, maskB = (__mmask16)0xFFFF, micmp;

    uint32_t sortedArraySize = 2;
    for (; sortedArraySize < array_length / 32; sortedArraySize <<= 1) {
        if (sortedArraySize >= 4) return;
        //printf("\n\n\n\n\n\n");
        //printf("Sorted Size %d\n", sortedArraySize);
        vindexA = _mm512_slli_epi32(vindexA, 1);
        vindexB = _mm512_slli_epi32(vindexB, 1);
        vindexBStop = _mm512_slli_epi32(vindexBStop, 1);
        for (uint32_t index = 0; index < array_length; index += 32 * sortedArraySize) {
            vindexAInner = vindexA;
            vindexBInner = vindexB;
            vindexCInner = vindexA;
            //print512_num("vindexBInner", vindexBInner);
            miAelems = _mm512_i32gather_epi32(vindexAInner, (const int *)array + index, 4);
            miBelems = _mm512_i32gather_epi32(vindexBInner, (const int *)array + index, 4);

            // print512_num("miAelems", miAelems);
            // print512_num("miBelems", miBelems);

            //compare the elements
            micmp = _mm512_cmple_epi32_mask(miAelems, miBelems);
            //printmmask16("micmp", micmp);
            // printmmask16("maskB", maskB);
            // printmmask16("maskA", maskA);
            //copy the elements to the final elements
            miCelems = _mm512_mask_blend_epi32(micmp, miBelems, miAelems);
            _mm512_i32scatter_epi32((int *)C + index, vindexCInner, miCelems, 4);
            //print512_num("vindexCInner", vindexCInner);
            //print512_num("miCelems", miCelems);

            // increase indexes
            vindexAInner = _mm512_mask_add_epi32(vindexAInner, micmp, vindexAInner, mione);
            vindexBInner = _mm512_mask_add_epi32(vindexBInner, ~micmp, vindexBInner, mione);
            vindexCInner = _mm512_add_epi32(vindexCInner, mione);
            uint32_t l1Index = index + 16;
            // printf("\n\n");
            for (; l1Index < index + 32 * sortedArraySize - 16; l1Index += 16) {


                //printf("\n\n\n");
                maskA = _mm512_cmplt_epi32_mask(vindexAInner, vindexB);
                maskB = _mm512_cmplt_epi32_mask(vindexBInner, vindexBStop);
                // print512_num("vindexBInner", vindexBInner);
                // print512_num("vindexBStop", vindexBStop);
                // printmmask16("mask", mask);
                // printmmask16("micmp", micmp);
                miAelems = _mm512_mask_i32gather_epi32(miAelems, micmp & maskA, vindexAInner, (const int *)array + index, 4);
                miBelems = _mm512_mask_i32gather_epi32(miBelems, (~micmp) & maskB, vindexBInner, (const int *)array + index, 4);
                // printmmask16("micmp", micmp);
                // printmmask16("maskB", maskB);
                // print512_num("miAelems", miAelems);
                // print512_num("miBelems", miBelems);

                //print512_num("miAelems", miAelems);
                //print512_num("miBelems", miBelems);

                //compare the elements
                // printmmask16("micmp", micmp);
                // printmmask16("maskB", maskB);
                // printmmask16("maskA", maskA);
                micmp = _mm512_mask_cmple_epi32_mask(maskA, miAelems, miBelems);
                // printmmask16("micmp", micmp);
                // printmmask16("maskB", maskB);
                // printmmask16("maskA", maskA);
                micmp |= ~maskB;
                // printmmask16("micmp", micmp);
                // printmmask16("maskB", maskB);
                // printmmask16("maskA", maskA);
                //copy the elements to the final elements
                miCelems = _mm512_mask_blend_epi32(micmp, miBelems, miAelems);
                //print512_num("miCelems", miCelems);
                //printf("scatteing the previous C at l1index %d and with the following vindexes\n", );
                _mm512_i32scatter_epi32((int *)C + index, vindexCInner, miCelems, 4);
                //print512_num("vindexCInner", vindexCInner);
                //print512_num("miCelems", miCelems);

                // increase indexes
                vindexAInner = _mm512_mask_add_epi32(vindexAInner, micmp, vindexAInner, mione);
                vindexBInner = _mm512_mask_add_epi32(vindexBInner, ~micmp, vindexBInner, mione);
                vindexCInner = _mm512_add_epi32(vindexCInner, mione);
                //printf("\n\n");
            }
            //printf("\n\n\n");
            maskA = _mm512_cmplt_epi32_mask(vindexAInner, vindexB);
            maskB = _mm512_cmplt_epi32_mask(vindexBInner, vindexBStop);
            //print512_num("vindexANext", vindexBStop);
            //print512_num("vindexBInner", vindexBInner);
            //printmmask16("maskA", maskA);
             //printmmask16("maskB", maskB);
             //printmmask16("micmp", micmp);
            miAelems = _mm512_mask_i32gather_epi32(miAelems, micmp & maskA, vindexAInner, (const int *)array + index, 4);
            miBelems = _mm512_mask_i32gather_epi32(miBelems, (~micmp) & maskB, vindexBInner, (const int *)array + index, 4);

            // print512_num("miAelems", miAelems);
            // print512_num("miBelems", miBelems);

            //print512_num("miAelems", miAelems);
            //print512_num("miBelems", miBelems);

            //compare the elements
            micmp = _mm512_mask_cmple_epi32_mask(maskA, miAelems, miBelems);
            micmp |= ~maskB;
            // printmmask16("micmp", micmp);
            // printmmask16("maskB", maskB);
            // printmmask16("maskA", maskA);
            //copy the elements to the final elements
            miCelems = _mm512_mask_blend_epi32(micmp, miBelems, miAelems);
            //print512_num("miCelems", miCelems);
            //printf("scatteing the previous C at l1index %d and with the following vindexes\n", );
            _mm512_i32scatter_epi32((int *)C + index, vindexCInner, miCelems, 4);
            //print512_num("vindexCInner", vindexCInner);
            //print512_num("miCelems", miCelems);
            //return;
        }
        // Pointer Swap
        vec_t* tmp = array;
        array = C;
        C = tmp;

        // for (uint32_t i = 0; i < 128; i++) {
        //     printf("C[%d]:%d\n", i, array[i]);
        // }
        //return;
    }

    // for (uint32_t i = 0; i < 1024; i++) {
    //     printf("C[%d]:%d\n", i, array[i]);
    // }

    //return;

    // printf("\n\n\n\n\n\n\n\n\n\n\n\n\n");


    uint32_t numberOfSwaps = 0;
    for (; sortedArraySize < array_length; sortedArraySize <<= 1) {
        for (uint32_t A_start = 0; A_start < array_length; A_start += 2 * sortedArraySize)
    	{
            uint32_t A_end = min(A_start + sortedArraySize, array_length - 1);
    		uint32_t B_start = A_end;
    		uint32_t B_end = min(A_start + 2 * sortedArraySize, array_length);
            uint32_t A_length = A_end - A_start;
            uint32_t B_length = B_end - B_start;

            // printf("A_start:%d\n", A_start);
            // printf("A_end:%d\n", A_end);
            // printf("A_length:%d\n", A_length);
            // printf("B_start:%d\n", B_start);
            // printf("B_end:%d\n", B_end);
            // printf("B_length:%d\n", B_length);
            // printf("\n\n");

            Merge(array + A_start, A_length, array + B_start, B_length, C + A_start, A_length + B_length, pointers);
    	}
        //pointer swap for C
        vec_t* tmp = array;
        array = C;
        C = tmp;
        numberOfSwaps++;

        // for (uint32_t i = 0; i < 128; i++) {
        //     printf("C[%d]:%d\n", i, array[i]);
        // }
        //return;
    }

    if (numberOfSwaps%2 == 1) {
        memcpy((void*)C,(void*)array, (array_length)*sizeof(vec_t));
        vec_t* tmp = array;
        array = C;
        C = tmp;
    }

    // for (uint32_t i = 0; i < 1024; i++) {
    //     printf("C[%d]:%d\n", i, array[i]);
    // }

}

template <MergeTemplate Merge>
void avx512SortNoMergePathV2(
    vec_t* array, vec_t* C, uint32_t array_length, const uint32_t splitNumber, struct memPointers* pointers)
{
    // Checks, can be removed for performance testing
    if (array_length % 32 != 0) {
        printf("Segment Sort Array must be divisible by 32");
    }

    __m512i vindexA = _mm512_set_epi32(30, 28, 26, 24, 22, 20, 18, 16, 14, 12, 10, 8, 6, 4, 2, 0);
    __m512i vindexB = _mm512_set_epi32(31, 29, 27, 25, 23, 21, 19, 17, 15, 13, 11, 9, 7, 5, 3, 1);
    __m512i vindexBStop = _mm512_set_epi32(32, 30, 28, 26, 24, 22, 20, 18, 16, 14, 12, 10, 8, 6, 4, 2);

    // Round one, take unsorted array into sub arrays of size 2
    for (uint32_t index = 0; index < array_length; index += 32) {
        // Get Elements
        __m512i miAelems = _mm512_load_epi32(array + index);
        __m512i miBelems = _mm512_load_epi32(array + index + 16);

        //compare the elements
        __mmask16 micmp = _mm512_cmple_epi32_mask(miAelems, miBelems);

        //copy the elements to the final elements
        __m512i miC1elems = _mm512_mask_blend_epi32(micmp, miBelems, miAelems);
        __m512i miC2elems = _mm512_mask_blend_epi32(micmp, miAelems, miBelems);

        _mm512_store_epi32((int *)array + index, miC1elems);
        _mm512_store_epi32((int *)array + index + 16, miC2elems);
    }

    static const __m512i roundTwoMax = _mm512_set_epi32(2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2);
    static const __m512i mione = _mm512_set_epi32(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1);
    static const __m512i miZero = _mm512_set_epi32(0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0);

    for (uint32_t index = 0; index < array_length; index += 64) {
        __m512i ACount = miZero;
        __m512i BCount = miZero;
        // print512_num("ACount", ACount);
        // print512_num("BCount", BCount);
        // Get Elements
        __m512i miA0elems = _mm512_load_epi32(array + index);
        __m512i miA1elems = _mm512_load_epi32(array + index + 16);
        __m512i miB0elems = _mm512_load_epi32(array + index + 32);
        __m512i miB1elems = _mm512_load_epi32(array + index + 48);

        // print512_num("miA0elems", miA0elems);
        // print512_num("miA1elems", miA1elems);
        // print512_num("miB0elems", miB0elems);
        // print512_num("miB1elems", miB1elems);

        __mmask16 micmp = _mm512_cmple_epi32_mask(miA0elems, miB0elems);
        __m512i miC0elems = _mm512_mask_blend_epi32(micmp, miB0elems, miA0elems);

        // print512_num("miC0elems", miC0elems);
        // print512_num("miA0elems", miA0elems);
        // print512_num("miB0elems", miB0elems);


        _mm512_store_epi32((int *)array + index, miC0elems);
        ACount = _mm512_mask_add_epi32(ACount, micmp, ACount, mione);
        BCount = _mm512_mask_add_epi32(BCount, ~micmp, BCount, mione);
        // printmmask16("micmp", micmp);
        // print512_num("ACount", ACount);
        // print512_num("BCount", BCount);
        miA0elems = _mm512_mask_blend_epi32(micmp, miA0elems, miA1elems);
        miB0elems = _mm512_mask_blend_epi32(micmp, miB1elems, miB0elems);
        micmp = _mm512_cmple_epi32_mask(miA0elems, miB0elems);
        __m512i miC1elems = _mm512_mask_blend_epi32(micmp, miB0elems, miA0elems);

        // print512_num("miC1elems", miC1elems);
        // print512_num("miA0elems", miA0elems);
        // print512_num("miB0elems", miB0elems);

        _mm512_store_epi32((int *)array + index + 16, miC1elems);
        ACount = _mm512_mask_add_epi32(ACount, micmp, ACount, mione);
        BCount = _mm512_mask_add_epi32(BCount, ~micmp, BCount, mione);
        // printmmask16("micmp", micmp);
        // print512_num("ACount", ACount);
        // print512_num("BCount", BCount);
        miA0elems = _mm512_mask_blend_epi32(micmp, miA0elems, miA1elems);
        miB0elems = _mm512_mask_blend_epi32(micmp, miB1elems, miB0elems);
        micmp = _mm512_cmple_epi32_mask(miA0elems, miB0elems);
        micmp &= _mm512_cmplt_epi32_mask(ACount, roundTwoMax);
        micmp |= _mm512_cmpge_epi32_mask(BCount, roundTwoMax);
        __m512i miC2elems = _mm512_mask_blend_epi32(micmp, miB0elems, miA0elems);

        // print512_num("miC2elems", miC2elems);
        // print512_num("miA0elems", miA0elems);
        // print512_num("miB0elems", miB0elems);

        _mm512_store_epi32((int *)array + index + 32, miC2elems);
        ACount = _mm512_mask_add_epi32(ACount, micmp, ACount, mione);
        BCount = _mm512_mask_add_epi32(BCount, ~micmp, BCount, mione);
        // printmmask16("micmp", micmp);
        // print512_num("ACount", ACount);
        // print512_num("BCount", BCount);
        miA0elems = _mm512_mask_blend_epi32(micmp, miA0elems, miA1elems);
        miB0elems = _mm512_mask_blend_epi32(micmp, miB1elems, miB0elems);
        micmp = _mm512_cmple_epi32_mask(miA0elems, miB0elems);
        micmp &= _mm512_cmplt_epi32_mask(ACount, roundTwoMax);
        micmp |= _mm512_cmpge_epi32_mask(BCount, roundTwoMax);
        __m512i miC3elems = _mm512_mask_blend_epi32(micmp, miB0elems, miA0elems);
        _mm512_store_epi32((int *)array + index + 48, miC3elems);

        // print512_num("miC3elems", miC3elems);
        // print512_num("miA0elems", miA0elems);
        // print512_num("miB0elems", miB0elems);
    }


    // Round Two, take sorted sub-arrays of size 2 into sub-arrays of size 4
    for (uint32_t index = 0; index < array_length; index += 64) {
        __m512i ACount = miZero;
        __m512i BCount = miZero;
        // print512_num("ACount", ACount);
        // print512_num("BCount", BCount);
        // Get Elements
        __m512i miA0elems = _mm512_load_epi32(array + index);
        __m512i miA1elems = _mm512_load_epi32(array + index + 16);
        __m512i miB0elems = _mm512_load_epi32(array + index + 32);
        __m512i miB1elems = _mm512_load_epi32(array + index + 48);

        // print512_num("miA0elems", miA0elems);
        // print512_num("miA1elems", miA1elems);
        // print512_num("miB0elems", miB0elems);
        // print512_num("miB1elems", miB1elems);

        __mmask16 micmp = _mm512_cmple_epi32_mask(miA0elems, miB0elems);
        __m512i miC0elems = _mm512_mask_blend_epi32(micmp, miB0elems, miA0elems);

        // print512_num("miC0elems", miC0elems);
        // print512_num("miA0elems", miA0elems);
        // print512_num("miB0elems", miB0elems);


        _mm512_store_epi32((int *)array + index, miC0elems);
        ACount = _mm512_mask_add_epi32(ACount, micmp, ACount, mione);
        BCount = _mm512_mask_add_epi32(BCount, ~micmp, BCount, mione);
        // printmmask16("micmp", micmp);
        // print512_num("ACount", ACount);
        // print512_num("BCount", BCount);
        miA0elems = _mm512_mask_blend_epi32(micmp, miA0elems, miA1elems);
        miB0elems = _mm512_mask_blend_epi32(micmp, miB1elems, miB0elems);
        micmp = _mm512_cmple_epi32_mask(miA0elems, miB0elems);
        __m512i miC1elems = _mm512_mask_blend_epi32(micmp, miB0elems, miA0elems);

        // print512_num("miC1elems", miC1elems);
        // print512_num("miA0elems", miA0elems);
        // print512_num("miB0elems", miB0elems);

        _mm512_store_epi32((int *)array + index + 16, miC1elems);
        ACount = _mm512_mask_add_epi32(ACount, micmp, ACount, mione);
        BCount = _mm512_mask_add_epi32(BCount, ~micmp, BCount, mione);
        // printmmask16("micmp", micmp);
        // print512_num("ACount", ACount);
        // print512_num("BCount", BCount);
        miA0elems = _mm512_mask_blend_epi32(micmp, miA0elems, miA1elems);
        miB0elems = _mm512_mask_blend_epi32(micmp, miB1elems, miB0elems);
        micmp = _mm512_cmple_epi32_mask(miA0elems, miB0elems);
        micmp &= _mm512_cmplt_epi32_mask(ACount, roundTwoMax);
        micmp |= _mm512_cmpge_epi32_mask(BCount, roundTwoMax);
        __m512i miC2elems = _mm512_mask_blend_epi32(micmp, miB0elems, miA0elems);

        // print512_num("miC2elems", miC2elems);
        // print512_num("miA0elems", miA0elems);
        // print512_num("miB0elems", miB0elems);

        _mm512_store_epi32((int *)array + index + 32, miC2elems);
        ACount = _mm512_mask_add_epi32(ACount, micmp, ACount, mione);
        BCount = _mm512_mask_add_epi32(BCount, ~micmp, BCount, mione);
        // printmmask16("micmp", micmp);
        // print512_num("ACount", ACount);
        // print512_num("BCount", BCount);
        miA0elems = _mm512_mask_blend_epi32(micmp, miA0elems, miA1elems);
        miB0elems = _mm512_mask_blend_epi32(micmp, miB1elems, miB0elems);
        micmp = _mm512_cmple_epi32_mask(miA0elems, miB0elems);
        micmp &= _mm512_cmplt_epi32_mask(ACount, roundTwoMax);
        micmp |= _mm512_cmpge_epi32_mask(BCount, roundTwoMax);
        __m512i miC3elems = _mm512_mask_blend_epi32(micmp, miB0elems, miA0elems);
        _mm512_store_epi32((int *)array + index + 48, miC3elems);

        // print512_num("miC3elems", miC3elems);
        // print512_num("miA0elems", miA0elems);
        // print512_num("miB0elems", miB0elems);
    }



        // for (uint32_t i = 0; i < 100; i += 32) {
        //     for (uint32_t j = i; j < 16; j++) {
        //         printf("array[%d]:%d\n", j, array[j]);
        //         printf("array[%d]:%d\n", j + 16, array[j + 16]);
        //         printf("array[%d]:%d\n", j + 32, array[j + 32]);
        //         printf("array[%d]:%d\n", j + 48, array[j + 48]);
        //     }
        // }

        return;


    static const __m512i roundThreeMax = _mm512_set_epi32(4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4);

    // Round three, take sorted sub-arrays of size 4 into sub-arrays of size 8
    for (uint32_t index = 0; index < array_length; index += 128) {
        __m512i ACount = miZero;
        __m512i BCount = miZero;
        // Get Elements
        __m512i miA0elems = _mm512_load_epi32(array + index);
        __m512i miA1elems = _mm512_load_epi32(array + index + 16);
        __m512i miA2elems = _mm512_load_epi32(array + index + 32);
        __m512i miA3elems = _mm512_load_epi32(array + index + 48);
        __m512i miB0elems = _mm512_load_epi32(array + index + 64);
        __m512i miB1elems = _mm512_load_epi32(array + index + 80);
        __m512i miB2elems = _mm512_load_epi32(array + index + 96);
        __m512i miB3elems = _mm512_load_epi32(array + index + 108);

        __mmask16 micmp = _mm512_cmple_epi32_mask(miA0elems, miB0elems);
        __m512i miC0elems = _mm512_mask_blend_epi32(micmp, miB0elems, miA0elems);
        _mm512_store_epi32((int *)array + index, miC0elems);
        ACount = _mm512_mask_add_epi32(ACount, micmp, ACount, mione);
        BCount = _mm512_mask_add_epi32(BCount, ~micmp, BCount, mione);
        miA0elems = _mm512_mask_blend_epi32(micmp, miA0elems, miA1elems);
        miA1elems = _mm512_mask_blend_epi32(micmp, miA1elems, miA2elems);
        miA2elems = _mm512_mask_blend_epi32(micmp, miA2elems, miA3elems);
        miB0elems = _mm512_mask_blend_epi32(micmp, miB1elems, miB0elems);
        miB1elems = _mm512_mask_blend_epi32(micmp, miB2elems, miB1elems);
        miB2elems = _mm512_mask_blend_epi32(micmp, miB3elems, miB2elems);
        micmp = _mm512_cmple_epi32_mask(miA0elems, miB0elems);
        __m512i miC1elems = _mm512_mask_blend_epi32(micmp, miB0elems, miA0elems);
        _mm512_store_epi32((int *)array + index + 16, miC1elems);
        ACount = _mm512_mask_add_epi32(ACount, micmp, ACount, mione);
        BCount = _mm512_mask_add_epi32(BCount, ~micmp, BCount, mione);
        miA0elems = _mm512_mask_blend_epi32(micmp, miA0elems, miA1elems);
        miB0elems = _mm512_mask_blend_epi32(micmp, miB1elems, miB0elems);
        micmp = _mm512_cmple_epi32_mask(miA0elems, miB0elems);
        micmp &= _mm512_cmple_epi32_mask(ACount, roundTwoMax);
        micmp |= _mm512_cmpge_epi32_mask(BCount, roundTwoMax);
        __m512i miC2elems = _mm512_mask_blend_epi32(micmp, miB0elems, miA0elems);
        _mm512_store_epi32((int *)array + index + 32, miC2elems);
        ACount = _mm512_mask_add_epi32(ACount, micmp, ACount, mione);
        BCount = _mm512_mask_add_epi32(BCount, ~micmp, BCount, mione);
        miA0elems = _mm512_mask_blend_epi32(micmp, miA0elems, miA1elems);
        miB0elems = _mm512_mask_blend_epi32(micmp, miB1elems, miB0elems);
        micmp = _mm512_cmple_epi32_mask(miA0elems, miB0elems);
        micmp &= _mm512_cmple_epi32_mask(ACount, roundTwoMax);
        micmp |= _mm512_cmpge_epi32_mask(BCount, roundTwoMax);
        __m512i miC3elems = _mm512_mask_blend_epi32(micmp, miB0elems, miA0elems);
        _mm512_store_epi32((int *)array + index + 48, miC3elems);
    }

    __m512i vindexAInner, vindexBInner, vindexCInner;
    __m512i miAelems, miBelems, miCelems;
    __mmask16 maskA = (__mmask16)0xFFFF, maskB = (__mmask16)0xFFFF, micmp;

    uint32_t sortedArraySize = 2;
    for (; sortedArraySize < array_length / 32; sortedArraySize <<= 1) {
        //printf("\n\n\n\n\n\n");
        //printf("Sorted Size %d\n", sortedArraySize);
        vindexA = _mm512_slli_epi32(vindexA, 1);
        vindexB = _mm512_slli_epi32(vindexB, 1);
        vindexBStop = _mm512_slli_epi32(vindexBStop, 1);
        for (uint32_t index = 0; index < array_length; index += 32 * sortedArraySize) {
            vindexAInner = vindexA;
            vindexBInner = vindexB;
            vindexCInner = vindexA;
            //print512_num("vindexBInner", vindexBInner);
            miAelems = _mm512_i32gather_epi32(vindexAInner, (const int *)array + index, 4);
            miBelems = _mm512_i32gather_epi32(vindexBInner, (const int *)array + index, 4);

            // print512_num("miAelems", miAelems);
            // print512_num("miBelems", miBelems);

            //compare the elements
            micmp = _mm512_cmple_epi32_mask(miAelems, miBelems);
            //printmmask16("micmp", micmp);
            // printmmask16("maskB", maskB);
            // printmmask16("maskA", maskA);
            //copy the elements to the final elements
            miCelems = _mm512_mask_blend_epi32(micmp, miBelems, miAelems);
            _mm512_i32scatter_epi32((int *)C + index, vindexCInner, miCelems, 4);
            //print512_num("vindexCInner", vindexCInner);
            //print512_num("miCelems", miCelems);

            // increase indexes
            vindexAInner = _mm512_mask_add_epi32(vindexAInner, micmp, vindexAInner, mione);
            vindexBInner = _mm512_mask_add_epi32(vindexBInner, ~micmp, vindexBInner, mione);
            vindexCInner = _mm512_add_epi32(vindexCInner, mione);
            uint32_t l1Index = index + 16;
            // printf("\n\n");
            for (; l1Index < index + 32 * sortedArraySize - 16; l1Index += 16) {


                //printf("\n\n\n");
                maskA = _mm512_cmplt_epi32_mask(vindexAInner, vindexB);
                maskB = _mm512_cmplt_epi32_mask(vindexBInner, vindexBStop);
                // print512_num("vindexBInner", vindexBInner);
                // print512_num("vindexBStop", vindexBStop);
                // printmmask16("mask", mask);
                // printmmask16("micmp", micmp);
                miAelems = _mm512_mask_i32gather_epi32(miAelems, micmp & maskA, vindexAInner, (const int *)array + index, 4);
                miBelems = _mm512_mask_i32gather_epi32(miBelems, (~micmp) & maskB, vindexBInner, (const int *)array + index, 4);
                // printmmask16("micmp", micmp);
                // printmmask16("maskB", maskB);
                // print512_num("miAelems", miAelems);
                // print512_num("miBelems", miBelems);

                //print512_num("miAelems", miAelems);
                //print512_num("miBelems", miBelems);

                //compare the elements
                // printmmask16("micmp", micmp);
                // printmmask16("maskB", maskB);
                // printmmask16("maskA", maskA);
                micmp = _mm512_mask_cmple_epi32_mask(maskA, miAelems, miBelems);
                // printmmask16("micmp", micmp);
                // printmmask16("maskB", maskB);
                // printmmask16("maskA", maskA);
                micmp |= ~maskB;
                // printmmask16("micmp", micmp);
                // printmmask16("maskB", maskB);
                // printmmask16("maskA", maskA);
                //copy the elements to the final elements
                miCelems = _mm512_mask_blend_epi32(micmp, miBelems, miAelems);
                //print512_num("miCelems", miCelems);
                //printf("scatteing the previous C at l1index %d and with the following vindexes\n", );
                _mm512_i32scatter_epi32((int *)C + index, vindexCInner, miCelems, 4);
                //print512_num("vindexCInner", vindexCInner);
                //print512_num("miCelems", miCelems);

                // increase indexes
                vindexAInner = _mm512_mask_add_epi32(vindexAInner, micmp, vindexAInner, mione);
                vindexBInner = _mm512_mask_add_epi32(vindexBInner, ~micmp, vindexBInner, mione);
                vindexCInner = _mm512_add_epi32(vindexCInner, mione);
                //printf("\n\n");
            }
            //printf("\n\n\n");
            maskA = _mm512_cmplt_epi32_mask(vindexAInner, vindexB);
            maskB = _mm512_cmplt_epi32_mask(vindexBInner, vindexBStop);
            //print512_num("vindexANext", vindexBStop);
            //print512_num("vindexBInner", vindexBInner);
            //printmmask16("maskA", maskA);
             //printmmask16("maskB", maskB);
             //printmmask16("micmp", micmp);
            miAelems = _mm512_mask_i32gather_epi32(miAelems, micmp & maskA, vindexAInner, (const int *)array + index, 4);
            miBelems = _mm512_mask_i32gather_epi32(miBelems, (~micmp) & maskB, vindexBInner, (const int *)array + index, 4);

            // print512_num("miAelems", miAelems);
            // print512_num("miBelems", miBelems);

            //print512_num("miAelems", miAelems);
            //print512_num("miBelems", miBelems);

            //compare the elements
            micmp = _mm512_mask_cmple_epi32_mask(maskA, miAelems, miBelems);
            micmp |= ~maskB;
            // printmmask16("micmp", micmp);
            // printmmask16("maskB", maskB);
            // printmmask16("maskA", maskA);
            //copy the elements to the final elements
            miCelems = _mm512_mask_blend_epi32(micmp, miBelems, miAelems);
            //print512_num("miCelems", miCelems);
            //printf("scatteing the previous C at l1index %d and with the following vindexes\n", );
            _mm512_i32scatter_epi32((int *)C + index, vindexCInner, miCelems, 4);
            //print512_num("vindexCInner", vindexCInner);
            //print512_num("miCelems", miCelems);
            //return;
        }
        // Pointer Swap
        vec_t* tmp = array;
        array = C;
        C = tmp;

        // for (uint32_t i = 0; i < 128; i++) {
        //     printf("C[%d]:%d\n", i, array[i]);
        // }
        //return;
    }

    // for (uint32_t i = 0; i < 1024; i++) {
    //     printf("C[%d]:%d\n", i, array[i]);
    // }

    //return;

    // printf("\n\n\n\n\n\n\n\n\n\n\n\n\n");


    uint32_t numberOfSwaps = 0;
    for (; sortedArraySize < array_length; sortedArraySize <<= 1) {
        for (uint32_t A_start = 0; A_start < array_length; A_start += 2 * sortedArraySize)
    	{
            uint32_t A_end = min(A_start + sortedArraySize, array_length - 1);
    		uint32_t B_start = A_end;
    		uint32_t B_end = min(A_start + 2 * sortedArraySize, array_length);
            uint32_t A_length = A_end - A_start;
            uint32_t B_length = B_end - B_start;

            // printf("A_start:%d\n", A_start);
            // printf("A_end:%d\n", A_end);
            // printf("A_length:%d\n", A_length);
            // printf("B_start:%d\n", B_start);
            // printf("B_end:%d\n", B_end);
            // printf("B_length:%d\n", B_length);
            // printf("\n\n");

            Merge(array + A_start, A_length, array + B_start, B_length, C + A_start, A_length + B_length, pointers);
    	}
        //pointer swap for C
        vec_t* tmp = array;
        array = C;
        C = tmp;
        numberOfSwaps++;

        // for (uint32_t i = 0; i < 128; i++) {
        //     printf("C[%d]:%d\n", i, array[i]);
        // }
        //return;
    }

    if (numberOfSwaps%2 == 1) {
        memcpy((void*)C,(void*)array, (array_length)*sizeof(vec_t));
        vec_t* tmp = array;
        array = C;
        C = tmp;
    }

    // for (uint32_t i = 0; i < 1024; i++) {
    //     printf("C[%d]:%d\n", i, array[i]);
    // }

}

template <MergeTemplate Merge>
void iterativeMergeSort(
    vec_t* array, vec_t* C, uint32_t array_length, const uint32_t splitNumber, struct memPointers* pointers)
{
    //vec_t* C = (vec_t*)xcalloc((array_length + 32), sizeof(vec_t));

    // We can't return a C that we alloced because this
    // will cause memory issues.
    // Therefore we might need to copy in the last step
    int numberOfSwaps = 0;

    uint32_t start = splitNumber; //Just in case splitNumber is invalid

    if (splitNumber > 1) {
        //sort individual arrays of size splitNumber
        for (uint32_t i = 0; i < array_length; i += splitNumber) {
            //adjust when array_length is not divisible by splitNumber
            uint32_t actualSubArraySize = min(splitNumber, array_length - i);
            qsort((void*)(array + i), actualSubArraySize, sizeof(vec_t), hostBasicCompare);
        }
    } else {
        start = 1;
    }

    //now do actual iterative merge sort
    for (uint32_t currentSubArraySize = start; currentSubArraySize < array_length; currentSubArraySize = 2 * currentSubArraySize)
    {
    	for (uint32_t A_start = 0; A_start < array_length; A_start += 2 * currentSubArraySize)
    	{
            uint32_t A_end = min(A_start + currentSubArraySize, array_length - 1);
    		uint32_t B_start = A_end;
    		uint32_t B_end = min(A_start + 2 * currentSubArraySize, array_length);
            uint32_t A_length = A_end - A_start;
            uint32_t B_length = B_end - B_start;

           Merge(array + A_start, A_length, array + B_start, B_length, C + A_start, A_length + B_length, pointers);
    	}
        //pointer swap for C
        vec_t* tmp = array;
        array = C;
        C = tmp;
        numberOfSwaps++;
    }

    if (numberOfSwaps%2 == 1) {
        memcpy((void*)C,(void*)array, (array_length)*sizeof(vec_t));
        vec_t* tmp = array;
        array = C;
        C = tmp;
    }

    //free(C);
}

/**
 * sort where array_length must be a power of 2
 */
template <MergeTemplate Merge>
void iterativeMergeSortPower2(
    vec_t* array, vec_t* C, uint32_t array_length, const uint32_t startSize, struct memPointers* pointers)
{
    for (uint32_t i = 0; i < array_length; i += startSize) {
        qsort((void*)(array + i), startSize, sizeof(vec_t), hostBasicCompare);
    }

    return;

    int numberOfSwaps = 0;

    //now do actual iterative merge sort
    for (uint32_t currentSubArraySize = startSize; currentSubArraySize < array_length; currentSubArraySize = 2 * currentSubArraySize)
    {
    	for (uint32_t A_start = 0; A_start < array_length; A_start += 2 * currentSubArraySize)
    	{
            Merge(array + A_start, currentSubArraySize, array + A_start + currentSubArraySize, currentSubArraySize, C + A_start, currentSubArraySize * 2, NULL);
    	}

        //pointer swap for C
        vec_t* tmp = array;
        array = C;
        C = tmp;
        numberOfSwaps++;
    }

    if (numberOfSwaps%2 == 1) {
        memcpy((void*)C,(void*)array, (array_length)*sizeof(vec_t));
        vec_t* tmp = array;
        array = C;
        C = tmp;
    }
}

/*
 * Sums the values of the array up to and not including the given index
 */
inline uint32_t arraySum(uint32_t* array, uint32_t sumToIndex) {
    uint32_t sum = 0;
    for (uint32_t i = 0; i < sumToIndex; i++) {
        sum += array[i];
    }
    return sum;
}

template <SortTemplate Sort, MergeTemplate Merge>
void parallelIterativeMergeSort(
    vec_t* array, vec_t* C, uint32_t array_length, const uint32_t splitNumber, struct memPointers* pointers)
{
    int earlyEnd = 1; //Set to zero if small sub array or error
    int numberOfSwaps = 0;

    #pragma omp parallel
    {
        //algorithm does not support array_length < num threads
        //for that case, just quick sort
        #pragma omp single
        {
            if (array_length < (uint32_t)omp_get_num_threads()) {
                quickSort(array, C, array_length, splitNumber, NULL);
                earlyEnd = 0;
            }
        }

        if (earlyEnd) {
            uint32_t threadNum = omp_get_thread_num();
            uint32_t numberOfThreads = omp_get_num_threads();
            uint32_t* ASplitters = pointers->ASplitters + (numberOfThreads + 1) * threadNum;
            uint32_t* BSplitters = pointers->BSplitters + (numberOfThreads + 1) * threadNum;
            uint32_t* arraySizes = pointers->arraySizes + numberOfThreads * threadNum;

            uint32_t numberOfSubArrays = numberOfThreads;
            uint32_t initialSubArraySize = array_length / numberOfThreads;

            //Calculate the size of each subarray
            for (uint32_t thread = 0; thread < (uint32_t)numberOfThreads; thread++) {
                arraySizes[thread] = initialSubArraySize;
                if (((thread % 2) == 1) && thread < 2*(array_length % (uint32_t)numberOfThreads)) {
                    arraySizes[thread]++;
                } else if ((array_length % (uint32_t)numberOfThreads) > numberOfSubArrays/2 && thread < 2*((array_length % (uint32_t)numberOfThreads) - numberOfSubArrays/2)) {
                    arraySizes[thread]++;
                }
            }

            uint32_t threadStartIndex = arraySum(arraySizes, threadNum);
            uint32_t currentSubArraySize = arraySizes[threadNum];

            // Each thread does its own sort
            Sort(array + threadStartIndex, C + threadStartIndex, currentSubArraySize, splitNumber, NULL);

            uint32_t leftOverThreadsCounter, groupNumber, mergeHeadThreadNum, arraySizesIndex, numPerMergeThreads, leftOverThreads, deferedSubArray = 0, deferedSize = 0;

            //check if odd number of subarrays
            if (numberOfSubArrays % 2 == 1) {
                deferedSubArray = 1; //acts like a boolean
                deferedSize = arraySizes[numberOfSubArrays - 1];
                numberOfSubArrays--;
            }

            //begin merging
            #pragma omp barrier
            while (currentSubArraySize < array_length && (numberOfSubArrays > 1 || deferedSubArray)) {
                currentSubArraySize = arraySizes[0];
                numPerMergeThreads = numberOfThreads/(numberOfSubArrays/2);
                leftOverThreads = numberOfThreads%(numberOfSubArrays/2);

                //determines which threads will merge which sub arrays
                leftOverThreadsCounter = leftOverThreads;
                groupNumber = 0;
                mergeHeadThreadNum = 0;
                for (uint32_t i = 0; i < (uint32_t)numberOfThreads && numPerMergeThreads != 0; ) {
                    if (leftOverThreadsCounter) {
                        leftOverThreadsCounter--;
                        i += (numPerMergeThreads + 1);
                    } else {
                        i += numPerMergeThreads;
                    }
                    if (threadNum < i) {
                        break;
                    }
                    mergeHeadThreadNum = i;
                    groupNumber++;
                }

                arraySizesIndex = groupNumber*2; //points to index of A in array sizes for this thread

                //now asign left over threads starting at the front
                for (uint32_t i = 0; i < leftOverThreads; i++) {
                    if (threadNum >= (numPerMergeThreads+1)*i && threadNum < (numPerMergeThreads+1)*(i+1)) {
                        numPerMergeThreads++;
                    }
                }

                uint32_t AStartMergePath = arraySum(arraySizes, arraySizesIndex);
                uint32_t BStartMergePath = AStartMergePath + arraySizes[arraySizesIndex];

                MergePathSplitter(
                    array + AStartMergePath, arraySizes[arraySizesIndex],
                    array + BStartMergePath, arraySizes[arraySizesIndex + 1],
                    C + AStartMergePath, arraySizes[arraySizesIndex] + arraySizes[arraySizesIndex + 1],
                    numPerMergeThreads,
                    ASplitters + mergeHeadThreadNum, BSplitters + mergeHeadThreadNum); //Splitters[subArrayStart thread num] should be index zero

                uint32_t A_start = AStartMergePath + ASplitters[threadNum];
                uint32_t A_end = AStartMergePath + ASplitters[threadNum + 1];
                uint32_t A_length = A_end - A_start;
                uint32_t B_start = BStartMergePath + BSplitters[threadNum];
                uint32_t B_end = BStartMergePath + BSplitters[threadNum + 1];
                uint32_t B_length = B_end - B_start;
                uint32_t C_start = ASplitters[threadNum] + BSplitters[threadNum] + AStartMergePath; //start C at offset of previous
                uint32_t C_length = A_length + B_length;

                Merge(array + A_start, A_length, array + B_start, B_length, C + C_start, C_length, NULL);

                //number of sub arrays is now cut in half
                numberOfSubArrays = numberOfSubArrays/2;

                //Add up array sizes
                for (uint32_t i = 0, index = 0; i < numberOfSubArrays; i++) {
                    arraySizes[i] = arraySizes[index] + arraySizes[index + 1];
                    index += 2;
                }

                //Take care of odd sized number of sub arrays
                //If there is a defered sub array, we need to
                //Manually copy it here
                if (numberOfSubArrays % 2 == 1 && deferedSubArray) {
                    #pragma omp single
                    {
                        memcpy((void*)(C+array_length-deferedSize),
                            (void*)(array+array_length-deferedSize),
                            deferedSize*sizeof(vec_t));
                    }
                    deferedSubArray = 0;
                    arraySizes[numberOfSubArrays] = deferedSize;
                    numberOfSubArrays++;
                } else if (numberOfSubArrays % 2 == 1 && numberOfSubArrays != 1) {
                    deferedSubArray = 1;
                    deferedSize = arraySizes[numberOfSubArrays - 1];
                    numberOfSubArrays--;
                } else if (deferedSubArray) {
                    #pragma omp single
                    {
                        memcpy((void*)(C+array_length-deferedSize),
                            (void*)(array+array_length-deferedSize),
                            deferedSize*sizeof(vec_t));
                    }
                }

                //swap pointers
                #pragma omp barrier
                #pragma omp single
                {
                    vec_t* tmp = array;
                    array = C;
                    C = tmp;
                    numberOfSwaps++;
                }
            }
        }
    }
    //must return original array
    if (numberOfSwaps > 0 && numberOfSwaps%2 == 1) {
        memcpy((void*)C,(void*)array, (array_length+32)*sizeof(vec_t));
        vec_t* tmp = array;
        array = C;
        C = tmp;
    }
}

template <SortTemplate Sort, MergeTemplate Merge>
void parallelIterativeMergeSortPower2(
    vec_t* array, vec_t* C, uint32_t array_length, const uint32_t splitNumber, struct memPointers* pointers)
{
    int numberOfSwaps = 0;
    #pragma omp parallel
    {
        uint32_t threadNum = omp_get_thread_num();
        uint32_t numberOfThreads = omp_get_num_threads();

        // Initialize each threads memory
        uint32_t* ASplitters = pointers->ASplitters + (numberOfThreads + 1) * threadNum;
        uint32_t* BSplitters = pointers->BSplitters + (numberOfThreads + 1) * threadNum;

        uint32_t subArraySize = array_length / numberOfThreads;

        // Each thread does its own sort
        Sort(array + subArraySize*threadNum, C + subArraySize*threadNum, subArraySize, splitNumber, NULL);

        uint32_t numberOfSubArrays = numberOfThreads;

        #pragma omp barrier
        // Begin merging
        while (subArraySize < array_length) {
            uint32_t numPerMergeThreads = numberOfThreads/(numberOfSubArrays/2);
            uint32_t mergeGroupNumber = threadNum/numPerMergeThreads;

            MergePathSplitter(
                array + subArraySize*mergeGroupNumber*2, subArraySize,
                array + subArraySize*mergeGroupNumber*2 + subArraySize, subArraySize,
                C + subArraySize*mergeGroupNumber*2, subArraySize*2,
                numPerMergeThreads,
                ASplitters, BSplitters);

            uint32_t A_start = subArraySize*mergeGroupNumber*2 + ASplitters[threadNum%numPerMergeThreads];
            uint32_t A_end = subArraySize*mergeGroupNumber*2 + ASplitters[threadNum%numPerMergeThreads + 1];
            uint32_t A_length = A_end - A_start;
            uint32_t B_start = subArraySize*mergeGroupNumber*2 + subArraySize + BSplitters[threadNum%numPerMergeThreads];
            uint32_t B_end = subArraySize*mergeGroupNumber*2 + subArraySize + BSplitters[threadNum%numPerMergeThreads + 1];
            uint32_t B_length = B_end - B_start;
            uint32_t C_start = ASplitters[threadNum%numPerMergeThreads] + BSplitters[threadNum%numPerMergeThreads] + subArraySize*mergeGroupNumber*2;
            uint32_t C_length = A_length + B_length;

            Merge(array + A_start, A_length, array + B_start, B_length, C + C_start, C_length, NULL);

            numberOfSubArrays /= 2;
            subArraySize *= 2;

            //swap pointers
            #pragma omp barrier
            #pragma omp single
            {
                vec_t* tmp = array;
                array = C;
                C = tmp;
                numberOfSwaps++;
            }
        }
    }

    //must return original array
    if (numberOfSwaps > 0 && numberOfSwaps%2 == 1) {
        memcpy((void*)C,(void*)array, (array_length+32)*sizeof(vec_t));
        vec_t* tmp = array;
        array = C;
        C = tmp;
    }
}

template <SortTemplate Sort, MergeTemplate Merge>
void parallelIterativeMergeSortV2(
    vec_t* array, vec_t* C, uint32_t array_length, const uint32_t splitNumber, struct memPointers* pointers)
{
    int earlyEnd = 1; //Set to zero if small sub array or error
    int numberOfSwaps = 0;

    #pragma omp parallel
    {
        //algorithm does not support array_length < num threads
        //for that case, just quick sort
        #pragma omp single
        {
            if (array_length < (uint32_t)omp_get_num_threads()) {
                quickSort(array, C, array_length, splitNumber, NULL);
                earlyEnd = 0;
            }
        }

        if (earlyEnd) {
            // Heres whats happening here:
            // We basically want a private copy of these three variables for each thread
            // However, if we use the openmp private declaration, it will take longer to create each thread
            // because of the extra memory allocations. We don't want excess memory allocations to throw off
            // the timing of the algorithm. Therefore,  we pass in a variable with enough memory space for every
            // thread. Then we calculate the offset from those pointers.
            uint32_t* ASplitters = pointers->ASplitters + (omp_get_num_threads() + 1) * omp_get_thread_num();
            uint32_t* BSplitters = pointers->BSplitters + (omp_get_num_threads() + 1) * omp_get_thread_num();
            uint32_t* arraySizes = pointers->arraySizes + omp_get_num_threads() * omp_get_thread_num();

            uint32_t threadNum = omp_get_thread_num();
            uint32_t numberOfSubArrays = omp_get_num_threads();
            uint32_t initialSubArraySize = array_length / omp_get_num_threads();

            //Calculate the size of each subarray
            for (uint32_t thread = 0; thread < (uint32_t)omp_get_num_threads(); thread++) {
                arraySizes[thread] = initialSubArraySize;
                if (((thread % 2) == 1) && thread < 2*(array_length % (uint32_t)omp_get_num_threads())) {
                    arraySizes[thread]++;
                } else if ((array_length % (uint32_t)omp_get_num_threads()) > numberOfSubArrays/2 && thread < 2*((array_length % (uint32_t)omp_get_num_threads()) - numberOfSubArrays/2)) {
                    arraySizes[thread]++;
                }
            }

            uint32_t threadStartIndex = arraySum(arraySizes, threadNum);
            uint32_t currentSubArraySize = arraySizes[threadNum];

            // Each thread does its own sort
            Sort(array + threadStartIndex, C + threadStartIndex, currentSubArraySize, splitNumber, NULL);

            #pragma omp barrier
            while (currentSubArraySize < array_length && numberOfSubArrays > 1) {
                for (uint32_t mergeColumnIndex = 0; mergeColumnIndex < numberOfSubArrays;) {
                    #pragma omp barrier
                    uint32_t mergeColumnOffset = arraySum(arraySizes, mergeColumnIndex);

                    MergePathSplitter(
                        array + mergeColumnOffset, arraySizes[mergeColumnIndex],
                        array + mergeColumnOffset + arraySizes[mergeColumnIndex], arraySizes[mergeColumnIndex + 1],
                        C + mergeColumnOffset, arraySizes[mergeColumnIndex] + arraySizes[mergeColumnIndex + 1],
                        omp_get_num_threads(),
                        ASplitters, BSplitters);

                    uint32_t A_start = mergeColumnOffset + ASplitters[omp_get_thread_num()];
                    uint32_t A_length = ASplitters[omp_get_thread_num() + 1] - ASplitters[omp_get_thread_num()];
                    uint32_t B_start = mergeColumnOffset + arraySizes[mergeColumnIndex] + BSplitters[omp_get_thread_num()];
                    uint32_t B_length = BSplitters[omp_get_thread_num() + 1] - BSplitters[omp_get_thread_num()];
                    uint32_t C_start = mergeColumnOffset + ASplitters[omp_get_thread_num()] + BSplitters[omp_get_thread_num()];
                    uint32_t C_length = (ASplitters[omp_get_thread_num() + 1] + BSplitters[omp_get_thread_num() + 1]) - (ASplitters[omp_get_thread_num()] + BSplitters[omp_get_thread_num()]);

                    Merge(array + A_start, A_length, array + B_start, B_length, C + C_start, C_length, NULL);

                    if ((numberOfSubArrays - mergeColumnIndex) == 3) {
                        // We have an odd number of sub arrays
                        #pragma omp barrier
                        #pragma omp single
                        {
                            memcpy(array + mergeColumnOffset, C + mergeColumnOffset, (arraySizes[mergeColumnIndex] + arraySizes[mergeColumnIndex + 1]) * sizeof(vec_t));
                        }
                        arraySizes[mergeColumnIndex] = arraySizes[mergeColumnIndex] + arraySizes[mergeColumnIndex + 1];
                        arraySizes[mergeColumnIndex + 1] = arraySizes[mergeColumnIndex + 2];
                        numberOfSubArrays--;
                    } else {
                        mergeColumnIndex += 2;
                    }
                }
                #pragma omp barrier
                //number of sub arrays is now cut in half
                numberOfSubArrays = numberOfSubArrays/2;

                //Add up array sizes
                for (uint32_t i = 0, index = 0; i < numberOfSubArrays; i++) {
                    arraySizes[i] = arraySizes[index] + arraySizes[index + 1];
                    index += 2;
                }

                //swap pointers
                #pragma omp barrier
                #pragma omp single
                {
                    vec_t* tmp = array;
                    array = C;
                    C = tmp;
                    numberOfSwaps++;
                }
            }

            //must return original array
            if (numberOfSwaps > 0 && numberOfSwaps%2 == 1) {
                memcpy((void*)C,(void*)array, (array_length+32)*sizeof(vec_t));
                vec_t* tmp = array;
                array = C;
                C = tmp;
            }
        }
    }
}


/*
 * Template Instantiations
 */

template void iterativeMergeSort<serialMerge>(vec_t* array, vec_t* C, uint32_t array_length, const uint32_t splitNumber, struct memPointers* pointers);
template void iterativeMergeSort<serialMergeNoBranch>(vec_t* array, vec_t* C, uint32_t array_length, const uint32_t splitNumber, struct memPointers* pointers);
template void iterativeMergeSort<bitonicMergeReal>(vec_t* array, vec_t* C, uint32_t array_length, const uint32_t splitNumber, struct memPointers* pointers);
#ifdef AVX512
template void iterativeMergeSort<avx512Merge>(vec_t* array, vec_t* C, uint32_t array_length, const uint32_t splitNumber, struct memPointers* pointers);
#endif

template void avx512SortNoMergePath<serialMerge>(vec_t* array, vec_t* C, uint32_t array_length, const uint32_t splitNumber, struct memPointers* pointers);
template void avx512SortNoMergePath<serialMergeNoBranch>(vec_t* array, vec_t* C, uint32_t array_length, const uint32_t splitNumber, struct memPointers* pointers);
template void avx512SortNoMergePath<bitonicMergeReal>(vec_t* array, vec_t* C, uint32_t array_length, const uint32_t splitNumber, struct memPointers* pointers);
#ifdef AVX512
template void avx512SortNoMergePath<avx512Merge>(vec_t* array, vec_t* C, uint32_t array_length, const uint32_t splitNumber, struct memPointers* pointers);
#endif

template void avx512SortNoMergePathV2<serialMerge>(vec_t* array, vec_t* C, uint32_t array_length, const uint32_t splitNumber, struct memPointers* pointers);
template void avx512SortNoMergePathV2<serialMergeNoBranch>(vec_t* array, vec_t* C, uint32_t array_length, const uint32_t splitNumber, struct memPointers* pointers);
template void avx512SortNoMergePathV2<bitonicMergeReal>(vec_t* array, vec_t* C, uint32_t array_length, const uint32_t splitNumber, struct memPointers* pointers);
#ifdef AVX512
template void avx512SortNoMergePathV2<avx512Merge>(vec_t* array, vec_t* C, uint32_t array_length, const uint32_t splitNumber, struct memPointers* pointers);
#endif

template void iterativeMergeSortPower2<serialMerge>(vec_t* array, vec_t* C, uint32_t array_length, const uint32_t splitNumber, struct memPointers* pointers);
template void iterativeMergeSortPower2<serialMergeNoBranch>(vec_t* array, vec_t* C, uint32_t array_length, const uint32_t splitNumber, struct memPointers* pointers);
template void iterativeMergeSortPower2<bitonicMergeReal>(vec_t* array, vec_t* C, uint32_t array_length, const uint32_t splitNumber, struct memPointers* pointers);
#ifdef AVX512
template void iterativeMergeSortPower2<avx512Merge>(vec_t* array, vec_t* C, uint32_t array_length, const uint32_t splitNumber, struct memPointers* pointers);
#endif

template void parallelIterativeMergeSort<iterativeMergeSort<serialMerge>,serialMerge>(vec_t* array, vec_t* C, uint32_t array_length, const uint32_t splitNumber, struct memPointers* pointers);
template void parallelIterativeMergeSort<iterativeMergeSort<serialMergeNoBranch>,serialMergeNoBranch>(vec_t* array, vec_t* C, uint32_t array_length, const uint32_t splitNumber, struct memPointers* pointers);
template void parallelIterativeMergeSort<iterativeMergeSort<bitonicMergeReal>,bitonicMergeReal>(vec_t* array, vec_t* C, uint32_t array_length, const uint32_t splitNumber, struct memPointers* pointers);
#ifdef AVX512
template void parallelIterativeMergeSort<iterativeMergeSort<avx512Merge>,avx512Merge>(vec_t* array, vec_t* C, uint32_t array_length, const uint32_t splitNumber, struct memPointers* pointers);
#endif

template void parallelIterativeMergeSortPower2<iterativeMergeSort<serialMerge>,serialMerge>(vec_t* array, vec_t* C, uint32_t array_length, const uint32_t splitNumber, struct memPointers* pointers);
template void parallelIterativeMergeSortPower2<iterativeMergeSort<serialMergeNoBranch>,serialMergeNoBranch>(vec_t* array, vec_t* C, uint32_t array_length, const uint32_t splitNumber, struct memPointers* pointers);
template void parallelIterativeMergeSortPower2<iterativeMergeSort<bitonicMergeReal>,bitonicMergeReal>(vec_t* array, vec_t* C, uint32_t array_length, const uint32_t splitNumber, struct memPointers* pointers);
#ifdef AVX512
template void parallelIterativeMergeSortPower2<iterativeMergeSort<avx512Merge>,avx512Merge>(vec_t* array, vec_t* C, uint32_t array_length, const uint32_t splitNumber, struct memPointers* pointers);
#endif

template void parallelIterativeMergeSortV2<iterativeMergeSort<serialMerge>,serialMerge>(vec_t* array, vec_t* C, uint32_t array_length, const uint32_t splitNumber, struct memPointers*);
template void parallelIterativeMergeSortV2<iterativeMergeSort<serialMergeNoBranch>,serialMergeNoBranch>(vec_t* array, vec_t* C, uint32_t array_length, const uint32_t splitNumber, struct memPointers*);
template void parallelIterativeMergeSortV2<iterativeMergeSort<bitonicMergeReal>,bitonicMergeReal>(vec_t* array, vec_t* C, uint32_t array_length, const uint32_t splitNumber, struct memPointers*);
#ifdef AVX512
template void parallelIterativeMergeSortV2<iterativeMergeSort<avx512Merge>,avx512Merge>(vec_t* array, vec_t* C, uint32_t array_length, const uint32_t splitNumber, struct memPointers*);
#endif

#endif
