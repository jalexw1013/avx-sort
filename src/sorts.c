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


void bitonicMergeReal(vec_t* A, uint32_t A_length,
                      vec_t* B, uint32_t B_length,
                      vec_t* C, uint32_t C_length){
    uint32_t Aindex = 0,Bindex = 0, Cindex = 0;
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
    if (temp[3] <= A[Aindex])
    {
        Aindex -= 4;
        for(int ii=0; ii < 4; ii++)
        {
            A[Aindex + ii] = temp[ii];
        }
    }
    else
    {
        Bindex -= 4;
        for(int ii=0; ii < 4; ii++)
        {
            B[Bindex + ii] = temp[ii];
        }
    }
    for (Cindex; Cindex < C_length; Cindex++)
    {
        if (Aindex < A_length && Bindex < B_length)
        {
            if (A[Aindex] < B[Bindex])
            {
                C[Cindex] = A[Aindex];
                Aindex++;
            }
            else
            {
                C[Cindex] = B[Bindex];
                Aindex++;
            }
        }
        else
        {
            while (Aindex < A_length)
            {
                C[Cindex] = A[Aindex];
                Aindex++;
                Cindex++;
            }
            while (Bindex < B_length)
            {
                C[Cindex] = B[Bindex];
                Bindex++;
                Cindex++;
            }
        }
    }
    return;
}

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

void serialMergeAVX512(vec_t* A, int32_t A_length,
    vec_t* B, int32_t B_length,
    vec_t* C, uint32_t C_length,
    uint32_t* ASplitters, uint32_t* BSplitters) {

        // for (int i = 0; i < A_length; i++) {
        //     printf("A[%i]:%i\n", i, A[i]);
        // }
        //
        // for (int i = 0; i < B_length; i++) {
        //     printf("B[%i]:%i\n", i, B[i]);
        // }
        //
        // for (int i = 0; i < 17; i++) {
        //     printf("ASplitters[%i]:%i\n", i, ASplitters[i]);
        // }
        //
        // for (int i = 0; i < 17; i++) {
        //     printf("BSplitters[%i]:%i\n", i, BSplitters[i]);
        // }
        //
        // for (int i = 0; i < C_length; i++) {
        //     printf("C[%i]:%i\n", i, C[i]);
        // }

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
        __m512i mizero = _mm512_set_epi32(0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0);
        __m512i mione = _mm512_set_epi32(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1);
        __m512i minegone = _mm512_set_epi32(-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1);

        __mmask16 exceededAStop = _mm512_cmpgt_epi32_mask(vindexAStop, vindexA);
        __mmask16 exceededBStop = _mm512_cmpgt_epi32_mask(vindexBStop, vindexB);

        __m512i miPreviousCelems = _mm512_set_epi32(0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0);

        // printmmask16("A Stop", exceededAStop);
        // printmmask16("B Stop", exceededBStop);

        while ((exceededAStop | exceededBStop) != 0) {
            //get the current elements
            __m512i miAelems = _mm512_mask_i32gather_epi32(miPreviousCelems, exceededAStop, vindexA, A, 4);
            __m512i miBelems = _mm512_mask_i32gather_epi32(miPreviousCelems, exceededBStop, vindexB, B, 4);

            // printf("Test\n");
            //
            // print512_num("A Elements", miAelems);
            // print512_num("B Elements", miBelems);

            //compare the elements
            __mmask16 micmp = _mm512_cmple_epi32_mask(miAelems, miBelems);
            micmp = (micmp & exceededAStop);
            micmp = (~exceededBStop | micmp);

            //printmmask16("compare", micmp);

            //copy the elements to the final elements
            __m512i miCelems = _mm512_mask_add_epi32(miBelems, micmp, miAelems, mizero);
            miCelems = _mm512_mask_add_epi32(miCelems, (~exceededAStop) & (~exceededBStop), miPreviousCelems, mizero);
            miPreviousCelems = miCelems;

            // print512_num("C Elements", miCelems);
            // print512_num("V index C", vindexC);


            _mm512_mask_i32scatter_epi32(C, exceededAStop | exceededBStop, vindexC, miCelems, 4);
            //_mm512_i32scatter_epi32(C, vindexC, miCelems, 4);

            // for (int i = 0; i < C_length; i++) {
            //     printf("C[%i]:%i\n", i, C[i]);
            // }

            //exceededAStop = _mm512_cmpgt_epi32_mask(vindexAStop, vindexA);
            //exceededBStop = _mm512_cmpgt_epi32_mask(vindexBStop, vindexB);

            vindexA = _mm512_mask_add_epi32(vindexA, exceededAStop & micmp, vindexA, mione);
            vindexB = _mm512_mask_add_epi32(vindexB, exceededBStop & ~micmp, vindexB, mione);

            exceededAStop = _mm512_cmpgt_epi32_mask(vindexAStop, vindexA);
            exceededBStop = _mm512_cmpgt_epi32_mask(vindexBStop, vindexB);

            vindexC = _mm512_mask_add_epi32(vindexC, exceededAStop | exceededBStop, vindexC, mione);
            //vindexC = _mm512_add_epi32(vindexC, mione);

            //exceededAStop = _mm512_cmpgt_epi32_mask(vindexAStop, vindexA);
            //exceededBStop = _mm512_cmpgt_epi32_mask(vindexBStop, vindexB);

            // print512_num("V Index A", vindexA);
            // print512_num("V Index B", vindexB);
            // print512_num("V Index A Stop", vindexAStop);
            // print512_num("V Index B Stop", vindexBStop);

            //vindexA = _mm512_mask_add_epi32(vindexA, (~exceededAStop & micmp), vindexA, minegone);
            //vindexB = _mm512_mask_add_epi32(vindexB, (~exceededBStop & micmp), vindexB, minegone);
            //vindexC = _mm512_mask_add_epi32(vindexC, (~(exceededAStop | exceededBStop) & micmp), vindexC, minegone);

            // printmmask16("A Stop", exceededAStop);
            // printmmask16("B Stop", exceededBStop);
        }

        // for (int i = 0; i < C_length; i++) {
        //     printf("C[%i]:%i\n", i, C[i]);
        // }
}

/**
 * This function source Youngchao Liu http://lightpcc.sourceforge.net/homepage.htm#introduction
 */
/*static void AVX512mergeOutPlace(
    vec_t* A, int32_t A_length,
    vec_t* B, int32_t B_length,
    vec_t* C, uint32_t C_length){
		int l, r, p = left;

		int nbits, leadL, leadR, lead;
		const __m512i vecIndexInc = _mm512_set_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
		const __m512i vecReverse = _mm512_set_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
		const __m512i vecMaxInt = _mm512_set1_epi32(0x7fffffff);
		const __m512i vecMid = _mm512_set1_epi32(mid);
		const __m512i vecRight = _mm512_set1_epi32(right);
		const __m512i vecPermuteIndex16 = _mm512_set_epi32(7, 6, 5, 4, 3, 2, 1, 0, 15, 14, 13 ,12, 11, 10, 9, 8);
		const __m512i vecPermuteIndex8 = _mm512_set_epi32(11, 10, 9, 8, 15, 14, 13, 12, 3, 2, 1, 0, 7, 6, 5, 4);
	 	const __m512i vecPermuteIndex4 = _mm512_set_epi32(13, 12, 15, 14, 9, 8, 11, 10, 5, 4, 7, 6, 1, 0, 3, 2);
   	    const __m512i vecPermuteIndex2 = _mm512_set_epi32(14, 15, 12, 13, 10, 11, 8, 9, 6, 7, 4, 5, 2, 3, 0, 1);
		__m512i vecA, vecA2, vecB, vecB2, vecC, vecD, vecOL, vecOL2, vecOH, vecOH2;
		__m512i vecL1, vecH1, vecL2, vecH2, vecL3, vecH3, vecL4, vecH4;
		__mmask16 vecMaskA, vecMaskB, vecMaskOL, vecMaskOH;

		//use simd vectorization
		l = 0; r = 0;
		vecA = _mm512_load_epi32(A + l); l += 16;
		vecA2 = _mm512_load_epi32(A + l); l += 16;
		vecB = _mm512_load_epi32(B + r); r += 16;
		vecB2 = _mm512_load_epi32(B + r); r += 16;

		//enter the core loop
		do{
			//prefetch input[l] and input[r]
			//#pragma prefetch input

			if(_mm512_reduce_min_epi32(vecA) >= _mm512_reduce_max_epi32(vecB2)){
				_mm512_store_epi32(C + p, vecB); p += 16;
				_mm512_store_epi32(C + p, vecB2); p += 16;
				vecOH = vecA;
				vecOH2 = vecA2;
			}else if(_mm512_reduce_min_epi32(vecB) >= _mm512_reduce_max_epi32(vecA2)){
				_mm512_store_epi32(C + p, vecA); p += 16;
				_mm512_store_epi32(C + p, vecA2); p += 16;
				vecOH = vecB;
				vecOH2 = vecB2;
			}else{
				//in-register bitonic merge network
				vecB = _mm512_permutevar_epi32(vecReverse, vecB);	//reverse B
				vecB2 = _mm512_permutevar_epi32(vecReverse, vecB2);	//reverse B2
				vecC = vecB;
				vecB = vecB2;
				vecB2 = vecC;	//swap the content of the vectors
				//printVector(vecB, __LINE__);
				//printVector(vecB2, __LINE__);

				//Level 1
				vecL1 = _mm512_min_epi32(vecA, vecB);
				vecH1 = _mm512_max_epi32(vecA, vecB);
				vecL2 = _mm512_min_epi32(vecA2, vecB2);
				vecH2 = _mm512_max_epi32(vecA2, vecB2);
				//printVector(vecL1, __LINE__);
				//printVector(vecH1, __LINE__);

				//Level 2
				vecL3 = _mm512_min_epi32(vecL1, vecL2);
				vecL4 = _mm512_max_epi32(vecL1, vecL2);
				vecH3 = _mm512_min_epi32(vecH1, vecH2);
				vecH4 = _mm512_max_epi32(vecH1, vecH2);

				//Level 3
				vecA = _mm512_permutevar_epi32(vecPermuteIndex16, vecL3);
				vecB = _mm512_permutevar_epi32(vecPermuteIndex16, vecL4);
				vecC = _mm512_permutevar_epi32(vecPermuteIndex16, vecH3);
				vecD = _mm512_permutevar_epi32(vecPermuteIndex16, vecH4);
				vecL1 = _mm512_mask_min_epi32(vecL1, 0x00ff, vecA, vecL3);
                vecL2 = _mm512_mask_min_epi32(vecL2, 0x00ff, vecB, vecL4);
                vecH1 = _mm512_mask_min_epi32(vecH1, 0x00ff, vecC, vecH3);
                vecH2 = _mm512_mask_min_epi32(vecH2, 0x00ff, vecD, vecH4);
				vecL1 = _mm512_mask_max_epi32(vecL1, 0xff00, vecA, vecL3);
                vecL2 = _mm512_mask_max_epi32(vecL2, 0xff00, vecB, vecL4);
                vecH1 = _mm512_mask_max_epi32(vecH1, 0xff00, vecC, vecH3);
                vecH2 = _mm512_mask_max_epi32(vecH2, 0xff00, vecD, vecH4);
				//printVector(vecL2, __LINE__);
				//printVector(vecH2, __LINE__);

				//Level 4
				vecA = _mm512_permutevar_epi32(vecPermuteIndex8, vecL1);
				vecB = _mm512_permutevar_epi32(vecPermuteIndex8, vecL2);
                vecC = _mm512_permutevar_epi32(vecPermuteIndex8, vecH1);
                vecD = _mm512_permutevar_epi32(vecPermuteIndex8, vecH2);
              	vecL3 = _mm512_mask_min_epi32(vecL3, 0x0f0f, vecA, vecL1);
				vecL4 = _mm512_mask_min_epi32(vecL4, 0x0f0f, vecB, vecL2);
				vecH3 = _mm512_mask_min_epi32(vecH3, 0x0f0f, vecC, vecH1);
				vecH4 = _mm512_mask_min_epi32(vecH4, 0x0f0f, vecD, vecH2);
                vecL3 = _mm512_mask_max_epi32(vecL3, 0xf0f0, vecA, vecL1);
                vecL4 = _mm512_mask_max_epi32(vecL4, 0xf0f0, vecB, vecL2);
                vecH3 = _mm512_mask_max_epi32(vecH3, 0xf0f0, vecC, vecH1);
                vecH4 = _mm512_mask_max_epi32(vecH4, 0xf0f0, vecD, vecH2);

				//Level 5
                vecA = _mm512_permutevar_epi32(vecPermuteIndex4, vecL3);
                vecB = _mm512_permutevar_epi32(vecPermuteIndex4, vecL4);
              	vecC = _mm512_permutevar_epi32(vecPermuteIndex4, vecH3);
              	vecD = _mm512_permutevar_epi32(vecPermuteIndex4, vecH4);

              	vecL1 = _mm512_mask_min_epi32(vecL1, 0x3333, vecA, vecL3);
              	vecL2 = _mm512_mask_min_epi32(vecL2, 0x3333, vecB, vecL4);
				vecH1 = _mm512_mask_min_epi32(vecH1, 0x3333, vecC, vecH3);
				vecH2 = _mm512_mask_min_epi32(vecH2, 0x3333, vecD, vecH4);
                vecL1 = _mm512_mask_max_epi32(vecL1, 0xcccc, vecA, vecL3);
                vecL2 = _mm512_mask_max_epi32(vecL2, 0xcccc, vecB, vecL4);
                vecH1 = _mm512_mask_max_epi32(vecH1, 0xcccc, vecC, vecH3);
                vecH2 = _mm512_mask_max_epi32(vecH2, 0xcccc, vecD, vecH4);

				//Level 6
                vecA = _mm512_permutevar_epi32(vecPermuteIndex2, vecL1);
                vecB = _mm512_permutevar_epi32(vecPermuteIndex2, vecL2);
              	vecC = _mm512_permutevar_epi32(vecPermuteIndex2, vecH1);
              	vecD = _mm512_permutevar_epi32(vecPermuteIndex2, vecH2);

      	        vecOL = _mm512_mask_min_epi32(vecOL, 0x5555, vecA, vecL1);
				vecOL2 = _mm512_mask_min_epi32(vecOL2, 0x5555, vecB, vecL2);
      	        vecOH = _mm512_mask_min_epi32(vecOH, 0x5555, vecC, vecH1);
				vecOH2 = _mm512_mask_min_epi32(vecOH2, 0x5555, vecD, vecH2);
                vecOL = _mm512_mask_max_epi32(vecOL, 0xaaaa, vecA, vecL1);
                vecOL2 = _mm512_mask_max_epi32(vecOL2, 0xaaaa, vecB, vecL2);
                vecOH = _mm512_mask_max_epi32(vecOH, 0xaaaa, vecC, vecH1);
                vecOH2 = _mm512_mask_max_epi32(vecOH2, 0xaaaa, vecD, vecH2);

				//save vecL to the output vector: always memory aligned
				_mm512_store_epi32(C + p, vecOL); p += 16;
				_mm512_store_epi32(C + p, vecOL2); p += 16;
			}

			//condition check
			if(l + 32 >= mid || r + 32 >= right){
				break;
			}

			//determine which segment to use
			leadL = input[l];
			leadR = input[r];
			lead = _mm512_reduce_max_epi32(vecOH2);
			if(lead  < leadL && lead < leadR){
				_mm512_store_epi32(C + p, vecOH); p += 16;
				_mm512_store_epi32(C + p, vecOH2); p += 16;

				vecA = _mm512_load_epi32(A + l); l += 16;
				vecA2 = _mm512_load_epi32(A + l); l += 16;
				vecB = _mm512_load_epi32(B + r); r += 16;
				vecB2 = _mm512_load_epi32(B + r); r += 16;
			}else if(leadR < leadL){
				vecA = vecOH;
				vecA2 = vecOH2;
				vecB = _mm512_load_epi32(B + r); r += 16;
				vecB2 = _mm512_load_epi32(B + r); r += 16;
			}else{
				vecB = vecOH;
				vecB2 = vecOH2;
				vecA = _mm512_load_epi32(A + l); l += 16;
				vecA2 = _mm512_load_epi32(A + l); l += 16;
			}
		}while(1);

		//use non-vectorized code to process the leftover
	    if(l < mid && r < right){
		      if(input[r] < input[l]){
		        //write vecOH to the left segment
		        l -= 16; _mm512_store_epi32(A + l, vecOH2);
						l -= 16; _mm512_store_epi32(A + l, vecOH);
		      }else{
		        //write vecOH to the right segment
		        r -= 16; _mm512_store_epi32(B + r, vecOH2);
						r -= 16; _mm512_store_epi32(B + r, vecOH);
		      }
		    }else if (l < mid){
		      //write vecOH to the right segment
		      r -= 16; _mm512_store_epi32(B + r, vecOH2);
					r -= 16; _mm512_store_epi32(B + r, vecOH);
		    }else if(r < right){
		      //write vecOH to the left segment
		      l -= 16; _mm512_store_epi32(A + l, vecOH2);
					l -= 16; _mm512_store_epi32(A + l, vecOH);
		    }else{
					//write vecOH to the output as neither segment has leftover
					_mm512_store_epi32(C + p, vecOH); p += 16;
					_mm512_store_epi32(C + p, vecOH2);	p += 16;
			}

		//start serial merge
	   	while(l < mid && r < right) {
	    	if(B[r] < A[l]){
	      	//save the element from the right segment to temp array
	   			C[p++] = C[r++];
	    	}else{
	      	//save the element from the left segment to temp array
	     		C[p++] = A[l++];
	   		}
		}

		//copy the rest to the buffer
		if(l < mid){
			memcpy(C + p, A + l, (mid - l) * sizeof(uint32_t));
		}else if(r < right){
			memcpy(C + p, B + r, (B - r) * sizeof(uint32_t));
		}
	}*/

inline void iterativeComboMergeSort(vec_t* array, uint32_t array_length/*, void(*mergeFunction)(vec_t*,int32_t,vec_t*,int32_t,vec_t*,uint32_t)*/)
{
        vec_t* C = (vec_t*)xcalloc((array_length), sizeof(vec_t));
        //uint32_t initialSubArraySize = (array_length % omp_get_num_threads()) ? (array_length / omp_get_num_threads()) + 1 : (array_length / omp_get_num_threads());
        #pragma omp parallel
        {
            //parallelComboMergeSortParallelHelper(array, array_length, omp_get_num_threads(), , , C/*, mergeFunction*/);
            uint32_t threadNum = omp_get_thread_num();
            uint32_t initialSubArraySize = (array_length % omp_get_num_threads()) ? (array_length / omp_get_num_threads()) + 1 : (array_length / omp_get_num_threads());
            uint32_t start,stop;
            uint32_t i = threadNum*initialSubArraySize;
            start=i;
            stop=start+(i + initialSubArraySize < array_length)?initialSubArraySize:(array_length - i);
            // printf("%d %d %d %d\n",threadNum,start, stop, initialSubArraySize);
            //return;

            qsort(array + start,   stop, sizeof(vec_t), hostBasicCompare);

            #pragma omp barrier
            uint32_t currentSubArraySize = initialSubArraySize;
            while (currentSubArraySize < array_length) {
                if(threadNum==0)
                    printf("*");
                //merge one
                uint32_t A_start = threadNum * 2 * currentSubArraySize;
                if (A_start < array_length - 1) {
                    uint32_t B_start = min(A_start + currentSubArraySize - 1, array_length - 1);
                    uint32_t B_end = min(A_start + 2 * currentSubArraySize - 1, array_length - 1);
                    uint32_t A_length = B_start - A_start + 1;
                    uint32_t B_length = B_end - B_start;
                    bitonicMergeReal(array + A_start, A_length, array + B_start + 1, B_length, C + A_start, A_length + B_length);
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
                    bitonicMergeReal(C + A_start, A_length, C + B_start + 1, B_length, array + A_start, A_length + B_length);
                }
                currentSubArraySize = 2 * currentSubArraySize;
                #pragma omp barrier
            }
        }

        free(C);
}

#ifdef __INTEL_COMPILER
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
        tic_reset();
    	for (uint32_t A_start = 0; A_start < array_length - 1; A_start += 2 * currentSubArraySize)
    	{
    		uint32_t B_start = min(A_start + currentSubArraySize - 1, array_length - 1);
    		uint32_t B_end = min(A_start + 2 * currentSubArraySize - 1, array_length - 1);
            uint32_t A_length = B_start - A_start + 1;
            uint32_t B_length = B_end - B_start;

            serialMerge((*array) + A_start, A_length, (*array) + B_start + 1, B_length, C + A_start, A_length + B_length);
    	}
        float tmpF = tic_sincelast();
        printf("Time at Size %i : %i\n", currentSubArraySize, tmpF);

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
        tic_reset();
    	for (uint32_t A_start = 0; A_start < array_length - 1; A_start += 2 * currentSubArraySize)
    	{
    		uint32_t B_start = min(A_start + currentSubArraySize - 1, array_length - 1);
    		uint32_t B_end = min(A_start + 2 * currentSubArraySize - 1, array_length - 1);
            uint32_t A_length = B_start - A_start + 1;
            uint32_t B_length = B_end - B_start;

            MergePathSplitter((*array) + A_start, A_length, (*array) + B_start + 1, B_length, C + A_start, A_length + B_length, 16, ASplitters, BSplitters);
            serialMergeAVX512((*array) + A_start, A_length, (*array) + B_start + 1, B_length, C + A_start, A_length + B_length, ASplitters, BSplitters);
    	}
        float tmpF = tic_sincelast();
        printf("Time at Size %i : %i\n", currentSubArraySize, tmpF);

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
                MergePathSplitter((*array) + A_start, A_length, (*array) + B_start + 1, B_length, C + A_start, A_length + B_length, 16, ASplitters, BSplitters);
                serialMergeAVX512((*array) + A_start, A_length, (*array) + B_start + 1, B_length, C + A_start, A_length + B_length, ASplitters, BSplitters);
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
                MergePathSplitter((*array) + A_start, A_length, (*array) + B_start + 1, B_length, C + A_start, A_length + B_length, 16, ASplitters, BSplitters);
                serialMergeAVX512((*array) + A_start, A_length, (*array) + B_start + 1, B_length, C + A_start, A_length + B_length, ASplitters, BSplitters);
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
