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

/*void serialMergeIntrinsic( vec_t* A, int32_t A_length,
                                  vec_t* B, int32_t B_length,
                                  vec_t* C, uint32_t C_length){
    uint32_t ai = 0;
    uint32_t bi = 0;
    uint32_t ci = 0;

    __m128i mione = _mm_cvtsi32_si128(1);
    __m128i miand, miandnot;
    __m128i miAi = _mm_cvtsi32_si128(0);//ai);
    __m128i miBi = _mm_cvtsi32_si128(0);//bi);

    while(ai < A_length && bi < B_length) {
        __m128i miAelem = _mm_cvtsi32_si128(A[ai]);
        __m128i miBelem = _mm_cvtsi32_si128(B[bi]);
        __m128i micmp   = _mm_cmplt_epi32(miAelem,miBelem);
        miand           = _mm_and_si128(micmp,mione);
        miandnot        = _mm_andnot_si128(micmp,mione);
        miAelem         = _mm_and_si128(micmp,miAelem);
        miBelem         = _mm_andnot_si128(micmp,miBelem);
        miAi            = _mm_add_epi32(miAi,miand);
        miBi            = _mm_add_epi32(miBi,miandnot);
        C[ci++]         = _mm_cvtsi128_si32(_mm_add_epi32(miAelem,miBelem));
        ai              = _mm_cvtsi128_si32(miAi);
        bi              = _mm_cvtsi128_si32(miBi);
    }

    while(ai < A_length) {
        C[ci++] = A[ai++];
    }
    while(bi < B_length) {
        C[ci++] = B[bi++];
    }
}*/

/*void print256_num(__m256i var)
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
           val[0], val[1], val[2], val[3], val[4], val[5], val[6], val[7],
           val[8], val[9], val[10], val[11], val[12], val[13], val[14], val[15]);
    uint16_t num = (uint16_t)mask;
    printf("%s: %i\n", text, num);
}*/

void serialMergeAVX512(vec_t* A, int32_t A_length,
    vec_t* B, int32_t B_length,
    vec_t* C, uint32_t C_length,
    uint32_t* ASplitters, uint32_t* BSplitters) {

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
        __m512i miand, miandnot;
        __m512i miAi = _mm512_set_epi32(0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0);
        __m512i miBi = _mm512_set_epi32(0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0);
        int cmp[16] = {1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1};

        //temprary debugging Variables
        int a = 0;
        uint32_t *val = (uint32_t*) &vindexC;

        __mmask16 exceededAStop = _mm512_cmpge_epi32_mask(vindexAStop, vindexA);
        __mmask16 exceededBStop = _mm512_cmpge_epi32_mask(vindexBStop, vindexB);

        while ((exceededAStop & exceededBStop) != 0) {
            //code goes here

            //get the current elements
            __m512i miAelems = _mm512_i32gather_epi32(vindexA, A, 4);
            __m512i miBelems = _mm512_i32gather_epi32(vindexB, B, 4);

            //compare the elements
            __mmask16 micmp = _mm512_cmple_epi32_mask(miAelems, miBelems);
            micmp = (micmp & exceededAStop);
            micmp = (~exceededBStop | micmp);

            //copy the elements to the final elements
            __m512i miCelems = _mm512_mask_add_epi32(miBelems, micmp, miAelems, mizero);

            _mm512_i32scatter_epi32(C, vindexC, miCelems, 4);

            vindexA = _mm512_mask_add_epi32(vindexA, micmp, vindexA, mione);
            vindexB = _mm512_mask_add_epi32(vindexB, ~micmp, vindexB, mione);
            vindexC = _mm512_add_epi32(vindexC, mione);

            exceededAStop = _mm512_cmpge_epi32_mask(vindexAStop, vindexA);
            exceededBStop = _mm512_cmpge_epi32_mask(vindexBStop, vindexB);

            vindexA = _mm512_mask_add_epi32(vindexA, (~exceededAStop & micmp), vindexA, minegone);
            vindexB = _mm512_mask_add_epi32(vindexB, (~exceededBStop & micmp), vindexB, minegone);
            vindexC = _mm512_mask_add_epi32(vindexC, (~(exceededAStop | exceededBStop) & micmp), vindexC, minegone);


        }
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

/*#define PRINTEXTRA 0
#define STORE_AND_PRINT(str,reg) {if (PRINTEXTRA){vec_t arr[4]; _mm_storeu_si128((__m128i*)&arr,reg);printf("%s : %d %d %d %d\n", str, arr[3],arr[2],arr[1],arr[0]); }}

const uint8_t min1=(3<<6)| (3<<4)|(2<<2) |1;
const uint8_t max1=(2<<6)| (1<<4);

void mergeNetwork(vec_t* A, int32_t A_length,
                  vec_t* B, int32_t B_length,
                  vec_t* C, uint32_t C_length){
  uint32_t Aindex = 0,Bindex = 0, Cindex = 0;

  __m128i sA = _mm_loadu_si128((const __m128i*)&(A[Aindex]));
  __m128i sB = _mm_loadu_si128((const __m128i*)&(B[Bindex]));
  __m128i mifour = _mm_cvtsi32_si128(4);


   while ((Aindex < (A_length-4)) || (Bindex < (B_length-4)))
  {
      //STORE_AND_PRINT("original ", sA);
      //STORE_AND_PRINT("original ", sB);

      __m128i smin1 = _mm_min_epu32(sA, sB);
      __m128i smax1 = _mm_max_epu32(sA, sB);
      //STORE_AND_PRINT("first min", smin1);
      //STORE_AND_PRINT("first max", smax1);
      __m128i smin1shift = _mm_shuffle_epi32(smin1, (3<<6)| (3<<4)|(2<<2) |1);
      __m128i smax1shift = _mm_shuffle_epi32(smax1, (2<<6)| (1<<4));
      //STORE_AND_PRINT("shift min", smin1shift);
      //STORE_AND_PRINT("shift max", smax1shift);

      __m128i smin2 = _mm_min_epu32(smin1, smax1shift);
      __m128i smax2 = _mm_max_epu32(smax1, smin1shift);
      //STORE_AND_PRINT("2nd   min",smin2);
      //STORE_AND_PRINT("2nd   max",smax2);
      __m128i smin2shift = _mm_shuffle_epi32(smin2, (3<<6)| (3<<4)|(3<<2) |2);
      __m128i smax2shift = _mm_shuffle_epi32(smax2, (1<<6));
      //STORE_AND_PRINT("shift min", smin2shift);
      //STORE_AND_PRINT("shift max", smax2shift);

      __m128i smin3 = _mm_min_epu32(smin2, smax2shift);
      __m128i smax3 = _mm_max_epu32(smax2, smin2shift);
      //STORE_AND_PRINT("3rd   min",smin3);
      //STORE_AND_PRINT("3rd   max",smax3);
      __m128i smin3shift = _mm_shuffle_epi32(smin3, (3<<6)| (3<<4)|(3<<2) | 3);
      __m128i smax3shift = _mm_shuffle_epi32(smax3, 0);
      //STORE_AND_PRINT("shift min", smin3shift);
      //STORE_AND_PRINT("shift max", smax3shift);

      __m128i smin4 = _mm_min_epu32(smin3, smax3shift);
      __m128i smax4 = _mm_max_epu32(smax3, smin3shift);      //STORE_AND_PRINT("4th   min",smin4);
      //STORE_AND_PRINT("4th   max",smax4);

      #if (PRINTEXTRA==1)
        printf("\n");
      #endif
      _mm_storeu_si128((__m128i*)&(C[Cindex]), smin4);
      // calculate index for the next run
      sB=smax4;
      Cindex+=4;

      __m128i tempA = _mm_loadu_si128((const __m128i*)&(A[Aindex+4]));
      __m128i tempB = _mm_loadu_si128((const __m128i*)&(B[Bindex+4]));
      // __m128i tempAS = _mm_shuffle_epi32(tempA,0);
      // __m128i tempBS = _mm_shuffle_epi32(tempB,0);
      // __m128i cmpAB = _mm_cmplt_epi32(tempAS,tempBS);

      __m128i cmpAB = _mm_cmplt_epi32(_mm_shuffle_epi32(tempA,0),_mm_shuffle_epi32(tempB,0));

      __m128i addA  = _mm_and_si128(cmpAB,mifour);
      __m128i addB  = _mm_andnot_si128(cmpAB,mifour);
      Aindex       += _mm_cvtsi128_si32(addA);
      Bindex       += _mm_cvtsi128_si32(addB);

      tempA         = _mm_and_si128(cmpAB,tempA);
      tempB         = _mm_andnot_si128(cmpAB,tempB);
      sA            = _mm_add_epi32(tempA,tempB);

  //    printf("%d, %d, %d \n", Aindex, Bindex, Cindex);
  //    printf("%d, %d, %d \n", Aindex, Bindex, Cindex);
      // if (A[Aindex+4]<B[Bindex+4]) {
      //   Aindex+=4;
      //   sA = _mm_loadu_si128((const __m128i*)&(A[Aindex]));
      // }
      // else{
      //   Bindex+=4;
      //   sA = _mm_loadu_si128((const __m128i*)&(B[Bindex]));
      // }
    //    printf("INDICES are %d %d %d\n", Aindex,Bindex,Cindex);
    }//end while loop

 // printf("indices are %d %d %d\n", Aindex,Bindex,Cindex);
    vec_t last4[4]; int countlast4=0;
    _mm_storeu_si128((__m128i*)&(last4), sB);

    while (countlast4<4 && Aindex < A_length && Bindex < B_length){
      if (last4[countlast4] < A[Aindex] && A[Aindex] < B[Bindex]){
        C[Cindex++] = A[Aindex++];
        countlast4++;
      }
      else if (last4[countlast4] < B[Bindex] && B[Bindex] < A[Aindex]){
        C[Cindex++] = B[Bindex];
        countlast4++;
      }
      else{
        C[Cindex++] = A[Aindex] < B[Bindex] ? A[Aindex++] : B[Bindex++];
      }
    }

    while(Aindex < A_length && Bindex < B_length) {
      C[Cindex++] = A[Aindex] < B[Bindex] ? A[Aindex++] : B[Bindex++];
    }
    while(Aindex < A_length) C[Cindex++] = A[Aindex++];
    while(Bindex < B_length) C[Cindex++] = B[Bindex++];
 // printf("indices are %d %d %d\n", Aindex,Bindex,Cindex);

  //  float time_elapsed = tic_sincelast();
  // check sanity of results
    // for(int i = 0; i < C_length; ++i) {
    //   //assert(C[i] == globalC[i]);
    //   if(C[i]!=CSorted[i])
    //   {
    //     printf("\n %d,%d,%d \n", i,C[i],CSorted[i]);
    //     return;
    //   }
    // }
}*/

/**
 * this function is a recursive helper for the quick sort function
 */
/*void quickSortHelperRecursive(vec_t* arr, int a, int b) {
    assert(arr != NULL);
    if (a >= b) {
        return;
    }
    int left = a + 1;
    int right = b;
    int pivotIndex = (rand() % (b - a)) + a;
    uint32_t pivot = arr[pivotIndex];
    uint32_t temp;
    arr[pivotIndex] = arr[a];
    arr[a] = pivot;
    while (left <= right) {
        while (left <= right && arr[left] < pivot) {
            left++;
        }
        while (left <= right && arr[right] > pivot) {
            right--;
        }
        if (left <= right) {
            temp = arr[left];
            arr[left] = arr[right];
            arr[right] = temp;
            left++;
            right--;
        }
    }
    temp = arr[right];
    arr[right] = arr[a];
    arr[a] = temp;
    quickSortHelperRecursive(arr, a, right - 1);
    quickSortHelperRecursive(arr, right + 1, b);
}*/

/**
 * Implementation of quick sort.
 * Uses the the recursive quick sort helper function
 */
/*void quickSortRecursive(vec_t* arr, uint32_t arr_length) {
    srand(time(NULL));
    assert(arr != NULL);
    quickSortHelperRecursive(arr, 0, arr_length - 1);
}*/

/**
 * copies an array from input to dest only between the startIndex and endIndex
 * will fail if endIndex - startIndex > dest
 * not inclusive of the end index (ie to copy array of size 5 use index 0-5)
 */
void copyArrayInRange(vec_t* input, vec_t* dest, uint32_t startIndex, uint32_t endIndex) {
    assert(endIndex > startIndex);
    int counter = 0;
    for (int i = startIndex; i < endIndex; i++) {
        dest[counter] = input[i];
        counter++;
    }
}

/**
 * Implementation of merge sort.
 * Uses the the avx-512 merge function
 */
/*void mergeSortRecursive(vec_t* arr, uint32_t arr_length) {
    //validate array
    assert(arr != NULL);
    if (arr_length < 2) {
        return;
    }

    //split array in half
    uint32_t mid = arr_length / 2;
    vec_t a[mid];
    vec_t b[mid];
    copyArrayInRange(arr, a, 0, mid);
    copyArrayInRange(arr, b, mid, arr_length);

    //recursively split array again
    mergeSortRecursive(a, mid);
    mergeSortRecursive(b, (arr_length - mid));

    //merge parts
    serialMergeAVX512(
        a, (int32_t)mid,
        b, (int32_t)(arr_length - mid),
        arr, arr_length);
}*/

/*int uint32Compare(const void *one, const void *two) {
    uint32_t first = *(uint32_t*)one;
    uint32_t second = *(uint32_t*)two;
    if (first < second) {
        return -1;
    } else if (first > second) {
        return 1;
    } else {
        return 0;
    }
}*/

//must be multiple of cpus
/*void parallelComboSort(vec_t* array, uint32_t array_length,void(*mergeFunction)(vec_t*,int32_t,vec_t*,int32_t,vec_t*,uint32_t), int threads) {

    //allocate memory to swap array with
    vec_t* C = (vec_t*)xmalloc((array_length) * sizeof(vec_t));

    printf("Preparing to sort array!! Yay!!\n");
    for (int i = 0; i < array_length; i++) {
        printf("Array%i:%i\n", i, array[i]);
    }

    //Calculate variables for the number of arrays and elements
    int numSubArrays = threads;
    int elementsPerArray = array_length / numSubArrays;
    int numExtraElems = array_length % numSubArrays;

    //create splitters to mark beggining and end of each subarray
    //vec_t* splitters[numSubArrays + 1];
    int splittersLengths[numSubArrays + 1];
    splittersLengths[0] = 0;
    //splitters[0] = array;
    for (int i = 1; i < numSubArrays + 1; i++) {
        //splitters[i] = splitters[i - 1] + elementsPerArray;
        splittersLengths[i] = splittersLengths[i - 1] + elementsPerArray;
        if (numExtraElems > 0) {
            //splitters[i] += 1;
            splittersLengths[i]++;
            numExtraElems--;
        }
    }

    //create C splitters
    /*vec_t* splittersC[numSubArrays + 1];
    int splittersLengthsC[numSubArrays + 1];
    splittersLengthsC[0] = 0;
    splittersC[0] = array;

    /*printf("\n");
    for (int j = 0; j < numSubArrays + 1; j++) {
        printf("Splitters%i:%i\n", j, splittersLengths[j]);
    }

    //printf("\n");
    //#pragma omp parallel for
    for (int i = 0; i < numSubArrays; i++) {
        qsort((void*)(array + splittersLengths[i]), splittersLengths[i+1] - splittersLengths[i], sizeof(uint32_t), hostBasicCompare);
        /*printf("Just sorted this array!! Yay!!\n");
        for (int j = 0; j < splittersLengths[i+1] - splittersLengths[i]; j++) {
            printf("Sub%i:%i\n", j, (array + splittersLengths[i])[j]);
        }
        printf("Total array now!! Yay!!\n");
        for (int j = 0; j < array_length; j++) {
            printf("Array%i:%i\n", j, array[j]);
        }
        printf("\n");
    }
    //printf("\n");

    int count = 0;
    while (numSubArrays > 1) {
        for (int i = 0; i < numSubArrays - 1; i += 2) {
            printf("numSubArrays:%i\n", numSubArrays);
            printf("i%i\n", i);
            count++;
            /*printf("\n\nPreparing for Merge\n");
            printf("Array before Merge:\n");
            for (int j = 0; j < array_length; j++) {
                printf("Array%i:%i\n", j, array[j]);
            }
            printf("C before Merge:\n");
            for (int j = 0; j < array_length; j++) {
                printf("C%i:%i\n", j, C[j]);
            }
            printf("Merging Addresses:\n");
            printf("Array:%x\n", array);
            //printf("A:%x\n", splitters[i]);
            //printf("B:%x\n", splitters[i+1]);
            mergeFunction(array + splittersLengths[i], splittersLengths[i+1]-splittersLengths[i], array + splittersLengths[i+1], splittersLengths[i+2]-splittersLengths[i+1],C + splittersLengths[i], splittersLengths[i+2]-splittersLengths[i]);
            /*printf("Merging Done:\n");
            printf("Array after Merge:\n");
            for (int j = 0; j < array_length; j++) {
                printf("Array%i:%i\n", j, array[j]);
            }
            printf("C after Merge:\n");
            for (int j = 0; j < array_length; j++) {
                printf("C%i:%i\n", j, C[j]);
            }
            printf("\n");
        }
        if (numSubArrays % 2 == 1) {
            for (int i = splittersLengths[numSubArrays - 1]; i < splittersLengths[numSubArrays]; i++) {
                C[i] = array[i];
            }
        }

        copyArrayInRange(C, array, 0, array_length);
        if (numSubArrays < 4) {
            break;
        }

        /*printf("Reshuffling Splitters\n");
        printf("Splitters are before shuffle:\n");
        for (int j = 0; j < numSubArrays + 1; j++) {
            printf("Splitters%i:%i\n", j, splittersLengths[j]);
        }
        printf("There are:%i sub arrays\n", numSubArrays);
        int count = 0;
        for (int i = 0; i < numSubArrays + 1; i += 2) {
            splittersLengths[count++] = splittersLengths[i];
        }
        if (numSubArrays % 2 == 1) {
            splittersLengths[count++] = splittersLengths[numSubArrays];
        }
        for (;count < threads + 1; count++) {
            splittersLengths[count] = 0;
        }
        /*for (int j = 0; j < numSubArrays + 1; j++) {
            printf("Splitters%i:%i\n", j, splittersLengths[j]);
        }

        numSubArrays /= 2;
        if ((numSubArrays * 2) % 2 == 1 && numSubArrays % 2 == 1) {
            printf("\n\n\n\n\nYOOOOOOOO\n\n\n\n\n\n");
            numSubArrays++;
        }
        //copyArrayInRange(C, array, 0, array_length);
    }
    if (threads % 2 == 1) {
        //printf("Doing Final Merge.\n");
        //printf("A:%i Size:%i\n", 0, splittersLengths[numSubArrays]);
        //printf("B:%i Size:%i\n", splittersLengths[numSubArrays], splittersLengths[numSubArrays+1] - splittersLengths[numSubArrays]);
        //printf("Array before Merge:\n");
        //for (int j = 0; j < array_length; j++) {
        //    printf("Array%i:%i\n", j, array[j]);
        //}
        //printf("C before Merge:\n");
        //for (int j = 0; j < array_length; j++) {
        //    printf("C%i:%i\n", j, C[j]);
        //}
        mergeFunction(array, splittersLengths[numSubArrays], array + splittersLengths[numSubArrays], splittersLengths[numSubArrays+1] - splittersLengths[numSubArrays],C, splittersLengths[numSubArrays+1]);
        //printf("Merging Done:\n");
        //printf("Array after Merge:\n");
        //for (int j = 0; j < array_length; j++) {
        //    printf("Array%i:%i\n", j, array[j]);
        //}
        /*printf("C after Merge:\n");
        for (int j = 0; j < array_length; j++) {
            printf("C%i:%i\n", j, C[j]);
        }
        printf("\n");
    }
    copyArrayInRange(C, array, 0, array_length);

    printf("Array after Merge:\n");
    for (int j = 0; j < array_length; j++) {
        printf("Array%i:%i\n", j, array[j]);
    }
    printf("Count:%i\n", count);

    //just use single input and output and swap.

    /*numSubArrays /= 2;
    while (count > 0) {
        //#pragma omp parallel for
        for (int i = 0; i < numSubArrays; i++) {
            vec_t* C = (vec_t*)xmalloc((array_length/count) * sizeof(vec_t));
            mergeFunction(array + i*array_length/count, array_length/(count*2), array + i*array_length/count + array_length/(count*2), array_length/(count*2),C, array_length/count);
            copyArrayInRange(C, array + i*array_length/count, 0, array_length/count);
            free(C);
        }
        count /= 2;
    //}
}*/

//#define min(a,b) ({ __typeof__ (a) _a = (a); __typeof__ (b) _b = (b); _a < _b ? _a : _b; })

/*void iterativeComboMergeSort(vec_t* array, uint32_t array_length,void(*mergeFunction)(vec_t*,int32_t,vec_t*,int32_t,vec_t*,uint32_t), int threads) {
    vec_t* C = (vec_t*)xmalloc((array_length) * sizeof(vec_t));
    for (int subArraySize = 1; subArraySize <= array_length - 1; subArraySize *= 2) {
        for (int leftStart = 0; leftStart < array_length - 1; leftStart += 2*subArraySize) {
            printf("Hello\n");
            int middlePoint = leftStart + subArraySize - 1;
            int rightEnd = min(leftStart + 2*subArraySize - 1, array_length - 1);
            copyArrayInRange(array, C, 0, array_length);
            mergeFunction(array + leftStart, middlePoint - leftStart, array + middlePoint, rightEnd - middlePoint, C, rightEnd - leftStart);
            copyArrayInRange(C, array, 0, array_length);
        }
        printf("test\n");
    }
}*/

// Utility function to find minimum of two integers
//int min(int x, int y) { return (x<y)? x :y; }


/* Iterative mergesort function to sort arr[0...n-1] */
void iterativeComboMergeSort(vec_t* array, int32_t array_length)
{
    vec_t* C = (int*)xcalloc((array_length), sizeof(vec_t));

    for (uint32_t currentSubArraySize = 1; currentSubArraySize < array_length; currentSubArraySize = 2 * currentSubArraySize)
    {
    	for (uint32_t A_start = 0; A_start < array_length - 1; A_start += 2 * currentSubArraySize)
    	{
    		uint32_t B_start = min(A_start + currentSubArraySize - 1, array_length - 1);

    		uint32_t B_end = min(A_start + 2 * currentSubArraySize - 1, array_length - 1);

            uint32_t A_length = B_start - A_start + 1;
            uint32_t B_length = B_end - B_start;

            serialMerge(array + A_start, A_length, array + B_start + 1, B_length, C + A_start, A_length + B_length);
    	}

        currentSubArraySize = 2 * currentSubArraySize;
        if (currentSubArraySize >= array_length) {
            copyArrayInRange(C, array, 0, array_length);
            break;
        }

        for (uint32_t A_start = 0; A_start < array_length - 1; A_start += 2 * currentSubArraySize)
    	{
    		uint32_t B_start = min(A_start + currentSubArraySize - 1, array_length - 1);

    		uint32_t B_end = min(A_start + 2 * currentSubArraySize - 1, array_length - 1);

            uint32_t A_length = B_start - A_start + 1;
            uint32_t B_length = B_end - B_start;

            serialMerge(C + A_start, A_length, C + B_start + 1, B_length, array + A_start, A_length + B_length);
    	}
    }
    free(C);
}
