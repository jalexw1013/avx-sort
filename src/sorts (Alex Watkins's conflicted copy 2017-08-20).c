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

inline void serialMergeNoBranch(
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

inline void bitonicMergeReal(vec_t* A, uint32_t A_length,
                      vec_t* B, uint32_t B_length,
                      vec_t* C, uint32_t C_length)
{
    // printf("A_length:%i\n", A_length);
    // printf("B_length:%i\n", B_length);
    // TODO i think these can be 4s
    if (A_length < 5 || B_length < 5 || C_length < 5) {
        serialMerge(A,A_length,B,B_length,C,C_length);
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
#ifdef __INTEL_COMPILER
inline void bitonicAVX512Merge(vec_t* A, uint32_t A_length,
                            vec_t* B, uint32_t B_length,
                            vec_t* C, uint32_t C_length){
	int l, r, p = 0;

    uint32_t* output = C;
    int BSize = (int)B_length;
    int ASize = (int)A_length;

	int nbits, leadL, leadR, lead;
	const __m512i vecIndexInc = _mm512_set_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
	const __m512i vecReverse = _mm512_set_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
	const __m512i vecMaxInt = _mm512_set1_epi32(0x7fffffff);
	//const __m512i vecMid = _mm512_set1_epi32(mid);
	//const __m512i vecRight = _mm512_set1_epi32(right);
	const __m512i vecPermuteIndex16 = _mm512_set_epi32(7, 6, 5, 4, 3, 2, 1, 0, 15, 14, 13 ,12, 11, 10, 9, 8);
	const __m512i vecPermuteIndex8 = _mm512_set_epi32(11, 10, 9, 8, 15, 14, 13, 12, 3, 2, 1, 0, 7, 6, 5, 4);
	const __m512i vecPermuteIndex4 = _mm512_set_epi32(13, 12, 15, 14, 9, 8, 11, 10, 5, 4, 7, 6, 1, 0, 3, 2);
   	const __m512i vecPermuteIndex2 = _mm512_set_epi32(14, 15, 12, 13, 10, 11, 8, 9, 6, 7, 4, 5, 2, 3, 0, 1);
	__m512i vecC, vecD, vecOL, vecOH;
	__m512i vecL1, vecH1, vecL2, vecH2, vecL3, vecH3, vecL4, vecH4;
	__mmask16 vecMaskOL, vecMaskOH;

    /*for short segments*/
    if(ASize < 16 || BSize < 16){
 	      l = 0; r = 0;
  	      while(l < ASize && r < BSize){
      	       if(B[r] < A[l]){
  	                /*save the element from the right segment to temp array*/
    	            output[p++] = B[r++];
     	       }else{
        	        /*save the element from the left segment to temp array*/
        	        output[p++] = A[l++];
      	       }
          }

          /*copy the remaining to the output buffer*/
          if(l < ASize){
              memcpy(output + p, A + l, (ASize - l) * sizeof(uint32_t));
          }else if(r < BSize){
            	memcpy(output + p, B + r, (BSize - r) * sizeof(uint32_t));
          }
	     /*return*/
	  	 return;
	}

	/*use simd vectorization*/
	l = 0; r = 0;
	vecOL = _mm512_load_epi32(A + l);
	vecOH = _mm512_load_epi32(B + r);
	l += 16; r += 16;

	/*enter the core loop*/
	do{
		if(_mm512_reduce_min_epi32(vecOL) >= _mm512_reduce_max_epi32(vecOH)){
			_mm512_store_epi32(output + p, vecOH);
			p += 16;
			vecOH = vecOL;
		}else if(_mm512_reduce_min_epi32(vecOH) >= _mm512_reduce_max_epi32(vecOL)){
			_mm512_store_epi32(output + p, vecOL);
			p += 16;
		}else{
			/*in-register bitonic merge network*/
			vecOH = _mm512_permutevar_epi32(vecReverse, vecOH);	/*reverse B*/

			/*Level 1*/
			vecL1 = _mm512_min_epi32(vecOL, vecOH);
			vecH1 = _mm512_max_epi32(vecOL, vecOH);
			//printVector(vecL1, __LINE__);
			//printVector(vecH1, __LINE__);

			/*Level 2*/
			vecC = _mm512_permutevar_epi32(vecPermuteIndex16, vecL1);
			vecD = _mm512_permutevar_epi32(vecPermuteIndex16, vecH1);
			vecL2 = _mm512_mask_min_epi32(vecL2, 0x00ff, vecC, vecL1);
			vecH2 = _mm512_mask_min_epi32(vecH2, 0x00ff, vecD, vecH1);
			vecL2 = _mm512_mask_max_epi32(vecL2, 0xff00, vecC, vecL1);
			vecH2 = _mm512_mask_max_epi32(vecH2, 0xff00, vecD, vecH1);
			//printVector(vecL2, __LINE__);
			//printVector(vecH2, __LINE__);

			/*Level 3*/
			vecC = _mm512_permutevar_epi32(vecPermuteIndex8, vecL2);
			vecD = _mm512_permutevar_epi32(vecPermuteIndex8, vecH2);
      	    vecL3 = _mm512_mask_min_epi32(vecL3, 0x0f0f, vecC, vecL2);
      	    vecH3 = _mm512_mask_min_epi32(vecH3, 0x0f0f, vecD, vecH2);
      	    vecL3 = _mm512_mask_max_epi32(vecL3, 0xf0f0, vecC, vecL2);
      	    vecH3 = _mm512_mask_max_epi32(vecH3, 0xf0f0, vecD, vecH2);
     		//printVector(vecL3, __LINE__);
     		//printVector(vecH3, __LINE__);

			/*Level 4*/
      	    vecC = _mm512_permutevar_epi32(vecPermuteIndex4, vecL3);
          	vecD = _mm512_permutevar_epi32(vecPermuteIndex4, vecH3);
          	vecL4 = _mm512_mask_min_epi32(vecL4, 0x3333, vecC, vecL3);
          	vecH4 = _mm512_mask_min_epi32(vecH4, 0x3333, vecD, vecH3);
          	vecL4 = _mm512_mask_max_epi32(vecL4, 0xcccc, vecC, vecL3);
          	vecH4 = _mm512_mask_max_epi32(vecH4, 0xcccc, vecD, vecH3);
     		//printVector(vecL4, __LINE__);
     		//printVector(vecH4, __LINE__);

			/*Level 5*/
          	vecC = _mm512_permutevar_epi32(vecPermuteIndex2, vecL4);
          	vecD = _mm512_permutevar_epi32(vecPermuteIndex2, vecH4);
          	vecOL = _mm512_mask_min_epi32(vecOL, 0x5555, vecC, vecL4);
          	vecOH = _mm512_mask_min_epi32(vecOH, 0x5555, vecD, vecH4);
          	vecOL = _mm512_mask_max_epi32(vecOL, 0xaaaa, vecC, vecL4);
          	vecOH = _mm512_mask_max_epi32(vecOH, 0xaaaa, vecD, vecH4);
			//printVector(vecOL, __LINE__);
			//printVector(vecOH, __LINE__);

			/*save vecL to the output vector: always memory aligned*/
			_mm512_store_epi32(output + p, vecOL);
			p += 16;
		}

		/*condition check*/
		if(l + 16 >= ASize || r + 16 >= BSize){
			break;
		}

		/*determine which segment to use*/
		leadL = A[l];
		leadR = B[r];
		lead = _mm512_reduce_max_epi32(vecOH);
		if(lead <= leadL && lead <= leadR){
			_mm512_store_epi32(output + p, vecOH);
			vecOL = _mm512_load_epi32(A + l);
			vecOH = _mm512_load_epi32(B + r);
			p += 16;
			l += 16;
			r += 16;
		}else if(leadR < leadL){
			vecOL = _mm512_load_epi32(A + r);
			r += 16;
		}else{
			vecOL = _mm512_load_epi32(B + l);
			l += 16;
		}
	}while(1);

    while(l < ASize && r < BSize){
        if(B[r] < A[l]){
            output[p++] = B[r++];
        }else{
            output[p++] = A[l++];
        }
    }
	/*copy the remaining to the output buffer*/
	if(l < ASize){
		memcpy(output + p, A + l, (ASize - l) * sizeof(uint32_t));
	}else if(r < BSize){
		memcpy(output + p, B + r, (BSize - r) * sizeof(uint32_t));
	}
}
#endif

inline void avx512Merge(
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

void quickSort(
    vec_t* array, uint32_t array_length, const uint32_t splitNumber)
{
    qsort((void*)array, array_length, sizeof(vec_t), hostBasicCompare);
}

template <MergeTemplate Merge>
void iterativeMergeSort(
    vec_t* array, uint32_t array_length, const uint32_t splitNumber)
{
    vec_t* C = (vec_t*)xcalloc((array_length + 32), sizeof(vec_t));

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

            Merge(array + A_start, A_length, array + B_start, B_length, C + A_start, A_length + B_length);
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

    free(C);
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
    vec_t* array, uint32_t array_length, const uint32_t splitNumber)
{
    vec_t* C = (vec_t*)xcalloc((array_length + 32), sizeof(vec_t));
    int earlyEnd = 1; //Set to zero if small sub array or error
    int numberOfSwaps = 0;

    #pragma omp parallel
    {
        //algorithm does not support array_length < num threads
        //for that case, just quick sort
        #pragma omp single
        {
            if (array_length < (uint32_t)omp_get_num_threads()) {
                quickSort(array, array_length, splitNumber);
                earlyEnd = 0;
            }
        }

        if (earlyEnd) {
            uint32_t* ASplitters = (uint32_t*)xcalloc(omp_get_num_threads() + 1, sizeof(uint32_t));
            uint32_t* BSplitters = (uint32_t*)xcalloc(omp_get_num_threads() + 1, sizeof(uint32_t));
            uint32_t* arraySizes = (uint32_t*)xcalloc(omp_get_num_threads(), sizeof(uint32_t));

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

            Sort(array + threadStartIndex, currentSubArraySize, splitNumber);

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
                numPerMergeThreads = omp_get_num_threads()/(numberOfSubArrays/2);
                leftOverThreads = omp_get_num_threads()%(numberOfSubArrays/2);

                leftOverThreadsCounter = leftOverThreads;
                groupNumber = 0;
                mergeHeadThreadNum = 0;
                for (uint32_t i = 0; i < (uint32_t)omp_get_num_threads() && numPerMergeThreads != 0; ) {
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

                if (threadNum == 48) {
                    MergePathSplitter2(
                        array + AStartMergePath, arraySizes[arraySizesIndex],
                        array + BStartMergePath, arraySizes[arraySizesIndex + 1],
                        C + AStartMergePath, arraySizes[arraySizesIndex] + arraySizes[arraySizesIndex + 1],
                        numPerMergeThreads,
                        ASplitters + mergeHeadThreadNum, BSplitters + mergeHeadThreadNum, 1);
                } else {

                MergePathSplitter(
                    array + AStartMergePath, arraySizes[arraySizesIndex],
                    array + BStartMergePath, arraySizes[arraySizesIndex + 1],
                    C + AStartMergePath, arraySizes[arraySizesIndex] + arraySizes[arraySizesIndex + 1],
                    numPerMergeThreads,
                    ASplitters + mergeHeadThreadNum, BSplitters + mergeHeadThreadNum); //Splitters[subArrayStart thread num] should be index zero
}

                uint32_t A_start = AStartMergePath + ASplitters[threadNum];
                uint32_t A_end = AStartMergePath + ASplitters[threadNum + 1];
                uint32_t A_length = A_end - A_start;
                uint32_t B_start = BStartMergePath + BSplitters[threadNum];
                uint32_t B_end = BStartMergePath + BSplitters[threadNum + 1];
                uint32_t B_length = B_end - B_start;
                uint32_t C_start = ASplitters[threadNum] + BSplitters[threadNum] + AStartMergePath; //start C at offset of previous
                uint32_t C_length = A_length + B_length;

                Merge(array + A_start, A_length, array + B_start, B_length, C + C_start, C_length);

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


                #pragma omp barrier
                #pragma omp single
                {
                    vec_t* tmp = array;
                    array = C;
                    C = tmp;
                    numberOfSwaps++;
                }
            }
            free(ASplitters);
            free(BSplitters);
            free(arraySizes);
        }
    }
    if (numberOfSwaps > 0 && numberOfSwaps%2 == 1) {
        memcpy((void*)C,(void*)array, (array_length+32)*sizeof(vec_t));
        vec_t* tmp = array;
        array = C;
        C = tmp;
    }
    free(C);
}

/*
 * Template Instantiations
 */

template void iterativeMergeSort<serialMerge>(vec_t* array, uint32_t array_length, const uint32_t splitNumber);
template void iterativeMergeSort<serialMergeNoBranch>(vec_t* array, uint32_t array_length, const uint32_t splitNumber);
template void iterativeMergeSort<bitonicMergeReal>(vec_t* array, uint32_t array_length, const uint32_t splitNumber);
#ifdef AVX512
template void iterativeMergeSort<avx512Merge>(vec_t* array, uint32_t array_length, const uint32_t splitNumber);
#endif

template void parallelIterativeMergeSort<iterativeMergeSort<serialMerge>,serialMerge>(vec_t* array, uint32_t array_length, const uint32_t splitNumber);
template void parallelIterativeMergeSort<iterativeMergeSort<serialMergeNoBranch>,serialMergeNoBranch>(vec_t* array, uint32_t array_length, const uint32_t splitNumber);
template void parallelIterativeMergeSort<iterativeMergeSort<bitonicMergeReal>,bitonicMergeReal>(vec_t* array, uint32_t array_length, const uint32_t splitNumber);
#ifdef AVX512
template void parallelIterativeMergeSort<iterativeMergeSort<avx512Merge>,avx512Merge>(vec_t* array, uint32_t array_length, const uint32_t splitNumber);
#endif
