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
    vec_t** array, uint32_t array_length, const uint32_t splitNumber)
{
    qsort((void*)(*array), array_length, sizeof(vec_t), hostBasicCompare);
}

template <MergeTemplate Merge>
void iterativeMergeSort(
    vec_t** array, uint32_t array_length, const uint32_t splitNumber)
{
    vec_t* C = (vec_t*)xcalloc((array_length + 32), sizeof(vec_t));

    uint32_t start = splitNumber; //Just in case splitNumber is invalid

    if (splitNumber > 1) {
        //sort individual arrays of size splitNumber
        for (uint32_t i = 0; i < array_length; i += splitNumber) {
            //adjust when array_length is not divisible by splitNumber
            uint32_t actualSubArraySize = min(splitNumber, array_length - i);
            qsort((void*)((*array) + i), actualSubArraySize, sizeof(vec_t), hostBasicCompare);
        }
    } else {
        start = 1;
    }

    //now do actual iterative merge sort
    for (uint32_t currentSubArraySize = start; currentSubArraySize < array_length; currentSubArraySize = 2 * currentSubArraySize)
    {
    	for (uint32_t A_start = 0; A_start < array_length - 1; A_start += 2 * currentSubArraySize)
    	{
    		uint32_t B_start = min(A_start + currentSubArraySize - 1, array_length - 1);
    		uint32_t B_end = min(A_start + 2 * currentSubArraySize - 1, array_length - 1);
            uint32_t A_length = B_start - A_start + 1;
            uint32_t B_length = B_end - B_start;
            Merge((*array) + A_start, A_length, (*array) + B_start + 1, B_length, C + A_start, A_length + B_length);
    	}
        //pointer swap for C
        vec_t* tmp = *array;
        *array = C;
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

template <MergeTemplate Merge>
void parallelIterativeMergeSort(
    vec_t** array, uint32_t array_length, const uint32_t splitNumber)
{
        vec_t* C = (vec_t*)xcalloc((array_length + 32), sizeof(vec_t));
        int earlyEnd = 1; //Set to zero if small sub array or error

        #pragma omp parallel
        {
            //just quick sort for small array sizes
            #pragma omp single
            {
                if (array_length < (uint32_t)omp_get_num_threads()) {
                    quickSort(array, array_length, splitNumber);
                    earlyEnd = 0;
                }
            }

            //printf("Starting Thread: %i\n", omp_get_thread_num());

            //end if already sorted above
            if (earlyEnd) {

                //allocate splitters
                uint32_t* ASplitters = (uint32_t*)xcalloc(omp_get_num_threads() + 1, sizeof(uint32_t));
                uint32_t* BSplitters = (uint32_t*)xcalloc(omp_get_num_threads() + 1, sizeof(uint32_t));

                //Calculate indicies
                uint32_t threadNum = omp_get_thread_num();
                uint32_t numberOfSubArrays = omp_get_num_threads();
                uint32_t initialSubArraySize = array_length / omp_get_num_threads();

                //allocate and initilize size vectors (tracks array size)
                uint32_t* arraySizes = (uint32_t*)xcalloc(omp_get_num_threads(), sizeof(uint32_t));



                for (uint32_t thread = 0; thread < (uint32_t)omp_get_num_threads(); thread++) {
                    arraySizes[thread] = initialSubArraySize;
                    if (((thread % 2) == 1) && thread < 2*(array_length % (uint32_t)omp_get_num_threads())) {
                        arraySizes[thread]++;
                    } else if ((array_length % (uint32_t)omp_get_num_threads()) > numberOfSubArrays/2 && thread < 2*((array_length % (uint32_t)omp_get_num_threads()) - numberOfSubArrays/2)) {
                        arraySizes[thread]++;
                    }
                }

                if (threadNum == 4 || 1 == 1) {
                    for (uint32_t i = 0; i < numberOfSubArrays; i++) {
                        printf("%i:arraySizes[%i]:%i\n", omp_get_thread_num(), i, arraySizes[i]);
                    }
                }



                uint32_t threadStartIndex = arraySum(arraySizes, threadNum);
                uint32_t currentSubArraySize = arraySizes[threadNum];

                if (threadNum == 4 || 1 == 1) {
                    printf("%i:threadNum:%i\n", omp_get_thread_num(), threadNum);
                    printf("%i:numberOfSubArrays:%i\n", omp_get_thread_num(), numberOfSubArrays);
                    printf("%i:initialSubArraySize:%i\n", omp_get_thread_num(), initialSubArraySize);
                    printf("%i:threadStartIndex:%i\n", omp_get_thread_num(), threadStartIndex);
                    //printf("%i:subArraySize:%i\n", omp_get_thread_num(), subArraySize);
                }

                //in core sort
                qsort((*array) + threadStartIndex, currentSubArraySize, sizeof(vec_t), hostBasicCompare);

                //variables for defered sub array, this occurs when the number of threads is not 2
                uint32_t deferedSubArray = 0; //acts like a boolean
                uint32_t deferedSize = 0;

                if (numberOfSubArrays % 2 == 1) {
                    deferedSubArray = 1; //acts like a boolean
                    deferedSize = arraySizes[numberOfSubArrays - 1];

                    if (threadNum == 4 || 1 == 1) {
                        printf("%i:DeferingSubArrayAtStart\n", omp_get_thread_num());
                        printf("%i:deferedSize:%i\n", omp_get_thread_num(), deferedSize);
                    }

                    numberOfSubArrays--;
                }

                if (threadNum == 4 || 1 == 1) {
                    printf("%i:NewNumberofSubArrays:%i\n", omp_get_thread_num(), numberOfSubArrays);
                }

                //calulate a couple more variables
                //uint32_t currentSubArraySize = arraySizes[arraySizesIndex] + arraySizes[arraySizesIndex + 1];
                uint32_t numPerMergeThreads = 0;
                uint32_t assignmentNumPerMergeThreads = 0;
                uint32_t leftOverThreads = 0;
                if (numberOfSubArrays/2  > 0) {
                    printf("jkajskldjfkadsjkfkdsajfkdasfkndsjfkdnsjkfndskjfnjadkfnjkdsnfkjandsjknfajkdnsjkfnjkdnjkfnjkdsnjkfnjkdanfjksdnjkfnkjjkasdnjfnasdnjfnakdnfdnsjfnajdsnfnnsajkfnkjdsjkafnadkjfnkjsdnfjjdasnfjksndkfansdkjfkjdbfjbefeuiehurhuehrkeuhrkehrukehukre\n");
                    numPerMergeThreads = omp_get_num_threads()/(numberOfSubArrays/2);
                    assignmentNumPerMergeThreads = omp_get_num_threads()/(numberOfSubArrays/2);
                    leftOverThreads = omp_get_num_threads()%(numberOfSubArrays/2);//calulate how many left over threads there are
                } else {
                    numPerMergeThreads = omp_get_num_threads();
                    assignmentNumPerMergeThreads = omp_get_num_threads();
                    leftOverThreads = 0;
                }
                //now asign left over threads starting at the front
                for (uint32_t i = 0; i < leftOverThreads; i++) {
                    if (threadNum >= (numPerMergeThreads+1)*i && threadNum < (numPerMergeThreads+1)*(i+1)) {
                        numPerMergeThreads++;
                    }
                }

                uint32_t leftOverThreadsCounter = leftOverThreads;
                uint32_t groupNumber = 0;
                for (uint32_t i = 0; i < (uint32_t)omp_get_num_threads() && assignmentNumPerMergeThreads != 0; ) {
                    if (leftOverThreadsCounter) {
                        leftOverThreadsCounter--;
                        i += (assignmentNumPerMergeThreads + 1);
                    } else {
                        i += assignmentNumPerMergeThreads;
                    }
                    if (threadNum < i) {
                        break;
                    }
                    groupNumber++;
                }

                uint32_t arraySizesIndex = groupNumber*2; //points to index of A in array sizes for this thread

                //TODO Update this!!
                uint32_t mergeHeadThreadNum = 0;//the thread number that is the lowest in the this threads merge


                // #pragma omp barrier
                // #pragma omp single
                // {
                //     printf("After Sorting:\n");
                //     for (uint32_t i = 0; i < array_length;i++) {
                //         printf("Array[%i]:%i\n", i, (*array)[i]);
                //     }
                // }



                //begin merging
                #pragma omp barrier
                while (currentSubArraySize < array_length && numberOfSubArrays > 1) {
                    if (threadNum == 21 || 1 == 2) {
                        printf("%i:EnternumPerMergeThreads:%i\n", omp_get_thread_num(), numPerMergeThreads);
                        printf("%i:EnterDefered:%i\n", omp_get_thread_num(), deferedSubArray);

                        printf("%i:arraySizesIndex:%i\n", omp_get_thread_num(), arraySizesIndex);
                    }

                        //Start Point for Sub Arrays
                        uint32_t AStartMergePath = arraySum(arraySizes, arraySizesIndex);
                        uint32_t BStartMergePath = AStartMergePath + arraySizes[arraySizesIndex];

if (threadNum == 21 || 1 == 2) {
                        printf("%i:AStartMergePath:%i\n", omp_get_thread_num(), AStartMergePath);
                        printf("%i:BStartMergePath:%i\n", omp_get_thread_num(), BStartMergePath);
                        printf("%i:ASize:%i\n", omp_get_thread_num(), arraySizes[arraySizesIndex]);
                        printf("%i:BSize:%i\n", omp_get_thread_num(), arraySizes[arraySizesIndex + 1]);

                        printf("%i:Thread Entering Merge\n", omp_get_thread_num());
                    }

                    if (threadNum == 21 || 1 == 2) {
                        for (uint32_t i = 0; i < numberOfSubArrays; i++) {
                            printf("%i:PreMergearraySizes[%i]:%i\n", omp_get_thread_num(), i, arraySizes[i]);
                        }
                    }

                    if (threadNum == 21 || 1 == 2) {
                        printf("%i:Splitters Offset:%i\n", omp_get_thread_num(),numPerMergeThreads*(threadNum / numPerMergeThreads));
                        printf("%i:numPerMergeThreads:%i\n", omp_get_thread_num(), numPerMergeThreads);
                        printf("%i:A_Length:%i\n", omp_get_thread_num(), arraySizes[arraySizesIndex]);
                        printf("%i:B_Length:%i\n", omp_get_thread_num(), arraySizes[arraySizesIndex + 1]);
                    }

                        MergePathSplitter(
                            (*array) + AStartMergePath, arraySizes[arraySizesIndex],
                            (*array) + BStartMergePath, arraySizes[arraySizesIndex + 1],
                            C + AStartMergePath, arraySizes[arraySizesIndex] + arraySizes[arraySizesIndex + 1],
                            numPerMergeThreads,
                            ASplitters + numPerMergeThreads*(threadNum / numPerMergeThreads),
                            BSplitters + numPerMergeThreads*(threadNum / numPerMergeThreads)); //Splitters[subArrayStart thread num] should be index zero

                            if (threadNum == 21 || 1 == 2) {
                                for (uint32_t i = 0; i < numberOfSubArrays; i++) {
                                    printf("%i:AfterMergearraySizes[%i]:%i\n", omp_get_thread_num(), i, arraySizes[i]);
                                }
                            }

if (threadNum == 21 || 1 == 2) {
                        printf("%i:numPerMergeThreads:%i\n", omp_get_thread_num(), numPerMergeThreads);


                        for (uint32_t i = 0; i < numPerMergeThreads + 1; i++) {
                            printf("%i:ASplitters[%i]:%i\n", omp_get_thread_num(), i, (ASplitters + numPerMergeThreads*(threadNum / numPerMergeThreads))[i]);
                        }
                        for (uint32_t i = 0; i < numPerMergeThreads + 1; i++) {
                            printf("%i:BSplitters[%i]:%i\n", omp_get_thread_num(), i, (BSplitters + numPerMergeThreads*(threadNum / numPerMergeThreads))[i]);
                        }

                }

#pragma omp barrier

                // if (threadNum == 4 || 1 == 1) {
                //     printf("%i:SASplitters:%i\n", omp_get_thread_num(), ASplitters[threadNum]);
                //     printf("%i:SASplitters:%i\n", omp_get_thread_num(), ASplitters[threadNum]);
                //     printf("%i:SBSplitters:%i\n", omp_get_thread_num(), BSplitters[threadNum]);
                //     printf("%i:SBSplitters:%i\n", omp_get_thread_num(), BSplitters[threadNum]);
                // }

                    uint32_t A_start = AStartMergePath + ASplitters[threadNum];
                    uint32_t A_end = AStartMergePath + ASplitters[threadNum + 1];
                    uint32_t A_length = A_end - A_start;
                    uint32_t B_start = BStartMergePath + BSplitters[threadNum];
                    uint32_t B_end = BStartMergePath + BSplitters[threadNum + 1];
                    uint32_t B_length = B_end - B_start;
                    //printf("%i:Data: (%i mod %i) ? %i + %i + %i\n", omp_get_thread_num(),threadNum, numPerMergeThreads, ASplitters[threadNum], BSplitters[threadNum], A_start);
                    uint32_t C_start = (threadNum % numPerMergeThreads) ?  ASplitters[threadNum] + BSplitters[threadNum] + AStartMergePath : AStartMergePath; //start C at offset of previous
                    uint32_t C_length = A_length + B_length;

if (threadNum == 21 || 1 == 2) {
                    printf("%i:A_start:%i\n", omp_get_thread_num(), A_start);
                    printf("%i:A_end:%i\n", omp_get_thread_num(), A_end);
                    printf("%i:B_start:%i\n", omp_get_thread_num(), B_start);
                    printf("%i:B_end:%i\n", omp_get_thread_num(), B_end);
                    printf("%i:A_length:%i\n", omp_get_thread_num(), A_length);
                    printf("%i:B_length:%i\n", omp_get_thread_num(), B_length);
                    printf("%i:C_start:%i\n", omp_get_thread_num(), C_start);
                    printf("%i:C_length:%i\n", omp_get_thread_num(), C_length);
                }

                    #pragma omp barrier
                    #pragma omp single
                    {
                        for (uint32_t i = 0; i < array_length; i++) {
                            printf("CB[%i]:%i\n", i, (*array)[i]);
                        }
                    }

                    if (threadNum == 21 || 1 == 2) {
                        for (uint32_t i = 0; i < numberOfSubArrays; i++) {
                            printf("%i:PreMainMergearraySizes[%i]:%i\n", omp_get_thread_num(), i, arraySizes[i]);
                        }
                    }

                    Merge((*array) + A_start, A_length, (*array) + B_start, B_length, C + C_start, C_length);
                        // for (uint32_t i = 0; i < C_length; i++) {
                        //     printf("%i:CAAAA[%i]:%i\n", omp_get_thread_num(),i, (C + C_start)[i]);
                        // }
                        #pragma omp barrier
                        #pragma omp single
                        {
                            for (uint32_t i = 0; i < array_length; i++) {
                                printf("CA[%i]:%i\n", i, C[i]);
                            }
                        }
#pragma omp barrier

                    numberOfSubArrays = numberOfSubArrays/2;

                    if (threadNum == 21 || 1 == 2) {
                        for (uint32_t i = 0; i < numberOfSubArrays*2; i++) {
                            printf("%i:arraySizesBefore[%i]:%i\n", omp_get_thread_num(), i, arraySizes[i]);
                        }
                    }

                    int index = 0;
                    for (uint32_t i = 0; i < numberOfSubArrays; i++) {
                        arraySizes[i] = arraySizes[index] + arraySizes[index + 1];
                        index += 2;
                    }

                    if (threadNum == 21 || 1 == 2) {
                        for (uint32_t i = 0; i < numberOfSubArrays; i++) {
                            printf("%i:arraySizesAfter[%i]:%i\n", omp_get_thread_num(), i, arraySizes[i]);
                        }


                        printf("%i:CurrentDefferedSubArray:%i\n", omp_get_thread_num(), deferedSubArray);
                    }

                    if (numberOfSubArrays % 2 == 1 && deferedSubArray) {
                        #pragma omp single
                        {
                            memcpy((void*)(C+array_length-deferedSize), (void*)((*array)+array_length-deferedSize), deferedSize*sizeof(vec_t));
                            printf("Adding Deffered Sub Array back in\n");
                            printf("%i:Writing to Array Sizes\n", omp_get_thread_num());
                        }
                        deferedSubArray = 0;
                        arraySizes[numberOfSubArrays] = deferedSize;
                        numberOfSubArrays++;
                    } else if (numberOfSubArrays % 2 == 1 && numberOfSubArrays != 1) {
                        deferedSubArray = 1; //acts like a boolean
                        deferedSize = arraySizes[numberOfSubArrays - 1];

                        if (threadNum == 21 || 1 == 2) {
                            printf("%i:DeferingSubArray\n", omp_get_thread_num());
                            printf("%i:deferedSize:%i\n", omp_get_thread_num(), deferedSize);
                        }

                        numberOfSubArrays--;
                    } else if (deferedSubArray) {
                        //Copy sub array to C so it doesn't get lost
                        #pragma omp single
                        {
                            printf("Copying Defered Array\n");
                            memcpy((void*)(C+array_length-deferedSize), (void*)((*array)+array_length-deferedSize), deferedSize*sizeof(vec_t));
                        }
                    }

                    if (threadNum == 21 || 1 == 2) {
                        for (uint32_t i = 0; i < numberOfSubArrays; i++) {
                            printf("%i:arraySizes[%i]:%i\n", omp_get_thread_num(), i, arraySizes[i]);
                        }
                    }

                    currentSubArraySize = arraySizes[0];

                    if (threadNum == 21 || 1 == 2) {
                        printf("%i:NewNumberofSubArrays:%i\n", omp_get_thread_num(), numberOfSubArrays);
                    }
                    //printf("%i:\n", );

                    if (numberOfSubArrays/2 > 0) {
                        numPerMergeThreads = omp_get_num_threads()/(numberOfSubArrays/2);
                        assignmentNumPerMergeThreads = omp_get_num_threads()/(numberOfSubArrays/2);
                        leftOverThreads = omp_get_num_threads()%(numberOfSubArrays/2);//calulate how many left over threads there are
                    } else {
                        numPerMergeThreads = omp_get_num_threads();
                        assignmentNumPerMergeThreads = omp_get_num_threads();
                        leftOverThreads = 0;
                    }

                    //now asign left over threads starting at the front
                    for (uint32_t i = 0; i < leftOverThreads; i++) {
                        if (threadNum >= (numPerMergeThreads+1)*i && threadNum < (numPerMergeThreads+1)*(i+1)) {
                            numPerMergeThreads++;
                        }
                    }

                    leftOverThreadsCounter = leftOverThreads;
                    groupNumber = 0;
                    for (uint32_t i = 0; i < (uint32_t)omp_get_num_threads() && assignmentNumPerMergeThreads != 0; ) {
                        if (leftOverThreadsCounter) {
                            leftOverThreadsCounter--;
                            i += (assignmentNumPerMergeThreads + 1);
                        } else {
                            i += assignmentNumPerMergeThreads;
                        }
                        if (threadNum < i) {
                            break;
                        }
                        groupNumber++;
                    }

                    arraySizesIndex = groupNumber*2; //points to index of A in array sizes for this thread

                    if (threadNum == 21 || 1 == 2) {
                        printf("%i:newArraySizesIndex:%i\n", omp_get_thread_num(), arraySizesIndex);
                        printf("%i:newNumpermegethreads:%i\n", omp_get_thread_num(), numPerMergeThreads);
                    }

                    // arraySizesIndex /= 2;
                    // if (arraySizesIndex >= numberOfSubArrays - 1) {
                    //     arraySizesIndex = 0;
                    // }

                    #pragma omp barrier
                    #pragma omp single
                    {
                        for (uint32_t i = 0; i < array_length; i++) {
                            printf("C[%i]:%i\n", i, C[i]);
                        }

                        printf("%i:Swapping Array\n", omp_get_thread_num());
                        //pointer swap for C
                        vec_t* tmp = *array;
                        *array = C;
                        C = tmp;
                        // printf("%i:Swap Completed!!\n\n\n\n\n", omp_get_thread_num());
                    }
                    //printf("%i:EndNumPerMergeThreads:%i\n", omp_get_thread_num(), numPerMergeThreads);
                    printf("\n");
                    #pragma omp barrier
                    if (threadNum == 21 || 1 == 2) {
                        printf("%i:ExitnumPerMergeThreads:%i\n", omp_get_thread_num(), numPerMergeThreads);
                    }
                    if (threadNum == 21 || 1 == 2) {
                        for (uint32_t i = 0; i < numberOfSubArrays; i++) {
                            printf("%i:ExitarraySizes[%i]:%i\n", omp_get_thread_num(), i, arraySizes[i]);
                        }
                    }
                }
                //free(ASplitters);
                //free(BSplitters);
                //free(arraySizes);
            }
            // printf("\n\n%i:Ending:%i\n", omp_get_thread_num(), omp_get_thread_num());
            // printf("\n\n");
        }
        // printf("Almost Done\n");
        free(C);
        // printf("All Done\n");
}

template <MergeTemplate Merge>
int recursiveMergeSortHelper(vec_t* array, uint32_t array_length, vec_t* C, const uint32_t splitNumber)
{
    if(array_length < splitNumber)
    {
        quickSort(&array,(uint32_t)array_length,0);
        return 0;
    }

    uint32_t subSize = array_length >> 1 ;

    // Recursively sort them
    int first_ret, last_ret;
    first_ret = recursiveMergeSortHelper<Merge>(array, subSize, C, splitNumber);
    last_ret = recursiveMergeSortHelper<Merge>(array + subSize, array_length - subSize, C + subSize, splitNumber);

    vec_t *ip1,*ip2;

    ip1 = (first_ret == 0)? array : C;
    ip2 = (last_ret == 0)? array : C;

    //uint32_t i,i1,i2;
    //i = 0;
    //i1 = 0;
    //i2 = subSize;

    vec_t* op;
    op = (first_ret == 0)? C : array;

    Merge(ip1, (uint32_t)subSize, ip2 + subSize, (uint32_t)(array_length - subSize), op, (uint32_t)array_length);

    return (first_ret + 1)%2;
}

template <MergeTemplate Merge>
void recursiveMergeSort(vec_t** array, uint32_t array_length, const uint32_t splitNumber)
{
    if (splitNumber < 2) {
        printf("Invalid Split Number Given\n");
        exit(1);
    }
    int swap;

    vec_t* C = (vec_t*) malloc(sizeof(vec_t)*array_length);

    swap = recursiveMergeSortHelper<Merge>((*array),array_length,C,splitNumber);

    if(swap == 1)
    {
        vec_t* tmp = *array;
        *array = C;
        C = tmp;
    }

    free(C);
}

/*
 * Template Instantiations
 */

template void iterativeMergeSort<serialMerge>(vec_t** array, uint32_t array_length, const uint32_t splitNumber);
template void iterativeMergeSort<serialMergeNoBranch>(vec_t** array, uint32_t array_length, const uint32_t splitNumber);
template void iterativeMergeSort<bitonicMergeReal>(vec_t** array, uint32_t array_length, const uint32_t splitNumber);
#ifdef AVX512
template void iterativeMergeSort<avx512Merge>(vec_t** array, uint32_t array_length, const uint32_t splitNumber);
#endif
template void recursiveMergeSort<serialMerge>(vec_t** array, uint32_t array_length, const uint32_t splitNumber);
template void recursiveMergeSort<serialMergeNoBranch>(vec_t** array, uint32_t array_length, const uint32_t splitNumber);
template void recursiveMergeSort<bitonicMergeReal>(vec_t** array, uint32_t array_length, const uint32_t splitNumber);
#ifdef AVX512
template void recursiveMergeSort<avx512Merge>(vec_t** array, uint32_t array_length, const uint32_t splitNumber);
#endif
template void parallelIterativeMergeSort<serialMerge>(vec_t** array, uint32_t array_length, const uint32_t splitNumber);
template void parallelIterativeMergeSort<serialMergeNoBranch>(vec_t** array, uint32_t array_length, const uint32_t splitNumber);
template void parallelIterativeMergeSort<bitonicMergeReal>(vec_t** array, uint32_t array_length, const uint32_t splitNumber);
#ifdef AVX512
template void parallelIterativeMergeSort<avx512Merge>(vec_t** array, uint32_t array_length, const uint32_t splitNumber);
#endif
