/*
  Authors: Alex Watkins - https://github.com/jalexw1013 - http://alexwatkins.co
           Oded Green - https://github.com/ogreen
           Other minor authors are noted next to their contributions in the code

         Copyright (c) 2018 Alex Watkins, All rights reserved.

         Redistribution and use in source and binary forms, with or without
         modification, are permitted provided that the following conditions
         are met:

         1. Redistributions of source code must retain the above copyright
         notice, this list of conditions and the following disclaimer.

         2. Redistributions in binary form must reproduce the above copyright
         notice, this list of conditions and the following disclaimer in the
         documentation and/or other materials provided with the distribution.

         3. Neither the name of the copyright holder nor the names of its
         contributors may be used to endorse or promote products derived
         from this software without specific prior written permission.

         THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
         "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
         LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
         FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
         COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
         INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
         BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
         LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
         CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
         LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
         ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
         POSSIBILITY OF SUCH DAMAGE.
*/
#ifndef HEADER_FILE_MAIN
#define HEADER_FILE_MAIN

#define min(a,b) (a <= b)? a : b
#define max(a,b) (a <  b)? b : a

typedef uint32_t vec_t;

enum AlgoType{Merge, ParallelMerge, Sort, ParallelSort};

struct AlgoArgs {
    vec_t* A;
    uint32_t A_length;
    vec_t* B;
    uint32_t B_length;
    vec_t* C;
    uint32_t C_length;
    vec_t* CUnsorted;
    uint32_t threadNum;
    uint32_t numThreads;
    uint32_t* ASplitters;
    uint32_t* BSplitters;
    uint32_t* arraySizes;
};

void MergePathSplitter(
    vec_t * A, uint32_t A_length,
    vec_t * B, uint32_t B_length,
    vec_t * C, uint32_t C_length,
    uint32_t threads, uint32_t* ASplitters, uint32_t* BSplitters);

int hostBasicCompare(const void * a, const void * b);

#endif
