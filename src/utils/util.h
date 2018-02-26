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
#if !defined(UTIL_H_)
#define UTIL_H_

//Must be included for __m512i variable
#include <x86intrin.h>
#include <immintrin.h>
#include "../main.h"

#define ANSI_COLOR_RED     "\x1b[31m"
#define ANSI_COLOR_GREEN   "\x1b[32m"
#define ANSI_COLOR_YELLOW  "\x1b[33m"
#define ANSI_COLOR_BLUE    "\x1b[34m"
#define ANSI_COLOR_MAGENTA "\x1b[35m"
#define ANSI_COLOR_CYAN    "\x1b[36m"
#define ANSI_COLOR_RESET   "\x1b[0m"

#define MAX(a,b) \
   ({ __typeof__ (a) _a = (a); \
       __typeof__ (b) _b = (b); \
     _a > _b ? _a : _b; })

void tic_reset();
double tic_total();
double tic_sincelast();

void print512_num(char *text, __m512i var);
void print16intarray(char *text, int *val);
void printmmask16(char *text, __mmask16 mask);

int isPowerOfTwo(uint32_t n);
void printfcomma(uint64_t n);
void clearArray(vec_t* array, uint32_t length);

#endif
