#if !defined(UTIL_H_)
#define UTIL_H_

//Must be included for __m512i variable
#include <x86intrin.h>
#include <immintrin.h>

void tic_reset();
double tic_total();
double tic_sincelast();

#ifdef AVX512
void print512_num(char *text, __m512i var);
void print16intarray(char *text, int *val);
void printmmask16(char *text, __mmask16 mask);
#endif

#endif
