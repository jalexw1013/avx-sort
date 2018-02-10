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
