#include <sys/time.h>
#include <stdio.h>
#include <inttypes.h>

#include "util.h"

struct timeval tv;

double firsttic = 0;
double lasttic = 0;

void tic_reset() {
	gettimeofday(&tv, NULL);
	firsttic = (double)tv.tv_sec + 1.0e-6 * (double)tv.tv_usec;
	lasttic = firsttic;
}
double tic_total() {
	gettimeofday(&tv, NULL);
	lasttic = (double)tv.tv_sec + 1.0e-6 * (double)tv.tv_usec;
	return lasttic - firsttic;
}
double tic_sincelast() {
	gettimeofday(&tv, NULL);
	double rtnval = ((double)tv.tv_sec + 1.0e-6 * (double)tv.tv_usec) - lasttic;
	lasttic = (double)tv.tv_sec + 1.0e-6 * (double)tv.tv_usec;
	return rtnval;
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
    printf("%s: " BYTE_TO_BINARY_PATTERN " " BYTE_TO_BINARY_PATTERN "\n",text,
		BYTE_TO_BINARY(mask>>8), BYTE_TO_BINARY(mask));
}

int isPowerOfTwo(uint32_t n) {
    if (n == 0) {
        return 0;
	}
    while (n != 1) {
        if (n%2 != 0) {
            return 0;
		}
        n = n/2;
    }
    return 1;
}

void printfcomma(uint64_t n) {
    if (n < 0) {
        printf ("N/A");
        return;
    }
    if (n < 1000) {
        printf ("%lu", n);
        return;
    }
    printfcomma (n/1000);
    printf (",%03lu", n%1000);
}

void clearArray(vec_t* array, uint32_t length) {
    for (uint32_t i = 0; i < length; i++) {
        array[i] = 0;
    }
}
