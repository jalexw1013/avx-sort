
GCC = gcc
GCFLAGS += -fopenmp -std=c99 -march=native -funroll-loops -march=core-AVX512 -msse -msse2 -sse4.1 -sse4.2 -avx -msse3 -mavx2 -mfma -mfpmath=sse -ffp-contract=fast -mavx512f

ICC = icc
ICCFLAGS += -fopenmp -std=c99 -axMIC-AVX512 funroll-loops -Wall -pedantic

OS = $(shell uname)

TARGETS_NTHD = main util xmalloc



merge: main.c sorts.c util.c xmalloc.c
	$(CC) -O2 -o $@ $(CFLAGS) $^  $(LDFLAGS)

debug: main.c sorts.c util.c xmalloc.c
	$(CC) -O0 -o $@ -g -O1 -mavx2 -mavx512f $^  $(LDFLAGS)

sde64: main.c sorts.c util.c xmalloc.c
	sde/sde64 -- $(CC) -O2 -o $@ $(CFLAGS) $^  $(LDFLAGS)

clean:
	rm merge -f
	rm debug -f
	rm sde64 -f
