
GCC = gcc
GCFLAGS += -g -fopenmp -std=c99 -march=native -funroll-loops -ffp-contract=fast -mavx512f -Wall -pedantic

ICC = icc
ICCFLAGS += -fopenmp -std=c99 -axMIC-AVX512 funroll-loops -Wall -pedantic

OS = $(shell uname)

TARGETS_NTHD = main util xmalloc

merge: main.c sorts.c util.c xmalloc.c
	$(GCC) -O2 -o $@ $(GCFLAGS) $^
	
run:
	./merge

clean:
	rm merge -f
