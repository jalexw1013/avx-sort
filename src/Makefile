
GCC = gcc-6
GCFLAGS += -g -fopenmp -std=c99 -march=native -funroll-loops -ffp-contract=fast -Wall -pedantic

ICC = icc
ICCFLAGS += -openmp -std=c99 -axMIC-AVX512 funroll-loops -Wall -pedantic

OS = $(shell uname)

TARGETS_NTHD = main util xmalloc

merge: main.c sorts.c util.c xmalloc.c
	$(ICC) -O3 -o $@ $(GCFLAGS) $^

run:
	make merge
	./merge

clean:
	rm merge -f
