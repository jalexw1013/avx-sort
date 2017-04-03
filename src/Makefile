
GCC = gcc
GCFLAGS += -g -fopenmp -std=c99 -march=native -funroll-loops -ffp-contract=fast -Wall -pedantic -mavx512f

ICC = icc
ICCFLAGS += -qopenmp -std=c99 -axMIC-AVX512 -funroll-loops -Wall -pedantic

OS = $(shell uname)

TARGETS_NTHD = main util xmalloc

merge: main.c sorts.c util.c xmalloc.c
	$(ICC) -O3 -o $@ $(ICCFLAGS) $^

run:
	make merge
	./merge

clean:
	rm merge -f
