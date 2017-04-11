CC := g++
AVX512 := NO#Valid Options are YES or NO
#GCFLAGS += -fopenmp -std=c99 -march=native -funroll-loops -ffp-contract=fast -Wall -pedantic# -mavx512f

#ICC = icc
#ICCFLAGS += -qopenmp -std=c99 -axMIC-AVX512 -funroll-loops -Wall -pedantic

OS = $(shell uname)

FILES = main.c sorts.c util.c xmalloc.c

#set compiler
ifeq ($(CC),g++)
	CCFLAGS += -fopenmp -march=native -funroll-loops -ffp-contract=fast -Wall -pedantic
else ifeq ($(CC),gcc)
	CCFLAGS += -fopenmp -std=c99 -march=native -funroll-loops -ffp-contract=fast -Wall -pedantic
else ifeq ($(CC),icc)
	CCFLAGS += -qopenmp -std=c99 -funroll-loops -Wall -pedantic
else
    $(error Unknown Compiler Given)
endif

#check for AVX512
ifeq ($(AVX512),NO)

else ifeq ($(AVX512),YES)
	ifeq ($(CC),g++)
		CCFLAGS += -mavx512f -D AVX512
	else ifeq ($(CC),gcc)
		CCFLAGS += -mavx512f -D AVX512
	else ifeq ($(CC),icc)
		CCFLAGS += -axMIC-AVX512 -D AVX512
	else
		$(error Unknown Compiler Given)
	endif
else
    $(error Invalid value for variable AVX512. Valid values are YES or NO)
endif

merge:
	make clean
	$(CC) -O3 -o $@ $(CCFLAGS) $(FILES)

run:
	make merge
	./merge

clean:
	rm merge -f
