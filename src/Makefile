CC := g++
AVX512 := NO#Valid Options are YES or NO
MERGE := NO#Valid Options are YES or NO
SORT  := YES#Valid Options are YES or NO

OS = $(shell uname)

FILES = main.c sorts.c utils/util.c utils/xmalloc.c

#set compiler
ifeq ($(CC),g++)
	CCFLAGS += -std=c++11 -fopenmp -march=native -funroll-loops -ffp-contract=fast -Wall -pedantic
else ifeq ($(CC),gcc)
	CCFLAGS += -fopenmp -std=c99 -march=native -funroll-loops -ffp-contract=fast -Wall -pedantic
else ifeq ($(CC),icc)
	CCFLAGS += -qopenmp -std=c99 -funroll-loops -Wall -pedantic
else
    $(error Unknown Compiler Given)
endif

#check Whether merging and sorting
ifeq ($(MERGE),NO)

else ifeq ($(MERGE),YES)
	CCFLAGS += -D MERGE
else
    $(error Invalid value for variable Merge. Valid values are YES or NO)
endif
ifeq ($(SORT),NO)

else ifeq ($(SORT),YES)
	CCFLAGS += -D SORT
else
    $(error Invalid value for variable SORT. Valid values are YES or NO)
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

.PHONY: merge
merge:
	rm merge -f
	$(CC) -O3 -o $@ $(CCFLAGS) $(FILES)

run:
	make merge
	./merge

clean:
	rm merge -f
