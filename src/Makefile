CC := icpc
VERIFYOUTPUT := NO#Valid Options are YES or NO

OS = $(shell uname)

FILES = main.c sorts.c utils/util.c utils/xmalloc.c

#set compiler
ICPCAVAI := $(shell command -v icpc 2> /dev/null)
ifeq ($(CC),icpc)
	ifndef ICPCAVAI
    	$(warning "icpc is not available, g++ will be used instead")
		CC = g++
	endif
endif

#verify g++ version
ifeq ($(AVX512),YES)
	G++VERSIONGTQ4 := $(shell expr `g++ -dumpversion | cut -f1 -d.` \>= 4.9)
	ifneq "$(G++VERSIONGTQ4)" "1"
    	$(error "g++ must be version 4.9 or greater to run AVX-512")
	endif
endif

#set compiler flags
ifeq ($(CC),g++)
	CCFLAGS += -std=c++11 -fopenmp -march=native -funroll-loops -ffp-contract=fast -Wall -pedantic -fkeep-inline-functions
else ifeq ($(CC),gcc)
	CCFLAGS += -fopenmp -std=c99 -march=native -funroll-loops -ffp-contract=fast -Wall -pedantic -fkeep-inline-functions
else ifeq ($(CC),icc)
	CCFLAGS += -qopenmp -std=c99 -funroll-loops -Wall -pedantic -fkeep-inline-functions
else ifeq ($(CC),icpc)
	CCFLAGS += -qopenmp -std=c++11 -funroll-loops -Wall -pedantic -fkeep-inline-functions
else
    $(error Unknown Compiler Given)
endif

ifeq ($(VERIFYOUTPUT),NO)

else ifeq ($(VERIFYOUTPUT),YES)
	CCFLAGS += -D VERIFYOUTPUT
else
    $(error Invalid value for variable VERIFYOUTPUT. Valid values are YES or NO)
endif

ifeq ($(CC),g++)
	CCFLAGS += -mavx512f -D AVX512
else ifeq ($(CC),gcc)
	CCFLAGS += -mavx512f -D AVX512
else ifeq ($(CC),icc)
	CCFLAGS += -axMIC-AVX512 -D AVX512
else ifeq ($(CC),icpc)
	CCFLAGS += -axMIC-AVX512 -D AVX512 -L $IPPROOT/lib/intel64_lin/threaded -lippi -lipps -lippcore -lpthread
	CCFLAGS += -I $TBB_INCLUDE -Wl,-rpath,$TBB_LIBRARY_RELEASE -L $TBB_LIBRARY_RELEASE -ltbb
	#CCFLAGS += -ltbb
else
	$(error Unknown Compiler Given)
endif

.PHONY: merge
merge:
	rm merge -f
	$(CC) -O3 -o $@ $(CCFLAGS) $(FILES)

run:
	make merge
	numactl --membind=1 ./merge

clean:
	rm merge -f
