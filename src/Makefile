# Authors: Alex Watkins - https://github.com/jalexw1013 - http://alexwatkins.co
#          Oded Green - https://github.com/ogreen
#          Other minor authors are noted next to their contributions in the code
#
#  Copyright (c) 2018 Alex Watkins, All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions
#  are met:
#
#  1. Redistributions of source code must retain the above copyright
#  notice, this list of conditions and the following disclaimer.
#
#  2. Redistributions in binary form must reproduce the above copyright
#  notice, this list of conditions and the following disclaimer in the
#  documentation and/or other materials provided with the distribution.
#
#  3. Neither the name of the copyright holder nor the names of its
#  contributors may be used to endorse or promote products derived
#  from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
#  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
#  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
#  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
#  COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
#  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
#  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
#  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
#  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
#  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
#  POSSIBILITY OF SUCH DAMAGE.
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
