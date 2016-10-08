
CC = sde/sde64 -- gcc
CFLAGS += -fopenmp --std=c99 -march=native -fverbose-asm -funroll-loops -march=core-avx2 -fverbose-asm -mavx2 -mfma -ffp-contract=fast -mavx512f
#CFLAGS += -I $IPPROOT/include -L $IPPROOT/lib/intel64
#LDFLAGS+= -lippcore -lippvm -lipps

OS = $(shell uname)

TARGETS_NTHD = main util xmalloc



merge: main.c util.c xmalloc.c
	$(CC) -O${O} -o $@ $(CFLAGS) $^  $(LDFLAGS)

#	$(CC) -O${O} -g -Wa,-a,-ad  $@ $(CFLAGS) $^  $(LDFLAGS) > listing.lst

#	g++ -msse3 -O${O} *.c -o merge @ $(LDFLAGS)
#	g++ -msse3 -O${O} *.c -o  merge

#icc:
#	icc -fasm-blocks -std=c99 -O${O} *.c -g -o merge

clean:
	rm merge
