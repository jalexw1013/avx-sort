
CC = gcc
CFLAGS += -fopenmp --std=c99 -march=native -fverbose-asm -funroll-loops -march=core-avx2 -fverbose-asm -mavx2 -mfma -ffp-contract=fast -mavx512f
#CFLAGS += -I $IPPROOT/include -L $IPPROOT/lib/intel64
#LDFLAGS+= -lippcore -lippvm -lipps

OS = $(shell uname)

TARGETS_NTHD = main util xmalloc



merge: main.c sorts.c util.c xmalloc.c
	$(CC) -O${O} -o $@ $(CFLAGS) $^  $(LDFLAGS)

#	$(CC) -O${O} -g -Wa,-a,-ad  $@ $(CFLAGS) $^  $(LDFLAGS) > listing.lst

#	g++ -msse3 -O${O} *.c -o merge @ $(LDFLAGS)
#	g++ -msse3 -O${O} *.c -o  merge

#icc:
#	icc -fasm-blocks -std=c99 -O${O} *.c -g -o merge

debug: main.c sorts.c util.c xmalloc.c
	$(CC) -O${O} -o $@ -g -O1 -mavx2 -mavx512f $^  $(LDFLAGS)

sde64: main.c sorts.c util.c xmalloc.c
	sde/sde64 -- $(CC) -O${O} -o $@ $(CFLAGS) $^  $(LDFLAGS)

clean:
	rm merge -f
	rm debug -f
	rm sde64 -f
