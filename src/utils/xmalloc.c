/* -*- mode: C; fill-column: 70; -*- */
#include <stdlib.h>
#include <stdio.h>
#include <errno.h>
// #include <malloc.h>
#include "xmalloc.h"

void *
xmalloc (size_t sz)
{
    void * out;
    // out = memalign (16, sz);
    printf("5:%i\n", sz);
    out = malloc (sz);
    printf("6\n");
    if (!out) {
	perror ("Failed xmalloc: ");
	abort ();
    }
    printf("7\n");
    return out;
}
void *
xcalloc (size_t nelem, size_t sz)
{
    void * out;
    out = calloc (nelem, sz);
    if (!out) {
	perror ("Failed xcalloc: ");
	abort ();
    }
    return out;
}
void *
xrealloc (void *p, size_t sz)
{
    void * out;
    out = realloc (p, sz);
    if (!out) {
	perror ("Failed xrealloc: ");
	abort ();
    }
    return out;
}
