#ifndef SAFEMALLOC_H
#define SAFEMALLOC_H

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <assert.h>
#include <mpi.h>

int posix_memalign(void **memptr, size_t alignment, size_t size);

#define ALIGNMENT 128

void * safemalloc(size_t n);
void * typemalloc(MPI_Datatype dt, int n);

FILE * safefopen(const char *path, const char *mode);

#endif // SAFEMALLOC_H
