#include "safemalloc.h"

void * safemalloc(size_t n)
{
    void * ptr = NULL;
    int rc = posix_memalign( &ptr, ALIGNMENT, n);
    if ( ptr==NULL || rc!=0 ) {
        fprintf( stderr , "%ld bytes could not be allocated (rc=%d) \n" , (long)n, rc);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    return ptr;
}

void * typemalloc(MPI_Datatype dt, int n)
{
    int typesize = 0;
    MPI_Type_size(dt, &typesize);
    void * ptr = safemalloc(n*typesize);
    return ptr;
}

FILE * safefopen(const char * path, const char *mode)
{
    FILE * fp = fopen(path, mode);
    if ( fp==NULL ) {
        fprintf( stderr , "file at %s could not be opened \n" , path);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    return fp;
}
