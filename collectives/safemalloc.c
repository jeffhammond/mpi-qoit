#include "safemalloc.h"

void * safemalloc(size_t n)
{
    //void * ptr = malloc( n );

    int rc = -1;
    void * ptr = NULL;
    rc = posix_memalign( &ptr, ALIGNMENT, n);

    if ( ptr==NULL || rc!=0 )
    {
        fprintf( stderr , "%ld bytes could not be allocated \n" , (long)n );
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    return ptr;
}

void * typemalloc(MPI_Datatype dt, int n)
{
    int typesize = 0;
    MPI_Type_size(dt, &typesize);

    void * ptr = NULL;
    ptr = safemalloc(n*typesize);

    return ptr;
}

FILE * safefopen(const char * path, const char *mode)
{
    //void * ptr = malloc( n );

    FILE * fp = fopen(path, mode);

    if ( fp==NULL )
    {
        fprintf( stderr , "file at %s could not be opened \n" , path );
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    return fp;
}
