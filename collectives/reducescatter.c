#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>
#include <assert.h>
#include <math.h>
#include <mpi.h>

#include "safemalloc.h"

void reducescatter_only(FILE * output, MPI_Comm comm, int max_mem)
{
#if 0
	int comm_rank = -1, world_rank = -1, comm_size = 0;
	MPI_Comm_rank(comm, &comm_rank);
	MPI_Comm_size(comm, &comm_size);
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    if ( comm_rank == 0 )
        fprintf(output, "============== REDUCESCATTERBLOCK ==============\n");

    int root = 0;

    int max_count = max_mem/comm_size/sizeof(int);

    for (int c=1; c<max_count; c*=2)
    {

        int * in  = (int *) safemalloc(c*comm_size*sizeof(int));
        int * out = (int *) safemalloc(c*sizeof(int));

        for (int i=0 ; i<c*comm_size; i++)
            in [i] = 0;

        for (int i=0 ; i<c; i++)
            in [comm_rank*c+i] = comm_rank;

        for (int i=0 ; i<c; i++)
            out[i] = 0;

#ifdef PRE_BARRIER_HACK
        MPI_Barrier(comm);
#endif

        double t0 = MPI_Wtime();
        MPI_Reduce_scatter_block( in , out, c, MPI_INT, MPI_SUM, comm );
        double t1 = MPI_Wtime();
        double dt_redscat = t1-t0;

        int errors = 0;
        for (int i=0 ; i<c; i++)
            errors += (int) ( out[i] != comm_rank );

        if (errors>0)
        {
            fprintf(output, "MPI_Reduce_scatter_block had %d errors! \n", errors);
			for (int i=0 ; i<(c*comm_size); i++)
				fprintf(output, "%d: out[%d] = %d (correct is %d) \n",
						world_rank, i, out[i], comm_rank );
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        if ( comm_rank == root )
            fprintf(output, "MPI_Reduce_scatter_block %d integers in %lf seconds (%lf MB/s) \n",
                   c*comm_size, dt_redscat, 1.0e-6*c*comm_size*sizeof(int)/dt_redscat );

        free(out);
        free(in );
    }

    max_count = max_mem/comm_size/sizeof(double);

    for (int c=1; c<max_count; c*=2)
    {

        double * in  = (double *) safemalloc(c*comm_size*sizeof(double));
        double * out = (double *) safemalloc(c*sizeof(double));

        for (int i=0 ; i<c*comm_size; i++)
            in[i] = (double)0;

        for (int i=0 ; i<c; i++)
            in [comm_rank*c+i] = (double)comm_rank;

        for (int i=0 ; i<c; i++)
            out[i] = (double)0;

#ifdef PRE_BARRIER_HACK
        MPI_Barrier(comm);
#endif

        double t0 = MPI_Wtime();
        MPI_Reduce_scatter( in , out, c, MPI_DOUBLE, MPI_SUM, comm );
        double t1 = MPI_Wtime();
        double dt_redscat = t1-t0;

        int errors = 0;
        for (int i=0 ; i<c; i++)
            errors += (int) ( out[i] != (double)comm_rank );

        if (errors>0)
        {
            fprintf(output, "MPI_Reduce_scatter_block had %d errors! \n", errors);
			for (int i=0 ; i<(c*comm_size); i++)
				fprintf(output, "%d: out[%d] = %lf (correct is %lf) \n",
						world_rank, i, out[i], (double)comm_rank );
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        if ( comm_rank == root )
            fprintf(output, "MPI_Reduce_scatter_block %d doubles in %lf seconds (%lf MB/s) \n",
                   c*comm_size, dt_redscat, 1.0e-6*c*comm_size*sizeof(int)/dt_redscat );

        free(out);
        free(in );
    }
#endif

    return;
}

void reducescatterblock_only(FILE * output, MPI_Comm comm, int max_mem)
{
#if ( MPI_VERSION == 2 && MPI_SUBVERSION == 2 ) || MPI_VERSION >= 3

	int comm_rank = -1, world_rank = -1, comm_size = 0;
	MPI_Comm_rank(comm, &comm_rank);
	MPI_Comm_size(comm, &comm_size);
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    if ( comm_rank == 0 )
        fprintf(output, "============== REDUCESCATTERBLOCK ==============\n");

    int root = 0;

    int max_count = max_mem/comm_size/sizeof(int);

    for (int c=1; c<max_count; c*=2)
    {

        int * in  = (int *) safemalloc(c*comm_size*sizeof(int));
        int * out = (int *) safemalloc(c*sizeof(int));

        for (int i=0 ; i<c*comm_size; i++)
            in [i] = 0;

        for (int i=0 ; i<c; i++)
            in [comm_rank*c+i] = comm_rank;

        for (int i=0 ; i<c; i++)
            out[i] = 0;

#ifdef PRE_BARRIER_HACK
        MPI_Barrier(comm);
#endif

        double t0 = MPI_Wtime();
        MPI_Reduce_scatter_block( in , out, c, MPI_INT, MPI_SUM, comm );
        double t1 = MPI_Wtime();
        double dt_redscat = t1-t0;

        int errors = 0;
        for (int i=0 ; i<c; i++)
            errors += (int) ( out[i] != comm_rank );

        if (errors>0)
        {
            fprintf(output, "MPI_Reduce_scatter_block had %d errors! \n", errors);
			for (int i=0 ; i<(c*comm_size); i++)
				fprintf(output, "%d: out[%d] = %d (correct is %d) \n",
						world_rank, i, out[i], comm_rank );
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        if ( comm_rank == root )
            fprintf(output, "MPI_Reduce_scatter_block %d integers in %lf seconds (%lf MB/s) \n",
                   c*comm_size, dt_redscat, 1.0e-6*c*comm_size*sizeof(int)/dt_redscat );

        free(out);
        free(in );
    }

    max_count = max_mem/comm_size/sizeof(double);

    for (int c=1; c<max_count; c*=2)
    {

        double * in  = (double *) safemalloc(c*comm_size*sizeof(double));
        double * out = (double *) safemalloc(c*sizeof(double));

        for (int i=0 ; i<c*comm_size; i++)
            in[i] = (double)0;

        for (int i=0 ; i<c; i++)
            in [comm_rank*c+i] = (double)comm_rank;

        for (int i=0 ; i<c; i++)
            out[i] = (double)0;

#ifdef PRE_BARRIER_HACK
        MPI_Barrier(comm);
#endif

        double t0 = MPI_Wtime();
        MPI_Reduce_scatter_block( in , out, c, MPI_DOUBLE, MPI_SUM, comm );
        double t1 = MPI_Wtime();
        double dt_redscat = t1-t0;

        int errors = 0;
        for (int i=0 ; i<c; i++)
            errors += (int) ( out[i] != (double)comm_rank );

        if (errors>0)
        {
            fprintf(output, "MPI_Reduce_scatter_block had %d errors! \n", errors);
			for (int i=0 ; i<(c*comm_size); i++)
				fprintf(output, "%d: out[%d] = %lf (correct is %lf) \n",
						world_rank, i, out[i], (double)comm_rank );
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        if ( comm_rank == root )
            fprintf(output, "MPI_Reduce_scatter_block %d doubles in %lf seconds (%lf MB/s) \n",
                   c*comm_size, dt_redscat, 1.0e-6*c*comm_size*sizeof(int)/dt_redscat );

        free(out);
        free(in );
    }
#endif

    return;
}
