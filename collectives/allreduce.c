#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>
#include <assert.h>
#include <math.h>
#include <mpi.h>

#include "safemalloc.h"

void allreduce_only(FILE * output, MPI_Comm comm, int max_mem)
{
	int comm_rank = -1, world_rank = -1, comm_size = 0;
	MPI_Comm_rank(comm, &comm_rank);
	MPI_Comm_size(comm, &comm_size);
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    if ( comm_rank == 0 )
        fprintf(output, "============== ALLREDUCE ==============\n");

    int root = 0;

    int max_count = max_mem/sizeof(int);

    for (int c=1; c<max_count; c*=2)
    {
        int * in  = (int *) safemalloc(c*sizeof(int));
        int * out = (int *) safemalloc(c*sizeof(int));

        for (int i=0 ; i<c; i++)
            in [i] = 1;

        for (int i=0 ; i<c; i++)
            out[i] = 0;

#ifdef PRE_BARRIER_HACK
        MPI_Barrier(comm);
#endif

        double t0 = MPI_Wtime();
        MPI_Allreduce( in , out, c, MPI_INT, MPI_SUM, comm );
        double t1 = MPI_Wtime();
        double dt_allred = t1-t0;

        int errors = 0;
        for (int i=0 ; i<c; i++)
            errors += (int) ( out[i] != comm_size );

        if (errors>0)
        {
            fprintf(output, "MPI_Allreduce had %d errors! \n", errors);
			for (int i=0 ; i<(c*comm_size); i++)
				fprintf(output, "%d: out[%d] = %d (correct is %d) \n",
						world_rank, i, out[i], i/c );
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        if ( comm_rank == root )
            fprintf(output, "MPI_Allreduce %d integers in %lf seconds (%lf MB/s) \n",
                    c, dt_allred, 1.0e-6*c*sizeof(int)/dt_allred );

        free(out);
        free(in );
    }

	max_count = max_mem/sizeof(double);

	for (int c=1; c<max_count; c*=2)
	{
		double * in  = (double *) safemalloc(c*sizeof(double));
		double * out = (double *) safemalloc(c*sizeof(double));

		for (int i=0 ; i<c; i++)
			in [i] = (double)1;

		for (int i=0 ; i<c; i++)
			out[i] = (double)0;

#ifdef PRE_BARRIER_HACK
        MPI_Barrier(comm);
#endif

		double t0 = MPI_Wtime();
		MPI_Allreduce( in , out, c, MPI_DOUBLE, MPI_SUM, comm );
		double t1 = MPI_Wtime();
		double dt_allred = t1-t0;

		int errors = 0;
			for (int i=0 ; i<c; i++)
				errors += (int) ( out[i] != (double)comm_size );

		if (errors>0)
		{
			fprintf(output, "MPI_Reduce had %d errors! \n", errors);
			for (int i=0 ; i<(c*comm_size); i++)
				fprintf(output, "%d: out[%d] = %lf (correct is %lf) \n",
						world_rank, i, out[i], (double)i/c );
            MPI_Abort(MPI_COMM_WORLD, 1);
		}

		if ( comm_rank == root )
			fprintf(output, "MPI_Allreduce %d doubles in %lf seconds (%lf MB/s) \n",
					c, dt_allred, 1.0e-6*c*sizeof(double)/dt_allred );

		free(out);
		free(in );
	}

    return;
}
