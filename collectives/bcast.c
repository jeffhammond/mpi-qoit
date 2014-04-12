#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>
#include <assert.h>
#include <math.h>
#include <mpi.h>

#include "safemalloc.h"

void bcast_only(FILE * output, MPI_Comm comm, int max_mem)
{
    int comm_rank = -1, world_rank = -1, comm_size = 0;
    MPI_Comm_rank(comm, &comm_rank);
    MPI_Comm_size(comm, &comm_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    if ( comm_rank == 0 )
        fprintf(output, "============== BCAST ==============\n");

    /* testing only integers because it really ~shouldn't~ matter */

    int max_count = max_mem/sizeof(int);

    for (int c=1; c<max_count; c*=2)
    {
        int root = 0;

        int value = 0;
        int * buffer = (int *) safemalloc(c*sizeof(int));

        if ( comm_rank == root )
            value = comm_size;
        else
            value = 0;

        for (int i=0 ; i< c; i++)
            buffer[i] = value;

#ifdef PRE_BARRIER_HACK
        MPI_Barrier(comm);
#endif

        double t0 = MPI_Wtime();
        MPI_Bcast( buffer, c, MPI_INT, root, comm );
        double t1 = MPI_Wtime();
        double dt_bcast = t1-t0;

        int errors = 0;
        for (int i=0 ; i< c; i++)
            errors += (int) ( buffer[i] != comm_size );

        if (errors>0)
        {
            fprintf(output, "%d: MPI_Bcast had %d errors on rank %d! \n",
                   world_rank, errors, comm_rank);
            for (int i=0 ; i<c; i++)
                fprintf(output, "%d: buffer[%d] = %d (correct is %d) \n",
                       world_rank, i, buffer[i], comm_size );
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        if ( comm_rank == root )
            fprintf(output, "%d: MPI_Bcast %d integers in %lf seconds (%lf MB/s) \n",
                   world_rank, c, dt_bcast, 1.0e-6*c*sizeof(int)/dt_bcast );

        free(buffer);
    }

    return;
}

void bcast_vs_scatter_allgather(FILE * output, MPI_Comm comm, int max_mem)
{
    int comm_rank = -1, world_rank = -1, comm_size = 0;
    MPI_Comm_rank(comm, &comm_rank);
    MPI_Comm_size(comm, &comm_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    if ( comm_rank == 0 )
        fprintf(output, "============== BCAST VS SCATTER+ALLGATHER ==============\n");


    /* testing only intergers because it really ~shouldn't~ matter */

    int max_count = max_mem/sizeof(int);

    for (int c=1; c<max_count; c*=2)
        if (1) //(c%comm_size==0)
        {
            int root = 0;

            int value;
            if ( comm_rank == root )
                value = comm_size;
            else
                value = 0;

            double dt_bcast, dt_scalg, t0, t1;
            int errors;

            int * buffer = (int *) safemalloc(c*sizeof(int));

            /*****************************************************/

            for (int i=0 ; i<c; i++)
                buffer[i] = value;

#ifdef PRE_BARRIER_HACK
            MPI_Barrier(comm);
#endif

            t0 = MPI_Wtime();
            MPI_Bcast( buffer, c, MPI_INT, root, comm );
            t1 = MPI_Wtime();
            dt_bcast = t1-t0;

            errors = 0;
            for (int i=0 ; i<c; i++)
                errors += (int) ( buffer[i] != comm_size );

            if (errors>0)
            {
                fprintf(output, "%d: MPI_Bcast had %d errors on rank %d! \n",
                        world_rank, errors, comm_rank);
                for (int i=0 ; i<c; i++)
                    fprintf(output, "%d: buffer[%d] = %d (correct is %d) \n",
                            world_rank, i, buffer[i], comm_size );
                MPI_Abort(MPI_COMM_WORLD, 1);
            }

            /*****************************************************/

            int * temp = (int *) safemalloc((c/comm_size)*sizeof(int));

            for (int i=0 ; i<c; i++)
                buffer[i] = value;

            for (int i=0 ; i<(c/comm_size); i++)
                temp[i] = 0;

#ifdef PRE_BARRIER_HACK
        MPI_Barrier(comm);
#endif

            t0 = MPI_Wtime();
            MPI_Scatter( buffer, c/comm_size, MPI_INT, temp, c/comm_size, MPI_INT, root, comm );
            MPI_Allgather( temp, c/comm_size, MPI_INT, buffer, c/comm_size, MPI_INT, comm );
            t1 = MPI_Wtime();
            dt_scalg = t1-t0;

            errors = 0;
            for (int i=0 ; i<c; i++)
                errors += (int) ( buffer[i] != comm_size );

            if (errors>0)
            {
                fprintf(output, "%d: MPI_Scatter+MPI_Allgather had %d errors on rank %d! \n",
                       world_rank, errors, comm_rank);
                for (int i=0 ; i<c; i++)
                    fprintf(output, "%d: buffer[%d] = %d (correct is %d) \n",
                           world_rank, i, buffer[i], comm_size );
                MPI_Abort(MPI_COMM_WORLD, 1);
            }

            if ( comm_rank == root )
                fprintf(output, "%d: MPI_Bcast vs MPI_Scatter+MPI_Allgather %d integers in %lf vs %lf seconds (%lf vs %lf MB/s) \n",
                       world_rank, c, dt_bcast, dt_scalg, 1.0e-6*c*sizeof(int)/dt_bcast, 1.0e-6*c*sizeof(int)/dt_scalg );

            free(temp);

            /*****************************************************/

            free(buffer);
        }

    return;
}
