#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>
#include <assert.h>
#include <math.h>
#include <mpi.h>

#if MPI_VERSION == 2
#  warning PLEASE UPGRADE YOUR MPI IMPLEMENTATION TO MPI-3!!!
#elif MPI_VERSION < 2
#  error SORRY, BUT YOUR MPI IMPLEMENTATION DOES NOT SUPPORT RMA
#endif

void static_win_rma2(FILE * output, MPI_Comm comm, int max_mem)
{
    int comm_rank = -1, world_rank = -1, comm_size = 0;
    MPI_Comm_rank(comm, &comm_rank);
    MPI_Comm_size(comm, &comm_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    int maxcount = max_mem/sizeof(double);

    if ( comm_rank == 0 )
        fprintf(output, "============== MPI-2 RMA ==============\n");

    double * localbuffer  = NULL;
    double * bigwinlocptr = NULL;
    MPI_Win bigwin;

    MPI_Alloc_mem(maxcount*sizeof(double), MPI_INFO_NULL, &localbuffer);
    MPI_Alloc_mem(maxcount*sizeof(double), MPI_INFO_NULL, &bigwinlocptr);
    MPI_Win_create(bigwinlocptr, maxcount*sizeof(double), sizeof(double), MPI_INFO_NULL, comm, &bigwin);
    
    int target = (comm_rank+1)%comm_size;
#if DEBUG>0
    fprintf(output, "doing RMA from rank %d to rank %d \n", comm_rank, target);
    fflush(output);
#endif
    MPI_Barrier(comm);

    if (comm_rank==0) {
        fprintf(output, "%8s %29s %29s %29s \n", "",
                        "Put             ", "Get             ", "Acc (SUM)       ");
        fprintf(output, "%8s %29s %29s %29s \n", "doubles",
                        "time (us) bandwidth (MB/s)", "time (us) bandwidth (MB/s)", "time (us) bandwidth (MB/s)");
        fprintf(output, "------------------------------------------------------------------------------------------------------\n");
        fflush(output);
    }

    for (int count=1; count<maxcount; count*=2)
    {
        double t0, t1, dt[3], avg[3];
        int iter = maxcount/count;
        int errors = 0;

#if DEBUG>0
        fprintf(output, "%d: maxcount = %d, count = %d, iter = %d \n", comm_rank, maxcount, count, iter);
#endif

        /********************************/

        /* zero local buffer and window */
        for (int i=0; i<maxcount; i++)
            localbuffer[i] = 0.0;

        MPI_Win_fence(MPI_MODE_NOSTORE, bigwin);
        MPI_Put(localbuffer, maxcount, MPI_DOUBLE, comm_rank, 0 /* disp */, maxcount, MPI_DOUBLE, bigwin);
        MPI_Win_fence(MPI_MODE_NOSTORE, bigwin);

        /********************************/

        for (int i=0; i<maxcount; i++)
            localbuffer[i] = 1000.0+target;

        t0 = MPI_Wtime();
        MPI_Win_lock(MPI_LOCK_SHARED, target, MPI_MODE_NOCHECK, bigwin);
        for (int i=0; i<iter; i++) {
#if DEBUG>1
            fprintf(output, "%d: MPI_Put count=%d, disp=%d \n", comm_rank, count, i*count);
#endif
            MPI_Put(&(localbuffer[i*count]), count, MPI_DOUBLE, target, i*count /* disp */, count, MPI_DOUBLE, bigwin);
        }
        MPI_Win_unlock(target, bigwin);
        t1 = MPI_Wtime();
        dt[0] = t1-t0;

        /********************************/

#if 0
        MPI_Win_lock(MPI_LOCK_SHARED, target, MPI_MODE_NOCHECK, bigwin);
        MPI_Get(localbuffer, maxcount, MPI_DOUBLE, target, 0 /* disp */, maxcount, MPI_DOUBLE, bigwin);
        MPI_Win_unlock(target, bigwin);
#else
        t0 = MPI_Wtime();
        MPI_Win_lock(MPI_LOCK_SHARED, target, MPI_MODE_NOCHECK, bigwin);
        for (int i=0; i<iter; i++) {
#if DEBUG>1
            fprintf(output, "%d: MPI_Get count=%d, disp=%d \n", comm_rank, count, i*count);
#endif
            MPI_Get(&(localbuffer[i*count]), count, MPI_DOUBLE, target, i*count /* disp */, count, MPI_DOUBLE, bigwin);
        }
        MPI_Win_unlock(target, bigwin);
        t1 = MPI_Wtime();
        dt[1] = t1-t0;
#endif

        errors = 0;
        for (int i=0; i<(iter*count); i++)
            errors += (localbuffer[i] != (1000.0+target));

        if (errors>0) {
            fprintf(output, "MPI_Put correctness check has failed at comm_rank %d! (errors = %d) \n", world_rank, errors);
#if DEBUG>0
            for (int i=0; i<(iter*count); i++)
                fprintf(output, "rank = %d, i = %d, localbuffer[i] = %lf (%lf) \n", world_rank, i, localbuffer[i], 1000.0+target);
#endif
            MPI_Abort(comm, errors);
        }

        /********************************/

        /* zero local buffer and window */
        for (int i=0; i<maxcount; i++)
            localbuffer[i] = 1000.0+comm_rank;

        MPI_Win_fence(MPI_MODE_NOSTORE, bigwin);
        MPI_Put(localbuffer, maxcount, MPI_DOUBLE, comm_rank, 0 /* disp */, maxcount, MPI_DOUBLE, bigwin);
        MPI_Win_fence(MPI_MODE_NOSTORE, bigwin);

        /********************************/

        for (int i=0; i<maxcount; i++)
            localbuffer[i] = 1000.0+target;

        t0 = MPI_Wtime();
        MPI_Win_lock(MPI_LOCK_SHARED, target, MPI_MODE_NOCHECK, bigwin);
        for (int i=0; i<iter; i++) {
#if DEBUG>1
            fprintf(output, "%d: MPI_Acc count=%d, disp=%d \n", comm_rank, count, i*count);
#endif
            MPI_Accumulate(&(localbuffer[i*count]), count, MPI_DOUBLE, target, i*count /* disp */, count, MPI_DOUBLE, MPI_SUM, bigwin);
        }
        MPI_Win_unlock(target, bigwin);
        t1 = MPI_Wtime();
        dt[2] = t1-t0;

        /********************************/

        MPI_Win_lock(MPI_LOCK_SHARED, target, MPI_MODE_NOCHECK, bigwin);
        MPI_Get(localbuffer, maxcount, MPI_DOUBLE, target, 0 /* disp */, maxcount, MPI_DOUBLE, bigwin);
        MPI_Win_unlock(target, bigwin);

        errors = 0;
        for (int i=0; i<(iter*count); i++)
            errors += (localbuffer[i] != (2000.0+2*target));

        if (errors>0) {
            fprintf(output, "MPI_Acc correctness check has failed at comm_rank %d! (errors = %d) \n", world_rank, errors);
#if DEBUG>0
            for (int i=0; i<(iter*count); i++)
                fprintf(output, "rank = %d, i = %d, localbuffer[i] = %lf (%lf) \n", world_rank, i, localbuffer[i], (2000.0+2*target));
#endif
            MPI_Abort(comm, errors);
        } 
        
        for (int i=0; i<3; i++) 
            dt[i] /= iter;
        MPI_Reduce(dt, avg, 3, MPI_DOUBLE, MPI_SUM, 0, comm);
        for (int i=0; i<3; i++) 
            avg[i] /= comm_size;
        if (comm_rank==0) {
            fprintf(output, "%8d %14.4lf %14.4lf %14.4lf %14.4lf %14.4lf %14.4lf\n", count,
                            1.0e6*avg[0], (1.0e-6*count*sizeof(double))/avg[0],
                            1.0e6*avg[1], (1.0e-6*count*sizeof(double))/avg[1],
                            1.0e6*avg[2], (1.0e-6*count*sizeof(double))/avg[2]);
            fflush(output);
        }
    }
    if (comm_rank==0) {
        fprintf(output, "------------------------------------------------------------------------------------------------------\n");
        fflush(output);
    }

    MPI_Win_free(&bigwin);
    MPI_Free_mem(bigwinlocptr);
    MPI_Free_mem(localbuffer);

    return;
}

void static_win_rma3(FILE * output, MPI_Comm comm, int max_mem)
{
#if MPI_VERSION >= 3
    int comm_rank = -1, world_rank = -1, comm_size = 0;
    MPI_Comm_rank(comm, &comm_rank);
    MPI_Comm_size(comm, &comm_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    int maxcount = max_mem/sizeof(double);

    if ( comm_rank == 0 )
        fprintf(output, "============== MPI-3 RMA - STATIC WINDOW ==============\n");

    double * localbuffer  = NULL;
    double * bigwinlocptr = NULL;
    MPI_Win bigwin;

    MPI_Alloc_mem(maxcount*sizeof(double), MPI_INFO_NULL, &localbuffer);
    MPI_Win_allocate(maxcount*sizeof(double), sizeof(double), MPI_INFO_NULL, comm, &bigwinlocptr, &bigwin);
    MPI_Win_lock_all(MPI_MODE_NOCHECK, bigwin); /* this is implicitly MPI_LOCK_SHARED */
    
    int target = (comm_rank+1)%comm_size;
#if DEBUG>0
    fprintf(output, "doing RMA from rank %d to rank %d \n", comm_rank, target);
    fflush(output);
#endif
    MPI_Barrier(comm);

    if (comm_rank==0) {
        fprintf(output, "%8s %29s %29s %29s \n", "",
                        "Put             ", "Get             ", "Acc (SUM)       ");
        fprintf(output, "%8s %29s %29s %29s \n", "doubles",
                        "time (us) bandwidth (MB/s)", "time (us) bandwidth (MB/s)", "time (us) bandwidth (MB/s)");
        fprintf(output, "------------------------------------------------------------------------------------------------------\n");
        fflush(output);
    }

    for (int count=1; count<maxcount; count*=2)
    {
        double t0, t1, dt[3], avg[3];
        int iter = maxcount/count;
        int errors = 0;

#if DEBUG>0
        fprintf(output, "%d: maxcount = %d, count = %d, iter = %d \n", comm_rank, maxcount, count, iter);
#endif

        /********************************/

        MPI_Barrier(comm);

        /* zero local buffer and window */
        for (int i=0; i<maxcount; i++)
            localbuffer[i] = 0.0;

        MPI_Put(localbuffer, maxcount, MPI_DOUBLE, comm_rank, 0 /* disp */, maxcount, MPI_DOUBLE, bigwin);
        MPI_Win_flush(comm_rank, bigwin);

        MPI_Barrier(comm);

        /********************************/

        for (int i=0; i<maxcount; i++)
            localbuffer[i] = 1000.0+target;

        t0 = MPI_Wtime();
        for (int i=0; i<iter; i++) {
#if DEBUG>1
            fprintf(output, "%d: MPI_Put count=%d, disp=%d \n", comm_rank, count, i*count);
#endif
            MPI_Put(&(localbuffer[i*count]), count, MPI_DOUBLE, target, i*count /* disp */, count, MPI_DOUBLE, bigwin);
        }
        MPI_Win_flush(target, bigwin);
        t1 = MPI_Wtime();
        dt[0] = t1-t0;

        /********************************/

#if 0
        MPI_Get(localbuffer, maxcount, MPI_DOUBLE, target, 0 /* disp */, maxcount, MPI_DOUBLE, bigwin);
        MPI_Win_flush(target, bigwin);
#else
        t0 = MPI_Wtime();
        for (int i=0; i<iter; i++) {
#if DEBUG>1
            fprintf(output, "%d: MPI_Get count=%d, disp=%d \n", comm_rank, count, i*count);
#endif
            MPI_Get(&(localbuffer[i*count]), count, MPI_DOUBLE, target, i*count /* disp */, count, MPI_DOUBLE, bigwin);
        }
        MPI_Win_flush(target, bigwin);
        t1 = MPI_Wtime();
        dt[1] = t1-t0;
#endif

        errors = 0;
        for (int i=0; i<(iter*count); i++)
            errors += (localbuffer[i] != (1000.0+target));

        if (errors>0) {
            fprintf(output, "MPI_Put correctness check has failed at comm_rank %d! (errors = %d) \n", world_rank, errors);
#if DEBUG>0
            for (int i=0; i<(iter*count); i++)
                fprintf(output, "rank = %d, i = %d, localbuffer[i] = %lf (%lf) \n", world_rank, i, localbuffer[i], (1000.0+target));
#endif
            MPI_Abort(comm, errors);
        }

        /********************************/

        MPI_Barrier(comm);

        /* zero local buffer and window */
        for (int i=0; i<maxcount; i++)
            localbuffer[i] = 1000.0+comm_rank;

        MPI_Put(localbuffer, maxcount, MPI_DOUBLE, comm_rank, 0 /* disp */, maxcount, MPI_DOUBLE, bigwin);
        MPI_Win_flush(comm_rank, bigwin);

        MPI_Barrier(comm);

        /********************************/

        for (int i=0; i<maxcount; i++)
            localbuffer[i] = 1000.0+target;

        t0 = MPI_Wtime();
        for (int i=0; i<iter; i++) {
#if DEBUG>1
            fprintf(output, "%d: MPI_Acc count=%d, disp=%d \n", comm_rank, count, i*count);
#endif
            MPI_Accumulate(&(localbuffer[i*count]), count, MPI_DOUBLE, target, i*count /* disp */, count, MPI_DOUBLE, MPI_SUM, bigwin);
        }
        MPI_Win_flush(target, bigwin);
        t1 = MPI_Wtime();
        dt[2] = t1-t0;

        MPI_Get(localbuffer, maxcount, MPI_DOUBLE, target, 0 /* disp */, maxcount, MPI_DOUBLE, bigwin);
        MPI_Win_flush(target, bigwin);

        errors = 0;
        for (int i=0; i<(iter*count); i++)
            errors += (localbuffer[i] != (2000.0+2*target));

        if (errors>0) {
            fprintf(output, "MPI_Acc correctness check has failed at comm_rank %d! (errors = %d) \n", world_rank, errors);
#if DEBUG>0
            for (int i=0; i<(iter*count); i++)
                fprintf(output, "rank = %d, i = %d, localbuffer[i] = %lf (%lf) \n", world_rank, i, localbuffer[i], (2000.0+2*target));
#endif
            MPI_Abort(comm, errors);
        } 

        for (int i=0; i<3; i++) 
            dt[i] /= iter;
        MPI_Reduce(dt, avg, 3, MPI_DOUBLE, MPI_SUM, 0, comm);
        for (int i=0; i<3; i++) 
            avg[i] /= comm_size;
        if (comm_rank==0) {
            fprintf(output, "%8d %14.4lf %14.4lf %14.4lf %14.4lf %14.4lf %14.4lf\n", count,
                            1.0e6*avg[0], (1.0e-6*count*sizeof(double))/avg[0],
                            1.0e6*avg[1], (1.0e-6*count*sizeof(double))/avg[1],
                            1.0e6*avg[2], (1.0e-6*count*sizeof(double))/avg[2]);
            fflush(output);
        }
    }
    if (comm_rank==0) {
        fprintf(output, "------------------------------------------------------------------------------------------------------\n");
        fflush(output);
    }

    MPI_Win_unlock_all(bigwin);
    MPI_Win_free(&bigwin);
    MPI_Free_mem(localbuffer);

#endif // MPI_VERSION >= 3

    return;
}
