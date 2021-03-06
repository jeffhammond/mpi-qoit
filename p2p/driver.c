#include "driver.h"

#ifdef _OPENMP
#define OMP_PARALLEL_FOR _Pragma("omp parallel for schedule(static)")
#else
#define OMP_PARALLEL_FOR
#endif

int main(int argc, char * argv[])
{
    /*********************************************************************************
     *                            INITIALIZE MPI
     *********************************************************************************/

#ifdef _OPENMP
    int requested=MPI_THREAD_MULTIPLE;
#else
    int requested=MPI_THREAD_SINGLE;
#endif
    int provided;
    MPI_Init_thread( &argc, &argv, requested, &provided );

    int np, me;
    MPI_Comm_rank( MPI_COMM_WORLD, &me );
    MPI_Comm_size( MPI_COMM_WORLD, &np );
    if (me==0) printf("MPI test program running on %d ranks. \n", np);

    if (provided<requested) {
        if (me==0) printf("MPI_Init_thread returned %s instead of %s so the test will exit. \n",
                          MPI_THREAD_STRING(provided), MPI_THREAD_STRING(requested) );
        MPI_Finalize();
        return 0;
    }

    double tt0 = MPI_Wtime();
    MPI_Barrier( MPI_COMM_WORLD );

    /*********************************************************************************
     *                            RUN TESTS
     *********************************************************************************/

    int maxcount = (argc>1) ? atoi(argv[1]) : 1024;
    int reps     = (argc>2) ? atoi(argv[2]) : 10;
    int * sbuf;
    int * rbuf;
    MPI_Alloc_mem(maxcount*sizeof(int), MPI_INFO_NULL, &sbuf);
    MPI_Alloc_mem(maxcount*sizeof(int), MPI_INFO_NULL, &rbuf);

    for (int count = 1; count<=maxcount; count*=2) {
        int n = maxcount/count;
        MPI_Request * reqs = malloc(2*n*sizeof(MPI_Request));
        int target = (me+1)%np;
        int origin = (me==0) ? (np-1) : me-1;
        int tag = 0;
        double totaltime = 0.0;
        for (int r=0; r<reps; r++) {
            OMP_PARALLEL_FOR
            for (int i=0; i<maxcount; i++) {
                sbuf[i] = me;
            }
            OMP_PARALLEL_FOR
            for (int i=0; i<maxcount; i++) {
                rbuf[i] = -1;
            }
            MPI_Barrier(MPI_COMM_WORLD);
            double t0 = MPI_Wtime();
            OMP_PARALLEL_FOR
            for (int i=0; i<n; i++) {
                MPI_Irecv(&(rbuf[count*i]), count, MPI_INT, origin, tag, MPI_COMM_WORLD, &(reqs[i]));
                MPI_Isend(&(sbuf[count*i]), count, MPI_INT, target, tag, MPI_COMM_WORLD, &(reqs[n+i]));
                /* If we want to measure overlap, add computation here. */
            }
            MPI_Waitall(2*n, reqs, MPI_STATUSES_IGNORE);
            double t1 = MPI_Wtime();
            if (r==0) {
                /* check correctness */
                int errors = 0;
                for (int i=0; i<n; i++) {
                    errors += (rbuf[i]!=origin);
                    if (errors>0) {
                        printf("there were %d errors!\n", errors);
                        MPI_Abort(MPI_COMM_WORLD, 1);
                    }
                }
            } else {
                /* accumulate timing */
                totaltime += (t1-t0);
            }
        }
        totaltime /= reps; /* total time for one iteration */
        size_t bytes = count*sizeof(int);
        printf("%d: %d %zu byte messages in %e s - %lf us latency - bandwidth %lf MiB/s - %lf MMPS\n",
                me, n, bytes, totaltime, 1.e6*totaltime/n, 1.e-6*n*bytes/totaltime, 1.e-6*n/totaltime);
        fflush(stdout);
        MPI_Barrier(MPI_COMM_WORLD);
        free(reqs);
    }

    MPI_Free_mem(rbuf);
    MPI_Free_mem(sbuf);

    /*********************************************************************************
     *                            CLEAN UP
     *********************************************************************************/

    MPI_Barrier( MPI_COMM_WORLD );
    fflush(stdout);
    sleep(1);
    MPI_Barrier( MPI_COMM_WORLD );

    double tt1 = MPI_Wtime();
    if (me==0)
       printf("TEST FINISHED SUCCESSFULLY IN %lf SECONDS \n", tt1-tt0);
    fflush(stdout);

    MPI_Finalize();

    return 0;
}
