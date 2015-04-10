#include "driver.h"

int main(int argc, char * argv[])
{
    /*********************************************************************************
     *                            INITIALIZE MPI
     *********************************************************************************/

    int requested=MPI_THREAD_MULTIPLE, provided;
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

    double t0 = MPI_Wtime();
    MPI_Barrier( MPI_COMM_WORLD );

    /*********************************************************************************
     *                            RUN TESTS
     *********************************************************************************/


    /*********************************************************************************
     *                            CLEAN UP
     *********************************************************************************/

    MPI_Barrier( MPI_COMM_WORLD );

    double t1 = MPI_Wtime();
    if (me==0)
       printf("TEST FINISHED SUCCESSFULLY IN %lf SECONDS \n", t1-t0);
    fflush(stdout);

    MPI_Finalize();

    return 0;
}
