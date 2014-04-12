#ifndef DRIVER_H
#define DRIVER_H

void print_meminfo(FILE * output, char * message);

void bcast_only(FILE * output, MPI_Comm comm, int max_mem);
void bcast_vs_scatter_allgather(FILE * output, MPI_Comm comm, int max_mem);

void gather_only(FILE * output, MPI_Comm comm, int max_mem);
void allgather_only(FILE * output, MPI_Comm comm, int max_mem);

void scatter_only(FILE * output, MPI_Comm comm, int max_mem);
void alltoall_only(FILE * output, MPI_Comm comm, int max_mem);

void reduce_only(FILE * output, MPI_Comm comm, int max_mem);
void allreduce_only(FILE * output, MPI_Comm comm, int max_mem);

void reducescatter_only(FILE * output, MPI_Comm comm, int max_mem);
void reducescatterblock_only(FILE * output, MPI_Comm comm, int max_mem);

#endif // DRIVER_H
