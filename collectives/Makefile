CC      = mpicc
COPT    =

LD      = $(CC)
CFLAGS  = $(COPT) -std=c99
CFLAGS += -DDO_COMM_WORLD #-DDO_COMM_WORLD_DUP -DDO_WORLD_REORDERED 
#CFLAGS += -DDO_LEFT_RIGHT -DDO_ODD_EVEN -DDO_WORLD_MINUS_ONE -DDO_GEOM_PROG
#CFLAGS += -DDO_COMM_WORLD_JITTER
CFLAGS += -DPRE_BARRIER_HACK
#CFLAGS += -DDEBUG

LDFLAGS = $(COPT) $(CFLAGS)
LIBS    = -lm

TESTS := driver.x

OBJECTS := meminfo.o safemalloc.o bcast.o reduce.o allreduce.o gather.o allgather.o scatter.o alltoall.o reducescatter.o

HEADERS := safemalloc.h driver.h

all: $(TESTS)

%.x: %.c $(OBJECTS)
	$(LD) $(LDFLAGS) $< $(OBJECTS) $(LIBS) -o $@

%.o: %.c $(HEADERS)
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	$(RM) $(RMFLAGS) $(OBJECTS)
	$(RM) $(RMFLAGS) $(TESTS)
	$(RM) $(RMFLAGS) -r *.dSYM



