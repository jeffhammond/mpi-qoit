CC		= /opt/mpich/dev/intel/default/bin/mpicc
#CC     := mpicc
COPT   := -g -O2 -Wall -std=c99 -fopenmp

LD      = $(CC)
CFLAGS  = $(COPT) 
CFLAGS += -DDEBUG=0

LDFLAGS = $(CFLAGS)
LIBS    = 

TESTS := driver.x

OBJECTS := 

HEADERS := safemalloc.h driver.h

all: $(TESTS)

%.x: %.c $(OBJECTS)
	$(LD) $(LDFLAGS) $< $(OBJECTS) $(LIBS) -o $@

%.o: %.c $(HEADERS)
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	$(RM) $(RMFLAGS) $(OBJECTS)
	$(RM) $(RMFLAGS) $(TESTS)
	$(RM) $(RMFLAGS) *.dwarf
	$(RM) $(RMFLAGS) -r *.dSYM



