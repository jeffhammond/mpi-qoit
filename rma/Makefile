CC      = mpicc
COPT    = -g -O2 -Wall -std=gnu99

LD      = $(CC)
CFLAGS  = $(COPT) 
CFLAGS += -DDEBUG=0

LDFLAGS = $(COPT) $(CFLAGS)
LIBS    = -lm

TESTS := driver.x

#OBJECTS := static-win-rma2.o static-win-rma3.o
OBJECTS := static-win-rma.o

HEADERS := safemalloc.h driver.h

all: $(TESTS)

%.x: %.c $(OBJECTS)
	$(LD) $(LDFLAGS) $< $(OBJECTS) $(LIBS) -o $@

%.o: %.c $(HEADERS)
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	$(RM) $(RMFLAGS) $(OBJECTS)

realclean: clean
	$(RM) $(RMFLAGS) $(TESTS)
	$(RM) $(RMFLAGS) -r *.dSYM



