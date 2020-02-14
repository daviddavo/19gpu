CC = gcc 

EXE   = matrixmul 

SOURCES    = main.c


OBJS    = $(SOURCES:.c=.o)

CFLAGS     = -O3

LIBS = -lm 

SOURCEDIR = .

$(EXE) :$(OBJS) 
	$(CC) $(CFLAGS)  -o $@ $? $(LIBS)

$(SOURCEDIR)/%.o : $(SOURCEDIR)/%.c 
	$(CC) $(CFLAGS) -c -o $@ $<


clean:
	rm -f $(OBJS) $(EXE)
