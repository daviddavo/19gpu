CC =  g++
NVCC = nvcc 

EXE   = matrixmul 

SOURCES    = matrix_mul.cu


OBJS    = $(SOURCES:.cu=.o)

CFLAGS     = -O3

LIBS = -lm 

SOURCEDIR = .

$(EXE) :$(OBJS) 
	$(CC) $(CFLAGS)  -o $@ $? $(LIBS)

$(SOURCEDIR)/%.o : $(SOURCEDIR)/%.cu 
	$(CC) $(CFLAGS) -c -o $@ $<


clean:
	rm -f $(OBJS) $(EXE)
