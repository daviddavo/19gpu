CC =  g++
NVCC = /usr/local/cuda-10.1/bin/nvcc 

EXE   = matrix 

SOURCES    = matrix_mul.cu main.cu


OBJS    = $(SOURCES:.cu=.o) $(SOURCES:.c=.o)

CFLAGS     = -O3 -lcuda

LIBS = -lm 

SOURCEDIR = .

$(EXE) :$(OBJS) 
	$(CC) $(CFLAGS)  -o $@ $? $(LIBS)

$(SOURCEDIR)/%.o : $(SOURCEDIR)/%.c
	$(CC) $(CFLAGS) -c -o $@ $<

$(SOURCEDIR)/%.o : $(SOURCEDIR)/%.cu 
	$(NVCC) $(CFLAGS) -c -o $@ $<


clean:
	rm -f $(OBJS) $(EXE)

# nvcc -c matrix_mul.cu -I/usr/local/cuda/NVIDIA_GPU_Computing_SDK/C/common/inc/ -L/usr/local/cuda/lib64/
# g++ -o matrix matrix_mul.o main.c -I./ -I/usr/local/cuda/NVIDIA_GPU_Computing_SDK/C/common/inc/ -L/usr/local/cuda/lib64/ -lcuda -lcudart

