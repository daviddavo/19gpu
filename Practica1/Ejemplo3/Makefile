CC =  /usr/bin/g++
NVCC = /usr/local/cuda-10.1/bin/nvcc 

EXE   = matrix 

SOURCES    = main.c matrix_mul.cu 

OBJS    = $(SOURCES:.cu=.o) $(SOURCES:.c=.o)

CFLAGS     = -O3

LIBS = -lm 

SOURCEDIR = .

$(EXE) :$(OBJS) 
	$(CC) $(CFLAGS)  -o $@ $? $(LIBS)

$(SOURCEDIR)/matrix.o : $(SOURCEDIR)/matrix.c
	$(CC) $(CFLAGS) -c -o $@ matrix_mul.o $< -I./ -lcuda lcudart

$(SOURCEDIR)/%.o : $(SOURCEDIR)/%.cu 
	$(NVCC) $(CFLAGS) -c -o $@ $<

clean:
	rm -f main.o matrix_mul.o $(EXE)

# nvcc -c matrix_mul.cu -I/usr/local/cuda/NVIDIA_GPU_Computing_SDK/C/common/inc/ -L/usr/local/cuda/lib64/
# g++ -o matrix matrix_mul.o main.c -I./ -I/usr/local/cuda/NVIDIA_GPU_Computing_SDK/C/common/inc/ -L/usr/local/cuda/lib64/ -lcuda -lcudart

