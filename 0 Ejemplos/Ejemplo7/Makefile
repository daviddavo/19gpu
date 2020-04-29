
CXX = g++
CC = gcc

.PHONY: clean run

#OpenCV
CXX_OPENCV_FLAGS+=`pkg-config opencv --cflags`
LD_OPENCV_FLAGS+=`pkg-config opencv --libs`

# CUDA
CUDA_PATH ?= /usr/local/cuda
NVCC = $(CUDA_PATH)/bin/nvcc
# CUDA_INCLUDE=/usr/local/cuda/include/
CUDA_INCLUDE=$(CUDA_PATH)/include/
CUDA_LIB=$(CUDA_PATH)/lib64/
# CUDA_LIB=/opt/cuda/lib64/
# CUDA_LIB ?= $"/opt/cuda/lib64/:/opt/cuda/lib/:/usr/local/cuda/lib64/"

CFLAGS=-O3 -I. 
CXXFLAGS=-O3 -I.

LIBS =-lpng -lm -lcudart

SRC = png_io.o routinesCPU.o routinesGPU.o main.o
	
%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

%.o: %.cu
	$(NVCC) $(CFLAGS) -I. -I$(CUDA_INCLUDE) -L$(CUDA_LIB) $(LIBS) -c -o $@ $<


%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<

image: $(SRC) 
	$(CXX) -o image  $(SRC) $(CXXFLAGS) -I$(CUDA_INCLUDE) -L$(CUDA_LIB) $(LIBS) 

run:
	./processAll.sh

clean:
	rm -f *.o image
