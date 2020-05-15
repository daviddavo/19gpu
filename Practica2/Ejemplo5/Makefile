
ifndef CC
	CC = gcc
endif

CCFLAGS=-lm -Wno-deprecated-declarations

LIBS = -lOpenCL -fopenmp
TARGETS = common.c my_ocl.c nbody.c 

# Change this variable to specify the device type
# to the OpenCL device type of choice. You can also
# edit the variable in the source.
ifndef DEVICE
	DEVICE = CL_DEVICE_TYPE_DEFAULT
endif

# Check our platform and make sure we define the APPLE variable
# and set up the right compiler flags and libraries
PLATFORM = $(shell uname -s)
ifeq ($(PLATFORM), Darwin)
	LIBS = -framework OpenCL
endif

CCFLAGS += -D DEVICE=$(DEVICE)

all: release

debug: CCFLAGS += -g3 -DDEBUG
debug: main

release: CCFLAGS += -O3
release: main

main: $(TARGETS)
	$(CC) $^ $(CCFLAGS) $(LIBS) -o $@


clean:
	rm -f main
