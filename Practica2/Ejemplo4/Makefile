
ifndef CC
	CC = gcc
endif

CCFLAGS=-Wno-deprecated-declarations -O3 -lm

ifndef DEBUG
	CCFLAGS += -g
else
	CCFLAGS += -O3
endif

LIBS = -lOpenCL -fopenmp
TARGETS = my_ocl.c pi.c common.c

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

main: $(TARGETS)
	$(CC) $^ $(CCFLAGS) $(LIBS) -o $@


clean:
	rm -f main
