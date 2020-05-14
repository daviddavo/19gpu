#include <stdio.h>

#include "my_ocl.h"
#include "common.h"

#ifndef BLOCK_DIM
#define BLOCK_DIM 32
#endif

#define _STR(arg) #arg
#define STR(arg) #arg "=" _STR(arg)

int nbodiesOCL(int nBodies)
{
    cl_context context;
    cl_command_queue command_queue;
    cl_device_id device_id;
    cl_int err;
    cl_program program;
    cl_kernel nbodies_kernel;
    
    // We use float4 so vectors are aligned
    cl_mem dbweight; // float. Weight of each body (const)
    cl_mem dbvelin; // float4. Initial Velocity of each body
    cl_mem dbposin; // float4. Initial Position of each body

    if (getContextDefault(&context, &device_id) != EXIT_SUCCESS) {
        fprintf(stderr, "Failed creating context\n");
        return EXIT_FAILURE;
    }

    command_queue = clCreateCommandQueue(context, device_id, 0, &err);
    CL_CHECK(err);

    if (compileFromFileName(&program, "nbodies_kernel.cl", "-D " STR(BLOCK_DIM),
        context, device_id) != EXIT_SUCCESS) {
        fprintf(stderr, "Failed compiling program nbodies_kernel.cl\n");
        return EXIT_FAILURE;
    }

    nbodies_kernel = clCreateKernel(program, "nbodies_naive", &err);
    CL_CHECK(err);

	printf("Not Implemented yet!!\n");
}
