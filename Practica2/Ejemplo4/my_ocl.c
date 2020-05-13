#include <stdio.h>
#include "my_ocl.h"
#include "common.h"

#ifndef BLOCK_DIM 
#define BLOCK_DIM 32 
#endif

#define _STR(arg) #arg
#define STR(arg) #arg "=" _STR(arg)

int roundUp(unsigned toRound, unsigned mul) {
    if (mul == 0) return toRound;

    int r = toRound % mul;
    if (r == 0) return toRound;

    return toRound + mul - r;
}

double calc_piOCL(int n)
{
    cl_context context;
    cl_command_queue command_queue;
    cl_device_id device_id;
    cl_int err;
    cl_program program;
    cl_kernel pi_kernel;
    cl_mem dpi_arr;
    float * pi_arr;
    double pi;

    if (getContextDefault(&context, &device_id) != EXIT_SUCCESS) {
        fprintf(stderr, "Failed creating context\n");
        return -1.0f;
    }

    command_queue = clCreateCommandQueue(context, device_id, 0, &err);
    CL_CHECK(err);
    
    if (compileFromFileName(&program, "pi_kernels.cl", "-D " STR(BLOCK_DIM),
        context, device_id) != EXIT_SUCCESS) 
    {
        fprintf(stderr, "Failed compiling from filename\n");
        return -1.0f;
    }

    pi_kernel = clCreateKernel(program, "pi_naive", &err);
    CL_CHECK(err);

    size_t global = roundUp(n, BLOCK_DIM);
    size_t work = BLOCK_DIM;
    size_t nblocks = global / work;

    if (n % BLOCK_DIM) 
        printf("Warning!!, we'll be rounding up the number of kernels to %d\n", 
            global);

    printf("Running kernel with %d blocks of size %d. Total: %d\n", nblocks, work, global);

    dpi_arr = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * nblocks, NULL, NULL);

    // Setting kernel args
    CL_CHECK(clSetKernelArg(pi_kernel, 0, sizeof(cl_mem), &dpi_arr));

    CL_CHECK(clEnqueueNDRangeKernel(command_queue, pi_kernel, 1, NULL, 
        &global, &work, 0, NULL, NULL));

    pi_arr = malloc(sizeof(float) * nblocks);

    // Wait for it to finish
    clFinish(command_queue);

    CL_CHECK(clEnqueueReadBuffer(command_queue, dpi_arr, CL_TRUE, 0,
        sizeof(float) * nblocks, pi_arr, 0, NULL, NULL));

    // TODO: To opencl
    // Maybe make one thread do it
    pi = 0.0f;
    for (int i = 0; i < nblocks; i++) {
        pi += pi_arr[i];
    }
    pi = pi/global;

    clReleaseProgram(program);
    clReleaseKernel(pi_kernel);
    clReleaseCommandQueue(command_queue);
    clReleaseContext(context);
    free(pi_arr);
    return pi;
}
