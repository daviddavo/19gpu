#include <stdio.h>

#include "my_ocl.h"
#include "common.h"

#ifndef BLOCK_DIM
#define BLOCK_DIM 32
#endif

#define _STR(arg) #arg
#define STR(arg) #arg "=" _STR(arg)

void bodiesToCL(body * bodies, float weight[], cl_float3 vel[], cl_float3 pos[], int n) {
	for (int i = 0; i < n; i++) {
		weight[i] = bodies[i].m;

		pos[i].x  = bodies[i].x;
		pos[i].y  = bodies[i].y;
		pos[i].z  = bodies[i].z;

		vel[i].x = bodies[i].vx;
		vel[i].y = bodies[i].vy;
		vel[i].z = bodies[i].vz;
	}
}

int nbodiesOCL(body * data, const int nBodies, const int nIters, const float dt)
{
    cl_context context;
    cl_command_queue command_queue;
    cl_device_id device_id;
    cl_int err;
    cl_program program;
    cl_kernel nbodies_kernel;

    float * bweight;
    cl_float3 * bvelin;
    cl_float3 * bposin;
    
    // We use float4 so vectors are aligned
    cl_mem dbweight; // float. Weight of each body (const)
    cl_mem dbvelin; // float4. Initial Velocity of each body
    cl_mem dbposin; // float4. Initial Position of each body

    if (getContextDefault(&context, &device_id) != EXIT_SUCCESS) {
        fprintf(stderr, "Failed creating context\n");
        return EXIT_FAILURE;
    }

    command_queue = clCreateCommandQueue(context, device_id, CL_QUEUE_PROFILING_ENABLE, &err);
    CL_CHECK(err);

    if (compileFromFileName(&program, "nbodies_kernel.cl", "-D " STR(BLOCK_DIM),
        context, device_id) != EXIT_SUCCESS) {
        fprintf(stderr, "Failed compiling program nbodies_kernel.cl\n");
        return EXIT_FAILURE;
    }

    nbodies_kernel = clCreateKernel(program, "nbodies_naive", &err);
    CL_CHECK(err);

    size_t global = nBodies;

    bweight = calloc(sizeof(float), nBodies);
    bvelin   = calloc(sizeof(cl_float3), nBodies);
    bposin   = calloc(sizeof(cl_float3), nBodies);
    
    bodiesToCL(data, bweight, bvelin, bposin, nBodies);

    dbweight = clCreateBuffer(context, CL_MEM_COPY_HOST_PTR,
        sizeof(float) * nBodies, bweight, &err); CL_CHECK(err);
    dbvelin  = clCreateBuffer(context, CL_MEM_COPY_HOST_PTR,
        sizeof(cl_float3) * nBodies, bvelin, &err); CL_CHECK(err);
    dbposin  = clCreateBuffer(context, CL_MEM_COPY_HOST_PTR,
        sizeof(cl_float3) * nBodies, bposin, &err); CL_CHECK(err);

    CL_CHECK(clSetKernelArg(nbodies_kernel, 0, sizeof(cl_mem), &dbweight));
    CL_CHECK(clSetKernelArg(nbodies_kernel, 1, sizeof(cl_mem), &dbvelin));
    CL_CHECK(clSetKernelArg(nbodies_kernel, 2, sizeof(cl_mem), &dbposin));
    CL_CHECK(clSetKernelArg(nbodies_kernel, 3, sizeof(cl_float), &dt));
    CL_CHECK(clSetKernelArg(nbodies_kernel, 4, sizeof(cl_int), &nIters));

    cl_event event;
    cl_ulong tstart, tend;
    CL_CHECK(clEnqueueNDRangeKernel(command_queue, nbodies_kernel,
        1, NULL, &global, NULL, 0, NULL, &event));
    CL_CHECK(clWaitForEvents(1, &event));
    CL_CHECK(clFinish(command_queue));

    CL_CHECK(clGetEventProfilingInfo(event, 
        CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &tstart, NULL));
    CL_CHECK(clGetEventProfilingInfo(event,
        CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &tend, NULL));

    // tstart and tend are nanoseconds
    double ms = ((double)(tend - tstart))/(1000000.0l);
    printf("Total Kernel duration: %0.5lfms\n", ms);
    printf("%d bodies with %d iterations: %f Millions Interactions/second\n", nBodies, nIters, nBodies*nBodies*1e3/ms);

    CL_CHECK(clReleaseMemObject(dbweight));
    CL_CHECK(clReleaseMemObject(dbvelin));
    CL_CHECK(clReleaseMemObject(dbposin));

    CL_CHECK(clReleaseProgram(program));
    CL_CHECK(clReleaseKernel(nbodies_kernel));
    CL_CHECK(clReleaseCommandQueue(command_queue));
    CL_CHECK(clReleaseContext(context));
    
    return EXIT_SUCCESS;
}
