#ifndef __COMON_H
#define __COMON_H

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif /*__APPLE__*/

double getMicroSeconds();
char * err_code(cl_int err_in);
int output_device_info(cl_device_id device_id);
void init_seed();
void init2Drand(float **buffer, int n);
void init1Drand(float *buffer, int n);
float *getmemory1D( int nx );
float **getmemory2D(int nx, int ny);
int check(float *GPU, float *CPU, int n);
void printMATRIX(float *m, int n);

int getContextDefault(cl_context * context, cl_device_id * device_id);
int compileFromFileName(cl_program * program, char fname[], char compargs [], 
    cl_context context, cl_device_id device_id);

#define __CL_CHECK(f, fname) {\
    const cl_int err = f; \
    if (err != CL_SUCCESS) { \
        fprintf(stderr, "Error at %s:%d:%s ", __FILE__, __LINE__, fname); \
        fprintf(stderr, "CL code: %d (%s)\n", err, err_code(err)); \
        return EXIT_FAILURE; \
    }\
}

#define CL_CHECK(f) __CL_CHECK(f, #f)

#endif /* __COMMON_H */
