#include <stdio.h>
#include <stdlib.h>
#include "my_ocl.h"
#include "CL/cl.h"

// Remember to change it also on the *.cl
#define BLOCK_DIM 16

extern double getMicroSeconds();
extern char *err_code (cl_int err_in);
extern int output_device_info(cl_device_id device_id);
extern float *getmemory1D( int nx );
extern int check(float *GPU, float *CPU, int n);

int remove_noiseOCL(float *im, float *image_out, 
	float thredshold, int window_size,
	int height, int width)
{
	printf("Not Implemented yet!!\n");

    cl_mem dim;
    cl_mem dim_out;

	// OpenCL host variables
	cl_uint num_devs_returned;
	cl_context_properties properties[3];
	cl_device_id device_id;
	cl_int err;
	cl_platform_id platform_id;
	cl_uint num_platforms_returned;
	cl_context context;
	cl_command_queue command_queue;
	cl_program program;
	cl_kernel kernel_naive;
	size_t global[2];
    size_t work[2];
	
	// variables used to read kernel source file
	FILE *fp;
	long filelen;
	long readlen;
	char *kernel_src;  // char string to hold kernel source

	// read the kernel
	fp = fopen("ocl_kernels.cl","r");
	fseek(fp,0L, SEEK_END);
	filelen = ftell(fp);
	rewind(fp);

	kernel_src = malloc(sizeof(char)*(filelen+1));
	readlen = fread(kernel_src,1,filelen,fp);
	if(readlen!= filelen)
	{
		printf("error reading file\n");
		exit(1);
	}
	
	// ensure the string is NULL terminated
	kernel_src[filelen]='\0';

	// Set up platform and GPU device

	cl_uint numPlatforms;

	// Find number of platforms
	err = clGetPlatformIDs(0, NULL, &numPlatforms);
	if (err != CL_SUCCESS || numPlatforms <= 0)
	{
		printf("Error: Failed to find a platform!\n%s\n",err_code(err));
		return EXIT_FAILURE;
	}

	// Get all platforms
	cl_platform_id Platform[numPlatforms];
	err = clGetPlatformIDs(numPlatforms, Platform, NULL);
	if (err != CL_SUCCESS || numPlatforms <= 0)
	{
		printf("Error: Failed to get the platform!\n%s\n",err_code(err));
		return EXIT_FAILURE;
	}

	// Secure a GPU
	int i;
	for (i = 0; i < numPlatforms; i++)
	{
		err = clGetDeviceIDs(Platform[i], DEVICE, 1, &device_id, NULL);
		if (err == CL_SUCCESS)
		{
			break;
		}
	}

	if (device_id == NULL)
	{
		printf("Error: Failed to create a device group!\n%s\n",err_code(err));
		return EXIT_FAILURE;
	}

	err = output_device_info(device_id);

	// Create a compute context 
	context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
	if (!context)
	{
		printf("Error: Failed to create a compute context!\n%s\n", err_code(err));
		return EXIT_FAILURE;
	}

	// Create a command queue
	cl_command_queue commands = clCreateCommandQueue(context, device_id, 0, &err);
	if (!commands)
	{
		printf("Error: Failed to create a command commands!\n%s\n", err_code(err));
		return EXIT_FAILURE;
	}

	// create command queue 
	command_queue = clCreateCommandQueue(context,device_id, 0, &err);
	if (err != CL_SUCCESS)
	{	
		printf("Unable to create command queue. Error Code=%d\n",err);
		exit(1);
	}
	 
	// create program object from source. 
	// kernel_src contains source read from file earlier
	program = clCreateProgramWithSource(context, 1 ,(const char **)
                                          &kernel_src, NULL, &err);
	if (err != CL_SUCCESS)
	{	
		printf("Unable to create program object. Error Code=%d\n",err);
		exit(1);
	}       
	
	err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
	if (err != CL_SUCCESS)
	{
        	printf("Build failed. Error Code=%d\n", err);

		size_t len;
		char buffer[2048];
		// get the build log
		clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG,
                                  sizeof(buffer), buffer, &len);
		printf("--- Build Log -- \n %s\n",buffer);
		exit(1);
	}

	kernel_naive = clCreateKernel(program, "ocl_naive", &err);
	if (err != CL_SUCCESS)
	{	
		printf("Unable to create kernel object. Error Code=%d\n",err);
		exit(1);
	}

    // create buffer objects to input and output args of kernel function
    dim     = clCreateBuffer(context,  CL_MEM_READ_ONLY, sizeof(float) * height * width, NULL, NULL);
    dim_out = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * height * width, NULL, NULL);

    // Write image vector into compute device memory 
    err = clEnqueueWriteBuffer(command_queue, dim, CL_TRUE, 0, sizeof(float) *height*width, im, 0, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to write im to source array!\n%s\n", err_code(err));
        exit(1);
    }

	// set the kernel arguments
    err  = clSetKernelArg(kernel_naive, 0, sizeof(cl_mem), &dim_out);
    err |= clSetKernelArg(kernel_naive, 1, sizeof(cl_mem), &dim);
    err |= clSetKernelArg(kernel_naive, 2, sizeof(cl_uint), &width);
    err |= clSetKernelArg(kernel_naive, 3, sizeof(cl_uint), &height); 
	if (err != CL_SUCCESS)
	{
		printf("Unable to set kernel arguments. Error Code=%d (%s)\n",err, err_code(err));
		exit(1);
	}

	// set the global work dimension size
    // TODO: Revise this
	global[0] = height;
	global[1] = width;
    work[0] = BLOCK_DIM;
    work[1] = BLOCK_DIM;

    // NOW TO RUN THE DAMN KERNEL
	err = clEnqueueNDRangeKernel(command_queue, kernel_naive, 2, NULL, 
        global, NULL /* work */, 0, NULL, NULL);

	if (err != CL_SUCCESS)
	{	
		printf("Unable to enqueue kernel command 1. Error Code=%d (%s)\n",err, err_code(err));
		exit(1);
	}

	// wait for the command to finish
	clFinish(command_queue);

	// read the output back to host memory
	err = clEnqueueReadBuffer(command_queue, dim_out, CL_TRUE, 0, 
            sizeof(float) * width * height, image_out, 0, NULL, NULL);
	if (err != CL_SUCCESS)
	{	
		printf("Error enqueuing read buffer command. Error Code=%d (%s)\n",err, err_code(err));
		exit(1);
	}

	// clean up
	clReleaseProgram(program);
	clReleaseKernel(kernel_naive);
	clReleaseCommandQueue(command_queue);
	clReleaseContext(context);
	free(kernel_src);
	return 0;
}
