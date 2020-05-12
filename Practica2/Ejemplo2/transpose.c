// tranposeMatrix.c
#include <stdio.h>
#include <stdlib.h>
#include "CL/cl.h"

//pick up device type from compiler command line or from 
//the default type
#ifndef DEVICE
#define DEVICE CL_DEVICE_TYPE_DEFAULT
#endif


/* From common.c */
extern double getMicroSeconds();
extern char *err_code (cl_int err_in);
extern int output_device_info(cl_device_id device_id);
extern float *getmemory1D( int nx );
extern int check(float *GPU, float *CPU, int n);

/*
 * Traspose 1D version
 */
void transpose1D(float *in, float *out, int n)
{
	int i, j;

	for(j=0; j < n; j++) 
		for(i=0; i < n; i++) 
			out[j*n+i] = in[i*n+j]; 
}

int main(int argc, char **argv)
{
	int n;	
	float *darray1D;
	float *darray1D_trans;

	float *array1D;
	float *array1D_trans;
	float *array1D_trans_GPU;
	
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
	cl_kernel kernel;
	size_t global[2];
	
	// variables used to read kernel source file
	FILE *fp;
	long filelen;
	long readlen;
	char *kernel_src;  // char string to hold kernel source


	if (argc==2)
		n = atoi(argv[1]);
	else {
		n = 4096;
		printf("./exec n (by default n=%i)\n", n);
	}

	// initialize inputMatrix with some data and print it
	int x,y;
	int data=0;

	array1D           = getmemory1D( n*n );
	array1D_trans     = getmemory1D( n*n );
	array1D_trans_GPU = getmemory1D( n*n );

	init1Drand(array1D, n*n);

	// read the kernel
	fp = fopen("transpose_kernel.cl","r");
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

	kernel = clCreateKernel(program, "matrixTranspose", &err);
	if (err != CL_SUCCESS)
	{	
		printf("Unable to create kernel object. Error Code=%d\n",err);
		exit(1);
	}

	// create buffer objects to input and output args of kernel function
	darray1D       =   ....;

	darray1D_trans =   ....;

	// set the kernel arguments
	if ( clSetKernelArg(kernel, 0, sizeof(cl_mem), &....) ||
         clSetKernelArg(kernel, 1, sizeof(cl_mem), &.....) ||
         clSetKernelArg(kernel, 2, sizeof(cl_uint), &n) != CL_SUCCESS)
	{
		printf("Unable to set kernel arguments. Error Code=%d\n",err);
		exit(1);
	}

	// set the global work dimension size
	global[0]= n;
	global[1]= n;

	// Enqueue the kernel object with 
	// Dimension size = 2, 
	// global worksize = global, 
	// local worksize = NULL - let OpenCL runtime determine
	// No event wait list
	double t0d = getMicroSeconds();
	err = clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, 
                                   global, NULL, 0, NULL, NULL);
	double t1d = getMicroSeconds();

	if (err != CL_SUCCESS)
	{	
		printf("Unable to enqueue kernel command. Error Code=%d\n",err);
		exit(1);
	}

	// wait for the command to finish
	clFinish(command_queue);

	// read the output back to host memory
	err = clEnqueueReadBuffer(....);
	if (err != CL_SUCCESS)
	{	
		printf("Error enqueuing read buffer command. Error Code=%d\n",err);
		exit(1);
	}


	// Tranpose in CPU
	double t0h = getMicroSeconds();
	transpose1D(array1D, array1D_trans, n);
	double t1h = getMicroSeconds();

	if (check(array1D_trans_GPU, array1D_trans, n*n))
		printf("\n\nTranspose Host-Device differs!!\n");
	else
		printf("\n\nTranspose Host-Device tHost=%f (s.) tDevice=%f (s.)\n", (t1h-t0h)/1000000, (t1d-t0d)/1000000);


//	printMATRIX(array1D, n);
//	printMATRIX(array1D_trans_GPU, n);

	// clean up
	clReleaseProgram(program);
	clReleaseKernel(kernel);
	clReleaseCommandQueue(command_queue);
	clReleaseContext(context);
	free(kernel_src);
	free(array1D);
	free(array1D_trans);
	free(array1D_trans_GPU);
	return 0;
}


