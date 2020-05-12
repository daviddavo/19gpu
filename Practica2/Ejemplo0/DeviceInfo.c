/*
 * Display Device Information
 *
 * Script to print out some information about the OpenCL devices
 * and platforms available on your system
 *
 * History: C++ version written by Tom Deakin, 2012
 *          Ported to C by Tom Deakin, July 2013
*/

#include <stdio.h>
#include <stdlib.h>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#include <unistd.h>
#else
#include <CL/cl.h>
#endif

char *err_code (cl_int err_in)
{
    switch (err_in) {

        case CL_SUCCESS :
            return (char*)" CL_SUCCESS ";
        case CL_DEVICE_NOT_FOUND :
            return (char*)" CL_DEVICE_NOT_FOUND ";
        case CL_DEVICE_NOT_AVAILABLE :
            return (char*)" CL_DEVICE_NOT_AVAILABLE ";
        case CL_COMPILER_NOT_AVAILABLE :
            return (char*)" CL_COMPILER_NOT_AVAILABLE ";
        case CL_MEM_OBJECT_ALLOCATION_FAILURE :
            return (char*)" CL_MEM_OBJECT_ALLOCATION_FAILURE ";
        case CL_OUT_OF_RESOURCES :
            return (char*)" CL_OUT_OF_RESOURCES ";
        case CL_OUT_OF_HOST_MEMORY :
            return (char*)" CL_OUT_OF_HOST_MEMORY ";
        case CL_PROFILING_INFO_NOT_AVAILABLE :
            return (char*)" CL_PROFILING_INFO_NOT_AVAILABLE ";
        case CL_MEM_COPY_OVERLAP :
            return (char*)" CL_MEM_COPY_OVERLAP ";
        case CL_IMAGE_FORMAT_MISMATCH :
            return (char*)" CL_IMAGE_FORMAT_MISMATCH ";
        case CL_IMAGE_FORMAT_NOT_SUPPORTED :
            return (char*)" CL_IMAGE_FORMAT_NOT_SUPPORTED ";
        case CL_BUILD_PROGRAM_FAILURE :
            return (char*)" CL_BUILD_PROGRAM_FAILURE ";
        case CL_MAP_FAILURE :
            return (char*)" CL_MAP_FAILURE ";
        case CL_MISALIGNED_SUB_BUFFER_OFFSET :
            return (char*)" CL_MISALIGNED_SUB_BUFFER_OFFSET ";
        case CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST :
            return (char*)" CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST ";
        case CL_INVALID_VALUE :
            return (char*)" CL_INVALID_VALUE ";
        case CL_INVALID_DEVICE_TYPE :
            return (char*)" CL_INVALID_DEVICE_TYPE ";
        case CL_INVALID_PLATFORM :
            return (char*)" CL_INVALID_PLATFORM ";
        case CL_INVALID_DEVICE :
            return (char*)" CL_INVALID_DEVICE ";
        case CL_INVALID_CONTEXT :
            return (char*)" CL_INVALID_CONTEXT ";
        case CL_INVALID_QUEUE_PROPERTIES :
            return (char*)" CL_INVALID_QUEUE_PROPERTIES ";
        case CL_INVALID_COMMAND_QUEUE :
            return (char*)" CL_INVALID_COMMAND_QUEUE ";
        case CL_INVALID_HOST_PTR :
            return (char*)" CL_INVALID_HOST_PTR ";
        case CL_INVALID_MEM_OBJECT :
            return (char*)" CL_INVALID_MEM_OBJECT ";
        case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR :
            return (char*)" CL_INVALID_IMAGE_FORMAT_DESCRIPTOR ";
        case CL_INVALID_IMAGE_SIZE :
            return (char*)" CL_INVALID_IMAGE_SIZE ";
        case CL_INVALID_SAMPLER :
            return (char*)" CL_INVALID_SAMPLER ";
        case CL_INVALID_BINARY :
            return (char*)" CL_INVALID_BINARY ";
        case CL_INVALID_BUILD_OPTIONS :
            return (char*)" CL_INVALID_BUILD_OPTIONS ";
        case CL_INVALID_PROGRAM :
            return (char*)" CL_INVALID_PROGRAM ";
        case CL_INVALID_PROGRAM_EXECUTABLE :
            return (char*)" CL_INVALID_PROGRAM_EXECUTABLE ";
        case CL_INVALID_KERNEL_NAME :
            return (char*)" CL_INVALID_KERNEL_NAME ";
        case CL_INVALID_KERNEL_DEFINITION :
            return (char*)" CL_INVALID_KERNEL_DEFINITION ";
        case CL_INVALID_KERNEL :
            return (char*)" CL_INVALID_KERNEL ";
        case CL_INVALID_ARG_INDEX :
            return (char*)" CL_INVALID_ARG_INDEX ";
        case CL_INVALID_ARG_VALUE :
            return (char*)" CL_INVALID_ARG_VALUE ";
        case CL_INVALID_ARG_SIZE :
            return (char*)" CL_INVALID_ARG_SIZE ";
        case CL_INVALID_KERNEL_ARGS :
            return (char*)" CL_INVALID_KERNEL_ARGS ";
        case CL_INVALID_WORK_DIMENSION :
            return (char*)" CL_INVALID_WORK_DIMENSION ";
        case CL_INVALID_WORK_GROUP_SIZE :
            return (char*)" CL_INVALID_WORK_GROUP_SIZE ";
        case CL_INVALID_WORK_ITEM_SIZE :
            return (char*)" CL_INVALID_WORK_ITEM_SIZE ";
        case CL_INVALID_GLOBAL_OFFSET :
            return (char*)" CL_INVALID_GLOBAL_OFFSET ";
        case CL_INVALID_EVENT_WAIT_LIST :
            return (char*)" CL_INVALID_EVENT_WAIT_LIST ";
        case CL_INVALID_EVENT :
            return (char*)" CL_INVALID_EVENT ";
        case CL_INVALID_OPERATION :
            return (char*)" CL_INVALID_OPERATION ";
        case CL_INVALID_GL_OBJECT :
            return (char*)" CL_INVALID_GL_OBJECT ";
        case CL_INVALID_BUFFER_SIZE :
            return (char*)" CL_INVALID_BUFFER_SIZE ";
        case CL_INVALID_MIP_LEVEL :
            return (char*)" CL_INVALID_MIP_LEVEL ";
        case CL_INVALID_GLOBAL_WORK_SIZE :
            return (char*)" CL_INVALID_GLOBAL_WORK_SIZE ";
        case CL_INVALID_PROPERTY :
            return (char*)" CL_INVALID_PROPERTY ";
        default:
            return (char*)"UNKNOWN ERROR";

    }
}

int main(void)
{
    cl_int err;
    // Find the number of OpenCL platforms
    cl_uint num_platforms;
    err = clGetPlatformIDs(0, NULL, &num_platforms);
    if (err != CL_SUCCESS || num_platforms < 0)
    {
        printf("Error: could not find a platform\n%s\n", err_code(err));
        return EXIT_FAILURE;
    }
    // Create a list of platform IDs
    cl_platform_id platform[num_platforms];
    err = clGetPlatformIDs(num_platforms, platform, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: could not get platforms\n%s\n", err_code(err));
    }

    printf("\nNumber of OpenCL platforms: %d\n", num_platforms);
    printf("\n-------------------------\n");

    // Investigate each platform
    for (int i = 0; i < num_platforms; i++)
    {
        cl_char string[10240] = {0};
        // Print out the platform name
        err = clGetPlatformInfo(platform[i], CL_PLATFORM_NAME, sizeof(string), &string, NULL);
        if (err != CL_SUCCESS)
        {
            printf("Error: could not get platform information\n%s\n", err_code(err));
            return EXIT_FAILURE;
        }
        printf("Platform: %s\n", string);

        // Print out the platform vendor
        err = clGetPlatformInfo(platform[i], CL_PLATFORM_VENDOR, sizeof(string), &string, NULL);
        if (err != CL_SUCCESS)
        {
            printf("Error: could not get platform information\n%s\n", err_code(err));
            return EXIT_FAILURE;
        }
        printf("Vendor: %s\n", string);

        // Print out the platform OpenCL version
        err = clGetPlatformInfo(platform[i], CL_PLATFORM_VERSION, sizeof(string), &string, NULL);
        if (err != CL_SUCCESS)
        {
            printf("Error: could not get platform information\n%s\n", err_code(err));
            return EXIT_FAILURE;
        }
        printf("Version: %s\n", string);

        // Count the number of devices in the platform
        cl_uint num_devices;
        err = clGetDeviceIDs(platform[i], CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices);
        if (err != CL_SUCCESS)
        {
            printf("Error: could not get devices for platform\n%s\n", err_code(err));
            return EXIT_FAILURE;
        }
        // Get the device IDs
        cl_device_id device[num_devices];
        err = clGetDeviceIDs(platform[i], CL_DEVICE_TYPE_ALL, num_devices, device, NULL);
        if (err != CL_SUCCESS)
        {
            printf("Error: could not get devices for platform\n%s\n", err_code(err));
            return EXIT_FAILURE;
        }
        printf("Number of devices: %d\n", num_devices);

        // Investigate each device
        for (int j = 0; j < num_devices; j++)
        {
            printf("\t-------------------------\n");

            // Get device name
            err = clGetDeviceInfo(device[j], CL_DEVICE_NAME, sizeof(string), &string, NULL);
            if (err != CL_SUCCESS)
            {
                printf("Error: could not get device information\n%s\n", err_code(err));
                return EXIT_FAILURE;
            }
            printf("\t\tName: %s\n", string);

            // Get device OpenCL version
            err = clGetDeviceInfo(device[j], CL_DEVICE_OPENCL_C_VERSION, sizeof(string), &string, NULL);
            if (err != CL_SUCCESS)
            {
                printf("Error: could not get device information\n%s\n", err_code(err));
                return EXIT_FAILURE;
            }
            printf("\t\tVersion: %s\n", string);

            // Get Max. Compute units
            cl_uint num;
            err = clGetDeviceInfo(device[j], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &num, NULL);
            if (err != CL_SUCCESS)
            {
                printf("Error: could not get device information\n%s\n", err_code(err));
                return EXIT_FAILURE;
            }
            printf("\t\tMax. Compute Units: %d\n", num);

            // Get local memory size
            cl_ulong mem_size;
            err = clGetDeviceInfo(device[j], CL_DEVICE_LOCAL_MEM_SIZE, sizeof(cl_ulong), &mem_size, NULL);
            if (err != CL_SUCCESS)
            {
                printf("Error: could not get device information\n%s\n", err_code(err));
                return EXIT_FAILURE;
            }
            printf("\t\tLocal Memory Size: %ld KB\n", mem_size/1024);

            // Get global memory size
            err = clGetDeviceInfo(device[j], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong), &mem_size, NULL);
            if (err != CL_SUCCESS)
            {
                printf("Error: could not get device information\n%s\n", err_code(err));
                return EXIT_FAILURE;
            }
            printf("\t\tGlobal Memory Size: %ld MB\n", mem_size/(1024*1024));

            // Get maximum buffer alloc. size
            err = clGetDeviceInfo(device[j], CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(cl_ulong), &mem_size, NULL);
            if (err != CL_SUCCESS)
            {
                printf("Error: could not get device information\n%s\n", err_code(err));
                return EXIT_FAILURE;
            }
            printf("\t\tMax Alloc Size: %ld MB\n", mem_size/(1024*1024));

            // Get work-group size information
            size_t size;
            err = clGetDeviceInfo(device[j], CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &size, NULL);
            if (err != CL_SUCCESS)
            {
                printf("Error: could not get device information\n%s\n", err_code(err));
                return EXIT_FAILURE;
            }
            printf("\t\tMax Work-group Size: %ld\n", size);

            // Find the maximum dimensions of the work-groups
            err = clGetDeviceInfo(device[j], CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(cl_uint), &num, NULL);
            if (err != CL_SUCCESS)
            {
                printf("Error: could not get device information\n%s\n", err_code(err));
                return EXIT_FAILURE;
            }
            // Get the max. dimensions of the work-groups
            size_t dims[num];
            err = clGetDeviceInfo(device[j], CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(dims), &dims, NULL);
            if (err != CL_SUCCESS)
            {
                printf("Error: could not get device information\n%s\n", err_code(err));
                return EXIT_FAILURE;
            }
            printf("\t\tMax Work-item Dims: ( ");
            for (size_t k = 0; k < num; k++)
            {
                printf("%ld ", dims[k]);
            }
            printf(")\n");

            printf("\t-------------------------\n");
        }

        printf("\n-------------------------\n");
    }

    return EXIT_SUCCESS;
}
