// transpose_kernel.cl
// Kernel source file for calculating the transpose of a matrix

__kernel
void matrixTranspose(__global float * output,
                     __global const float * input,
                     const    uint    width)

{
    int i = get_global_id(0);
    int j = get_global_id(1);
    output[j*width + i] = input[i*width + j];
}


__kernel
void matrixTransposeLocal(__global float * output,
                          __global float * input,
                          __local float * tile,
                          const    uint    width)

{

}
