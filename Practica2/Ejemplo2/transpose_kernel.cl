// transpose_kernel.cl
// Kernel source file for calculating the transpose of a matrix

#define BLOCK_DIM 16

__kernel
void matrixTranspose(__global float * output,
                     __global const float * input,
                     const    uint    width)

{
    int i = get_global_id(0);
    int j = get_global_id(1);
    output[i*width + j] = input[j*width + i];
}


__kernel
void matrixTransposeLocal(__global float * output,
                          __global const float * input,
                          __local float * tile,
                          const    uint    width)

{
    unsigned i = get_global_id(0);
    unsigned j = get_global_id(1);
    
    unsigned k_in  = j * width + i;
    unsigned k_out = get_local_id(1) * BLOCK_DIM + get_local_id(0);

    if (i < width && j < width)
        tile[k_out] = input[k_in];

    barrier(CLK_LOCAL_MEM_FENCE);

    // if (get_global_id(0) == 0 && get_global_id(1) == 0) {
    //     for (int ii = 0; ii < 16; ii++) {
    //         for (int jj = 0; jj < 16; jj++) {
    //             printf("%3.1f ", tile[ii*width + jj]);
    //         }
    //         printf("\n");
    //     }
    // }

    i = get_group_id(1) * BLOCK_DIM + get_local_id(0);
    j = get_group_id(0) * BLOCK_DIM + get_local_id(1);
    
    k_in  = get_local_id(0) * BLOCK_DIM + get_local_id(1);
    k_out = j * width + i;

    output[k_out] = tile[k_in];
}
