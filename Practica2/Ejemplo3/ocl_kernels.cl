// ocl_kernels.cl

#define BLOCKS_DIM 16

__kernel void ocl_naive(
    __global float * image_out,
    __global const float * im,
    const uint width,
    const uint height
) {
    unsigned i = get_global_id(0);
    unsigned j = get_global_id(1);

    image_out[j*width + i] = im[j*width + i];
}
