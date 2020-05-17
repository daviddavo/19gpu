
__kernel void pi_naive(
    __global float * totalarea
) {
    double x;
    __local float area;
    __local float arr_area[BLOCK_DIM];
    size_t n = get_global_size(0);
    int n2 = BLOCK_DIM;
    // printf("i: %d, bdim: %d, lsize: %d, gsize: %d, total: %d\n", get_global_id(0), BLOCK_DIM, get_local_size(0), get_global_size(0), sizeof(totalarea));

    if (get_local_id(0) == 0) {
        area = 0;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    x = (get_global_id(0) + 0.5)/n;
    arr_area[get_local_id(0)] = 4.0/(1.0 + x*x); 

    if (get_local_id(0) == 0) {
        // TODO: Unfold && reduce
        for (int j = 1; j < get_local_size(0); j++) {
            arr_area[0] += arr_area[j];
        }

        totalarea[get_group_id(0)] = arr_area[0];
    }
}
