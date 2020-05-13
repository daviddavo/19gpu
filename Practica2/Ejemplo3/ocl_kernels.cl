// ocl_kernels.cl

#define BLOCKS_DIM 16
#define MAX_WINDOW_SIZE 5

void merge_sort(float arr[], int low, int mid, int high) { 

    int i,m,k,l;
    float temp[MAX_WINDOW_SIZE];

    l=low;
    i=low;
    m=mid+1;

    while((l<=mid)&&(m<=high)){

         if(arr[l]<=arr[m]){
             temp[i]=arr[l];
             l++;
         }
         else{
             temp[i]=arr[m];
             m++;
         }
         i++;
    }

    if(l>mid){
         for(k=m;k<=high;k++){
             temp[i]=arr[k];
             i++;
         }
    }
    else{
         for(k=l;k<=mid;k++){
             temp[i]=arr[k];
             i++;
         }
    }
   
    for(k=low;k<=high;k++){
         arr[k]=temp[k];
    }
}

void buble_sort(float array[], int size)
{
	int i, j;
	float tmp;

	for (i=1; i<size; i++)
		for (j=0 ; j<size - i; j++)
			if (array[j] > array[j+1]){
				tmp = array[j];
				array[j] = array[j+1];
				array[j+1] = tmp;
			}
}

// #define DEBUG_X 26
// #define DEBUG_Y 14

__kernel void ocl_naive(
    __global float * image_out,
    __global const float * im,
    const float thredshold,
    const uint window_size,
    const uint width,
    const uint height
) {
    float window[MAX_WINDOW_SIZE*MAX_WINDOW_SIZE];
    float median;
    unsigned i = get_global_id(0);
    unsigned j = get_global_id(1);
    unsigned ws2 = (window_size-1)/2;

    if (ws2 <= i && i < (width - ws2) && ws2 <= j && j < (height - ws2)) {
        for (int ii = 0; ii < window_size; ii++) {
            for (int jj = 0; jj < window_size; jj++) {
                window[jj*window_size+ii] = im[(j+jj-ws2)*width + i+ii-ws2];
#ifdef DEBUG_X
                if (i == DEBUG_X && j == DEBUG_Y) printf("%3.2f ", window[jj*window_size+ii]);
#endif
            }
#ifdef DEBUG_X
            if (i == DEBUG_X && j == DEBUG_Y) printf("\n");
#endif
        }
        
        buble_sort(window, window_size*window_size);
        median = window[(window_size*window_size-1)>>1];
#ifdef DEBUG_X
        if (i == DEBUG_X && j == DEBUG_Y) 
            printf("i: %d, j: %d, median: %3.2f, thr: %f, diff: %f, rel: %f\n",
            i, j, median, thredshold, median - im[j*width+i], fabs(median - im[j*width+i])/median);
#endif

        image_out[j*width+i] = (fabs((median - im[j*width+i])/median) > thredshold)?median:im[j*width+i];
    } else {
        // image_out[j*width+i] = im[j*width+i];
        image_out[j*width+i] = 0;
    }
}
