#ifndef _OCL_H

#define _OCL_H

int remove_noiseOCL(float *im, float *image_out, 
	float thredshold, int window_size,
	int height, int width);
#endif
