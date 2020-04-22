#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <assert.h>

#include "routinesGPU.h"
#include "routinesCPU.h"

#define TPBLKX 32
#define TPBLKY 24
#define TPBLK 768 // Theoretical limit of 768
#define PI 3.1415926535f

#define CUDACHECK(f) { \
    const cudaError_t error = f; \
    if (error != cudaSuccess) { \
        printf("Error: %s:%d, ", __FILE__, __LINE__); \
        printf("code: %d, reason: %s\n", error, cudaGetErrorString(error)); \
        exit(1); \
    } \
}

__global__ void cu_canny_nr_naive(uint8_t *im, float *NR, int height, int width) {
	unsigned k = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned i = k / width;
	unsigned j = k % width;

	if (k < height*width && i >= 2 && i < height-2 && j >= 2 && j < width-2) {
		NR[k] =
			 (2.0*im[(i-2)*width+(j-2)] +  4.0*im[(i-2)*width+(j-1)] +  5.0*im[(i-2)*width+(j)] +  4.0*im[(i-2)*width+(j+1)] + 2.0*im[(i-2)*width+(j+2)]
			+ 4.0*im[(i-1)*width+(j-2)] +  9.0*im[(i-1)*width+(j-1)] + 12.0*im[(i-1)*width+(j)] +  9.0*im[(i-1)*width+(j+1)] + 4.0*im[(i-1)*width+(j+2)]
			+ 5.0*im[(i  )*width+(j-2)] + 12.0*im[(i  )*width+(j-1)] + 15.0*im[(i  )*width+(j)] + 12.0*im[(i  )*width+(j+1)] + 5.0*im[(i  )*width+(j+2)]
			+ 4.0*im[(i+1)*width+(j-2)] +  9.0*im[(i+1)*width+(j-1)] + 12.0*im[(i+1)*width+(j)] +  9.0*im[(i+1)*width+(j+1)] + 4.0*im[(i+1)*width+(j+2)]
			+ 2.0*im[(i+2)*width+(j-2)] +  4.0*im[(i+2)*width+(j-1)] +  5.0*im[(i+2)*width+(j)] +  4.0*im[(i+2)*width+(j+1)] + 2.0*im[(i+2)*width+(j+2)])
			/159.0;
	}
}

/**
 * Este kernel está limitado por el elevado número de registros usados. ¿Mover la "matriz" de constantes a shared memory tal vez?
 */
__global__ void cu_canny_nr_shared(const uint8_t * im, float *NR, int height, int width) {
	unsigned i = blockIdx.x * blockDim.x + threadIdx.x + 2;
	unsigned j = blockIdx.y * blockDim.y + threadIdx.y + 2;
	unsigned k =  i * width + j;

	__shared__ uint8_t im_shared [TPBLKX+4][TPBLKY+4];
	im_shared[threadIdx.x+2][threadIdx.y+2] = im[k];

	// Los 4 bordes
	if (threadIdx.x < 2) {
		im_shared[threadIdx.x][threadIdx.y+2] = im[(i-2)*width + j];
	}
	if (threadIdx.y < 2) {
		im_shared[threadIdx.x+2][threadIdx.y] = im[(i)*width + j - 2];
	}
	if (threadIdx.x >= TPBLKX - 2) {
		im_shared[threadIdx.x+4][threadIdx.y+2] = im[(i+2)*width + j];
	}
	if (threadIdx.y >= TPBLKY - 2) {
		im_shared[threadIdx.x+2][threadIdx.y+4] = im[i*width + j + 2];
	}

	// Las 4 esquinas
	if (threadIdx.x < 2 && threadIdx.y < 2) {
		im_shared[threadIdx.x  ][threadIdx.y  ] = im[(i-2)*width + j - 2];
	}
	if (threadIdx.x < 2 && threadIdx.y >= TPBLKY - 2) {
		im_shared[threadIdx.x  ][threadIdx.y+4] = im[(i-2)*width + j + 2];
	}
	if (threadIdx.x >= TPBLKX - 2 && threadIdx.y < 2) {
		im_shared[threadIdx.x+4][threadIdx.y  ] = im[(i+2)*width + j - 2];
	}
	if (threadIdx.x >= TPBLKX - 2 && threadIdx.y >= TPBLKX - 2) {
		im_shared[threadIdx.x+4][threadIdx.y+4] = im[(i+2)*width + j + 2];
	}

	__syncthreads();

	// Quitamos (i*width + j) < height*width pues hacemos dicha comprobación
	// antes de invocar al kernel. Esto aumentó el non-predicted warp efficency
	if (i < height - 2 && j < width - 2) {
		/*
		for (int ii = 0; ii < 5; ii++) {
			for (int jj = 0; jj < 5; jj++) {
				assert(im_shared[threadIdx.x+ii][threadIdx.y+jj] == im[(i+ii-2)*width + j+jj-2]);
			}
		}
		*/

		NR[k] =
			( 2.0*im_shared[threadIdx.x  ][threadIdx.y] +  4.0*im_shared[threadIdx.x  ][threadIdx.y+1] +  5.0*im_shared[threadIdx.x  ][threadIdx.y+2] +  4.0*im_shared[threadIdx.x  ][threadIdx.y+3] + 2.0*im_shared[threadIdx.x  ][threadIdx.y+4]
			+ 4.0*im_shared[threadIdx.x+1][threadIdx.y] +  9.0*im_shared[threadIdx.x+1][threadIdx.y+1] + 12.0*im_shared[threadIdx.x+1][threadIdx.y+2] +  9.0*im_shared[threadIdx.x+1][threadIdx.y+3] + 4.0*im_shared[threadIdx.x+1][threadIdx.y+4]
			+ 5.0*im_shared[threadIdx.x+2][threadIdx.y] + 12.0*im_shared[threadIdx.x+2][threadIdx.y+1] + 15.0*im_shared[threadIdx.x+2][threadIdx.y+2] + 12.0*im_shared[threadIdx.x+2][threadIdx.y+3] + 5.0*im_shared[threadIdx.x+2][threadIdx.y+4]
			+ 4.0*im_shared[threadIdx.x+3][threadIdx.y] +  9.0*im_shared[threadIdx.x+3][threadIdx.y+1] + 12.0*im_shared[threadIdx.x+3][threadIdx.y+3] +  9.0*im_shared[threadIdx.x+3][threadIdx.y+4] + 4.0*im_shared[threadIdx.x+3][threadIdx.y+4]
			+ 2.0*im_shared[threadIdx.x+4][threadIdx.y] +  4.0*im_shared[threadIdx.x+4][threadIdx.y+1] +  5.0*im_shared[threadIdx.x+4][threadIdx.y+3] +  4.0*im_shared[threadIdx.x+4][threadIdx.y+4] + 2.0*im_shared[threadIdx.x+4][threadIdx.y+4])
			/159.0;
	}
}

__global__ void cu_canny_g(float * NR, float * Gx, float * Gy, float * G, float * phi, int height, int width) {
	unsigned k = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned i = k / width;
	unsigned j = k % width;

	if (k < height*width && i >= 2 && i < height-2 && j >= 2 && j < width-2) {
		Gx[k] =
			(1.0*NR[(i-2)*width+(j-2)] +  2.0*NR[(i-2)*width+(j-1)] +  (-2.0)*NR[(i-2)*width+(j+1)] + (-1.0)*NR[(i-2)*width+(j+2)]
			+ 4.0*NR[(i-1)*width+(j-2)] +  8.0*NR[(i-1)*width+(j-1)] +  (-8.0)*NR[(i-1)*width+(j+1)] + (-4.0)*NR[(i-1)*width+(j+2)]
			+ 6.0*NR[(i  )*width+(j-2)] + 12.0*NR[(i  )*width+(j-1)] + (-12.0)*NR[(i  )*width+(j+1)] + (-6.0)*NR[(i  )*width+(j+2)]
			+ 4.0*NR[(i+1)*width+(j-2)] +  8.0*NR[(i+1)*width+(j-1)] +  (-8.0)*NR[(i+1)*width+(j+1)] + (-4.0)*NR[(i+1)*width+(j+2)]
			+ 1.0*NR[(i+2)*width+(j-2)] +  2.0*NR[(i+2)*width+(j-1)] +  (-2.0)*NR[(i+2)*width+(j+1)] + (-1.0)*NR[(i+2)*width+(j+2)]);
		Gy[k] =
		 ((-1.0)*NR[(i-2)*width+(j-2)] + (-4.0)*NR[(i-2)*width+(j-1)] +  (-6.0)*NR[(i-2)*width+(j)] + (-4.0)*NR[(i-2)*width+(j+1)] + (-1.0)*NR[(i-2)*width+(j+2)]
		+ (-2.0)*NR[(i-1)*width+(j-2)] + (-8.0)*NR[(i-1)*width+(j-1)] + (-12.0)*NR[(i-1)*width+(j)] + (-8.0)*NR[(i-1)*width+(j+1)] + (-2.0)*NR[(i-1)*width+(j+2)]
		+    2.0*NR[(i+1)*width+(j-2)] +    8.0*NR[(i+1)*width+(j-1)] +    12.0*NR[(i+1)*width+(j)] +    8.0*NR[(i+1)*width+(j+1)] +    2.0*NR[(i+1)*width+(j+2)]
		+    1.0*NR[(i+2)*width+(j-2)] +    4.0*NR[(i+2)*width+(j-1)] +     6.0*NR[(i+2)*width+(j)] +    4.0*NR[(i+2)*width+(j+1)] +    1.0*NR[(i+2)*width+(j+2)]);

		G[k]   = sqrtf((Gx[k]*Gx[k])+(Gy[k]*Gy[k]));	//G = √Gx²+Gy²
		phi[k] = atan2f(fabs(Gy[k]),fabs(Gx[k]));

		if(fabs(phi[k])<=PI/8 )
			phi[k] = 0;
		else if (fabs(phi[k])<= 3*(PI/8))
			phi[k] = 45;
		else if (fabs(phi[k]) <= 5*(PI/8))
			phi[k] = 90;
		else if (fabs(phi[k]) <= 7*(PI/8))
			phi[k] = 135;
		else phi[k] = 0;
	}
}

__global__ void cu_canny_pedge(float * phi, float * G, uint8_t * pedge, int height, int width) {
	unsigned k = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned i = k / width;
	unsigned j = k % width;

	if (k < height*width && i >= 3 && i < height-3 && j >= 3 && j < width-3) {
		pedge[i*width+j] = 0;
		if(phi[i*width+j] == 0){
			if(G[i*width+j]>G[i*width+j+1] && G[i*width+j]>G[i*width+j-1]) //edge is in N-S
				pedge[i*width+j] = 1;

		} else if(phi[i*width+j] == 45) {
			if(G[i*width+j]>G[(i+1)*width+j+1] && G[i*width+j]>G[(i-1)*width+j-1]) // edge is in NW-SE
				pedge[i*width+j] = 1;

		} else if(phi[i*width+j] == 90) {
			if(G[i*width+j]>G[(i+1)*width+j] && G[i*width+j]>G[(i-1)*width+j]) //edge is in E-W
				pedge[i*width+j] = 1;

		} else if(phi[i*width+j] == 135) {
			if(G[i*width+j]>G[(i+1)*width+j-1] && G[i*width+j]>G[(i-1)*width+j+1]) // edge is in NE-SW
				pedge[i*width+j] = 1;
		}
	}
}

__global__ void cu_canny_hthr(uint8_t * out, float * G, uint8_t * pedge, int height, int width, float lt, float ht) {
	unsigned k = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned i = k / width;
	unsigned j = k % width;
	int ii, jj;

	if (k < height*width && i >= 3 && i < height-3 && j >= 3 && j < width-3) {
		out[i*width+j] = 0;
		if(G[i*width+j]>ht && pedge[i*width+j])
			out[i*width+j] = 255;
		else if(pedge[i*width+j] && G[i*width+j]>=lt && G[i*width+j]<ht)
			// check neighbours 3x3
			for (ii=-1;ii<=1; ii++)
				for (jj=-1;jj<=1; jj++)
					if (G[(i+ii)*width+j+jj]>ht)
						out[i*width+j] = 255;
	}
}

void cu_canny(uint8_t *im, uint8_t *image_out,
	float *NR, float *G, float *phi, float *Gx, float *Gy, uint8_t *pedge,
	float level,
	int height, int width)
{
	dim3 blockDim(TPBLKX, TPBLKY);
	// dim3 gridDim((height*width+TPBLK - 1)/TPBLK);
	// Nota: El "-2" es por los dos píxeles del borde de abajo a la derecha que no procesaremos
	dim3 gridDim((height + TPBLKX - 1 - 4)/TPBLKX, (width + TPBLKY - 1 - 4)/TPBLKY);

	uint8_t * d_im_in, * d_im_out, * d_pedge;
	float * d_NR, *d_Gx, *d_Gy, *d_G, *d_phi;
	CUDACHECK(cudaMalloc(&d_im_in, sizeof(uint8_t)*height*width));



	CUDACHECK(cudaMemcpy(d_im_in, im, sizeof(uint8_t)*height*width, cudaMemcpyHostToDevice));

	// TODO: Noise reduction
	printf("Running kernels<<<(%d, %d), (%d, %d)>>> on image size %dx%d\n",
			gridDim.x, gridDim.y, blockDim.x, blockDim.y, width, height);

	CUDACHECK(cudaMalloc(&d_NR, sizeof(float)*height*width));
	cu_canny_nr_shared<<<gridDim, blockDim>>>(d_im_in, d_NR, height, width);
	cu_canny_nr_naive<<<(height*width+TPBLK-1)/TPBLK, TPBLK>>>(d_im_in, d_NR, height, width);
	CUDACHECK(cudaMemcpy(NR, d_NR, sizeof(float)*height*width, cudaMemcpyDeviceToHost));

	gridDim = dim3((height*width+TPBLK-1)/TPBLK);
	blockDim = dim3(TPBLK);

	CUDACHECK(cudaMalloc(&d_Gx, sizeof(float)*height*width));
	CUDACHECK(cudaMalloc(&d_Gy, sizeof(float)*height*width));
	CUDACHECK(cudaMalloc(&d_G, sizeof(float)*height*width));
	CUDACHECK(cudaMalloc(&d_phi, sizeof(float)*height*width));
	cu_canny_g<<<gridDim, blockDim>>>(d_NR, d_Gx, d_Gy, d_G, d_phi, height, width);
	CUDACHECK(cudaMemcpy(Gx, d_Gx, sizeof(float)*height*width, cudaMemcpyDeviceToHost));
	CUDACHECK(cudaMemcpy(Gy, d_Gy, sizeof(float)*height*width, cudaMemcpyDeviceToHost));
	CUDACHECK(cudaMemcpy(G, d_G, sizeof(float)*height*width, cudaMemcpyDeviceToHost));
	CUDACHECK(cudaMemcpy(phi, d_phi, sizeof(float)*height*width, cudaMemcpyDeviceToHost));
	CUDACHECK(cudaFree(d_Gx));
	CUDACHECK(cudaFree(d_Gy));
	CUDACHECK(cudaFree(d_NR));

	CUDACHECK(cudaMalloc(&d_pedge, sizeof(uint8_t)*height*width));
	cu_canny_pedge<<<gridDim, blockDim>>>(d_phi, d_G, d_pedge, height, width);
	CUDACHECK(cudaMemcpy(pedge, d_pedge, sizeof(uint8_t)*height*width, cudaMemcpyDeviceToHost));

	CUDACHECK(cudaMalloc(&d_im_out, sizeof(uint8_t)*height*width));
	cu_canny_hthr<<<gridDim, blockDim>>>(d_im_out, d_G, d_pedge, height, width, level/2, level*2);
	CUDACHECK(cudaMemcpy(image_out, d_im_out, sizeof(uint8_t)*height*width, cudaMemcpyDeviceToHost));
	CUDACHECK(cudaFree(d_G));
	CUDACHECK(cudaFree(d_pedge));
	CUDACHECK(cudaFree(d_im_in));
	CUDACHECK(cudaFree(d_im_out));
}

__global__ void cu_htkernel(const uint8_t *im, int width, int height, int accu_width, int accu_height,
		uint32_t * accumulators)
{
	unsigned k = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned i = k / width;
	unsigned j = k % width;
	float center_x = width/2.0;
	float center_y = height/2.0;
	float hough_h = ((sqrt(2.0) * (float)(height>width?height:width)) / 2.0);
	float cos, sin;
	int theta;

	if( im[k] > 250 ) // Pixel is edge
	{
		for(theta=0;theta<180;theta++)
		{
			sincos(theta, &sin, &cos);
			float rho = ( ((float)j - center_x) * cos) + (((float)i - center_y) * sin);
			atomicAdd(&accumulators[ (int)((round(rho + hough_h) * 180.0)) + theta], 1);
		}
	}
}

void cu_houghtransform(uint8_t *im, int width, int height, uint32_t *accumulators, int accu_width, int accu_height,
	float *sin_table, float *cos_table)
{
	uint8_t * d_im;
	uint32_t * d_accumulators;
	float * d_sin_table, *d_cos_table;

	dim3 gridDim((height*width+1)/TPBLK);
	dim3 blockDim(TPBLK);

	// TODO: Remove sin/cos table and calculate on GPU
	CUDACHECK(cudaMalloc(&d_im, sizeof(uint8_t)*width*height));
	CUDACHECK(cudaMalloc(&d_accumulators, sizeof(uint32_t)*accu_width*accu_height));
	// CUDACHECK(cudaMalloc(&d_sin_table, sizeof(float)*180));
	// CUDACHECK(cudaMalloc(&d_cos_table, sizeof(float)*180));
	CUDACHECK(cudaMemcpy(d_im, im, sizeof(uint8_t)*width*height, cudaMemcpyHostToDevice));
	CUDACHECK(cudaMemset(d_accumulators, 0, sizeof(uint32_t)*accu_width*accu_height));
	// CUDACHECK(cudaMemcpy(d_sin_table, sin_table, sizeof(float)*180, cudaMemcpyHostToDevice));
	// CUDACHECK(cudaMemcpy(d_cos_table, cos_table, sizeof(float)*180, cudaMemcpyHostToDevice));
	cu_htkernel<<<gridDim, blockDim>>>(d_im, width, height, accu_width, accu_height,
		// d_sin_table, d_cos_table,
		d_accumulators);
	CUDACHECK(cudaMemcpy(accumulators, d_accumulators, sizeof(uint32_t)*accu_width*accu_height, cudaMemcpyDeviceToHost));
	CUDACHECK(cudaFree(d_im));
	CUDACHECK(cudaFree(d_accumulators));
	// CUDACHECK(cudaFree(d_sin_table));
	// CUDACHECK(cudaFree(d_cos_table));
}

__global__ void cu_glkernel(int threshold, uint32_t * accumulators, int accu_width, int accu_height, int width, int height,
	int *x1_lines, int *y1_lines, int *x2_lines, int *y2_lines, int *lines)
{
	unsigned k = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned rho = k / accu_width;
	unsigned theta = k % accu_width;
	uint32_t max;
	float sin, cos;

	if(accumulators[(rho*accu_width) + theta] >= threshold)
	{
		//Is this point a local maxima (9x9)
		max = accumulators[(rho*accu_width) + theta];
		for(int ii=-4;ii<=4;ii++)
		{
			for(int jj=-4;jj<=4;jj++)
			{
				if( (ii+rho>=0 && ii+rho<accu_height) && (jj+theta>=0 && jj+theta<accu_width) )
				{
					if( accumulators[((rho+ii)*accu_width) + (theta+jj)] > max )
					{
						max = accumulators[((rho+ii)*accu_width) + (theta+jj)];
					}
				}
			}
		}

		if(max == accumulators[(rho*accu_width) + theta]) //local maxima
		{
			int x1, y1, x2, y2;
			x1 = y1 = x2 = y2 = 0;

			sincos(theta, &sin, &cos);
			if(theta >= 45 && theta <= 135)
			{
				if (theta>90) {
					//y = (r - x cos(t)) / sin(t)
					x1 = width/2;
					y1 = ((float)(rho-(accu_height/2)) - ((x1 - (width/2) ) * cos)) / sin + (height / 2);
					x2 = width;
					y2 = ((float)(rho-(accu_height/2)) - ((x2 - (width/2) ) * cos)) / sin + (height / 2);
				} else {
					//y = (r - x cos(t)) / sin(t)
					x1 = 0;
					y1 = ((float)(rho-(accu_height/2)) - ((x1 - (width/2) ) * cos)) / sin + (height / 2);
					x2 = width*2/5;
					y2 = ((float)(rho-(accu_height/2)) - ((x2 - (width/2) ) * cos)) / sin + (height / 2);
				}
			} else {
				//x = (r - y sin(t)) / cos(t);
				y1 = 0;
				x1 = ((float)(rho-(accu_height/2)) - ((y1 - (height/2) ) * sin)) / cos + (width / 2);
				y2 = height;
				x2 = ((float)(rho-(accu_height/2)) - ((y2 - (height/2) ) * sin)) / cos + (width / 2);
			}
			x1_lines[*lines] = x1;
			y1_lines[*lines] = y1;
			x2_lines[*lines] = x2;
			y2_lines[*lines] = y2;
			(*lines)++;
		}
	}
}

void cu_getlines(int threshold, uint32_t *accumulators, int accu_width, int accu_height, int width, int height,
	float *sin_table, float *cos_table,
	int *x1_lines, int *y1_lines, int *x2_lines, int *y2_lines, int *lines)
{
	const int maxnlines = 10;

	uint32_t * d_accumulators;
	int * d_x1_lines, *d_x2_lines, *d_y1_lines, *d_y2_lines;
	int * d_lines;

	dim3 gridDim((accu_height*accu_width+1)/TPBLK);
	dim3 blockDim(TPBLK);

	CUDACHECK(cudaMalloc(&d_accumulators, sizeof(uint32_t)*accu_width*accu_height));
	CUDACHECK(cudaMalloc(&d_x1_lines, sizeof(int)*maxnlines));
	CUDACHECK(cudaMalloc(&d_y1_lines, sizeof(int)*maxnlines));
	CUDACHECK(cudaMalloc(&d_x2_lines, sizeof(int)*maxnlines));
	CUDACHECK(cudaMalloc(&d_y2_lines, sizeof(int)*maxnlines));
	CUDACHECK(cudaMalloc(&d_lines, sizeof(int)));
	CUDACHECK(cudaMemcpy(&d_accumulators, accumulators, sizeof(uint32_t)*accu_width*accu_height, cudaMemcpyHostToDevice));
	cu_glkernel<<<gridDim, blockDim>>>(threshold, d_accumulators, accu_width, accu_height, width, height,
		d_x1_lines, d_y1_lines, d_x2_lines, d_y2_lines, d_lines);
	CUDACHECK(cudaMemcpy(x1_lines, d_x1_lines, sizeof(int)*maxnlines, cudaMemcpyDeviceToHost));
	CUDACHECK(cudaMemcpy(y1_lines, d_y1_lines, sizeof(int)*maxnlines, cudaMemcpyDeviceToHost));
	CUDACHECK(cudaMemcpy(x2_lines, d_x2_lines, sizeof(int)*maxnlines, cudaMemcpyDeviceToHost));
	CUDACHECK(cudaMemcpy(y2_lines, d_y2_lines, sizeof(int)*maxnlines, cudaMemcpyDeviceToHost));
	CUDACHECK(cudaMemcpy(lines, d_lines, sizeof(int), cudaMemcpyDeviceToHost));

	CUDACHECK(cudaFree(d_accumulators));
	CUDACHECK(cudaFree(d_lines));
	CUDACHECK(cudaFree(x1_lines));
	CUDACHECK(cudaFree(y1_lines));
	CUDACHECK(cudaFree(x2_lines));
	CUDACHECK(cudaFree(y2_lines));
}


void line_asist_GPU(uint8_t *im, int height, int width,
	uint8_t *imEdge, float *NR, float *G, float *phi, float *Gx, float *Gy, uint8_t *pedge,
	float *sin_table, float *cos_table, 
	uint32_t *accum, int accu_height, int accu_width,
	int *x1, int *x2, int *y1, int *y2, int *nlines)
{
	int threshold;

	// TODO: Usar constant memory
	/* Canny */
	cu_canny(im, imEdge,
		NR, G, phi, Gx, Gy, pedge,
		1000.0f, //level
		height, width);

	/* hough transform */
	// TODO: Hough Transform

	houghtransform(imEdge, width, height, accum, accu_width, accu_height, sin_table, cos_table);

	if (width>height) threshold = width/6;
	else threshold = height/6;


	getlines(threshold, accum, accu_width, accu_height, width, height,
		sin_table, cos_table,
		x1, y1, x2, y2, nlines);
}
