#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <assert.h>

#include "routinesGPU.h"
#include "routinesCPU.h"

#define TPBLKX 32
#define TPBLKY 32
#define TPBLK 1024
#define DEG2RAD 0.017453f
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
	unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned j = blockIdx.y * blockDim.y + threadIdx.y;
	// unsigned k =  i*width + j;
	unsigned k = j*width + i;

	__shared__ uint8_t im_shared [TPBLKY+4][TPBLKX+4];

	/*
	if (threadIdx.y == 0 && threadIdx.x == 0) {
		for (int ii = 0; ii < TPBLKX+4; ii++) {
			for (int jj = 0; jj < TPBLKY+4; jj++) {
				im_shared[jj][ii] = 0;
			}
		}
	}
	__syncthreads();
	*/

	// assert(im_shared[threadIdx.y][threadIdx.x] == 0);
	im_shared[threadIdx.y][threadIdx.x] = im[k];
	// assert(im_shared[threadIdx.y][threadIdx.x] == im[k]);

	if (threadIdx.y >= TPBLKY - 4) {
		// assert(im_shared[threadIdx.y+4][threadIdx.x  ] == 0);
		im_shared[threadIdx.y+4][threadIdx.x  ] = im[(j+4)*width + i  ];
		// assert(im_shared[threadIdx.y+4][threadIdx.x  ] == im[(j+4)*width + i  ]);
	}

	if (threadIdx.x >= TPBLKX - 4) {
		// assert(im_shared[threadIdx.y][threadIdx.x+4] == 0);
		im_shared[threadIdx.y  ][threadIdx.x+4] = im[(j  )*width + i+4];
		// assert(im_shared[threadIdx.y  ][threadIdx.x+4] == im[(j  )*width + i+4]);
	}

	if (threadIdx.y >= TPBLKY - 4 && threadIdx.x >= TPBLKX - 4) {
		// assert(im_shared[threadIdx.y+4][threadIdx.x+4] == 0);
		im_shared[threadIdx.y+4][threadIdx.x+4] = im[(j+4)*width + i+4];
		// assert(im_shared[threadIdx.y+4][threadIdx.x+4] == im[(j+4)*width + i+4]);
	}

	__syncthreads();

	if (j < height - 4 && i < width - 4) {

#ifdef _VERIFY
		for (int ii = 0; ii < 5; ii++) {
			for (int jj = 0; jj < 5; jj++) {
				assert(im_shared[threadIdx.y+jj][threadIdx.x+ii] == im[(j+jj)*width + i+ii]);
			}
		}
#endif

		NR[(j+2)*width + i+2] =
			( 2.0*im_shared[threadIdx.y  ][threadIdx.x] +  4.0*im_shared[threadIdx.y  ][threadIdx.x+1] +  5.0*im_shared[threadIdx.y  ][threadIdx.x+2] +  4.0*im_shared[threadIdx.y  ][threadIdx.x+3] + 2.0*im_shared[threadIdx.y  ][threadIdx.x+4]
			+ 4.0*im_shared[threadIdx.y+1][threadIdx.x] +  9.0*im_shared[threadIdx.y+1][threadIdx.x+1] + 12.0*im_shared[threadIdx.y+1][threadIdx.x+2] +  9.0*im_shared[threadIdx.y+1][threadIdx.x+3] + 4.0*im_shared[threadIdx.y+1][threadIdx.x+4]
			+ 5.0*im_shared[threadIdx.y+2][threadIdx.x] + 12.0*im_shared[threadIdx.y+2][threadIdx.x+1] + 15.0*im_shared[threadIdx.y+2][threadIdx.x+2] + 12.0*im_shared[threadIdx.y+2][threadIdx.x+3] + 5.0*im_shared[threadIdx.y+2][threadIdx.x+4]
			+ 4.0*im_shared[threadIdx.y+3][threadIdx.x] +  9.0*im_shared[threadIdx.y+3][threadIdx.x+1] + 12.0*im_shared[threadIdx.y+3][threadIdx.x+2] +  9.0*im_shared[threadIdx.y+3][threadIdx.x+3] + 4.0*im_shared[threadIdx.y+3][threadIdx.x+4]
			+ 2.0*im_shared[threadIdx.y+4][threadIdx.x] +  4.0*im_shared[threadIdx.y+4][threadIdx.x+1] +  5.0*im_shared[threadIdx.y+4][threadIdx.x+2] +  4.0*im_shared[threadIdx.y+4][threadIdx.x+3] + 2.0*im_shared[threadIdx.y+4][threadIdx.x+4])
			/159.0;
	}
}

/*
 * Idea de este kernel: Que cada kernel haga el cálculo de N píxeles (2, por ejemplo),
 * aumentando el uso de memoria compartida por kernel, y disminuyendo el porcentaje
 * del área superpuesta que copiamos entre los kernels
 * Para un kernel que computa N*M píxeles, el pct de los píxeles que escribimos es
 * (N*M)/((N+4)*(M+4))
 * 32x32:		79.01%	1 ppl (píxel por kernel)
 * 32x64:		83.66%	2 ppk
 * 64x64:		88.58%  2x2 ppk
 * 128x128:		94.03%  4x4 ppk
 */
__global__ void cu_canny_nr_shared2(const uint8_t * im, float *NR, int height, int width) {
	unsigned i = blockIdx.x * blockDim.x*4 + threadIdx.x;
	unsigned j = blockIdx.y * blockDim.y + threadIdx.y;
	// unsigned k =  i*width + j;
	unsigned k = j*width + i;

	__shared__ volatile uint8_t im_shared [TPBLKY+4][TPBLKX*4+4];

	#pragma unroll
	for (int ii = 0; ii < 4*blockDim.x; ii += blockDim.x) {
		im_shared[threadIdx.y][threadIdx.x+ii] = im[k+ii];
	}

	if (threadIdx.y >= TPBLKY - 4) {
		#pragma unroll
		for (int ii = 0; ii < 4*blockDim.x; ii += blockDim.x) {
			im_shared[threadIdx.y+4][threadIdx.x+ii] = im[(j+4)*width + i + ii];
		}
	}

	if (threadIdx.x+3*blockDim.x >= TPBLKX*4 - 4) {
		im_shared[threadIdx.y  ][threadIdx.x+4+3*blockDim.x] = im[(j  )*width + i+4+3*blockDim.x];
	}

	if (threadIdx.y >= TPBLKY - 4 && threadIdx.x+3*blockDim.x >= TPBLKX*4 - 4) {
		#pragma unroll
		for (int ii = 0; ii < 4*blockDim.x; ii += blockDim.x) {
			im_shared[threadIdx.y+4][threadIdx.x+4+ii] = im[(j+4)*width + i + 4 + ii];
		}
	}

	__syncthreads();

	if (j < height - 4) {
		for (int ii = 0; ii < 4*blockDim.x && i + ii < width - 4; ii += blockDim.x) {
			NR[((j+2)*width + i + 2 + ii)] =
				( 2.0*im_shared[threadIdx.y  ][threadIdx.x+ii] +  4.0*im_shared[threadIdx.y  ][threadIdx.x+ii+1] +  5.0*im_shared[threadIdx.y  ][threadIdx.x+ii+2] +  4.0*im_shared[threadIdx.y  ][threadIdx.x+ii+3] + 2.0*im_shared[threadIdx.y  ][threadIdx.x+ii+4]
				+ 4.0*im_shared[threadIdx.y+1][threadIdx.x+ii] +  9.0*im_shared[threadIdx.y+1][threadIdx.x+ii+1] + 12.0*im_shared[threadIdx.y+1][threadIdx.x+ii+2] +  9.0*im_shared[threadIdx.y+1][threadIdx.x+ii+3] + 4.0*im_shared[threadIdx.y+1][threadIdx.x+ii+4]
				+ 5.0*im_shared[threadIdx.y+2][threadIdx.x+ii] + 12.0*im_shared[threadIdx.y+2][threadIdx.x+ii+1] + 15.0*im_shared[threadIdx.y+2][threadIdx.x+ii+2] + 12.0*im_shared[threadIdx.y+2][threadIdx.x+ii+3] + 5.0*im_shared[threadIdx.y+2][threadIdx.x+ii+4]
				+ 4.0*im_shared[threadIdx.y+3][threadIdx.x+ii] +  9.0*im_shared[threadIdx.y+3][threadIdx.x+ii+1] + 12.0*im_shared[threadIdx.y+3][threadIdx.x+ii+2] +  9.0*im_shared[threadIdx.y+3][threadIdx.x+ii+3] + 4.0*im_shared[threadIdx.y+3][threadIdx.x+ii+4]
				+ 2.0*im_shared[threadIdx.y+4][threadIdx.x+ii] +  4.0*im_shared[threadIdx.y+4][threadIdx.x+ii+1] +  5.0*im_shared[threadIdx.y+4][threadIdx.x+ii+2] +  4.0*im_shared[threadIdx.y+4][threadIdx.x+ii+3] + 2.0*im_shared[threadIdx.y+4][threadIdx.x+ii+4])
				/159.0;
		}
	}
}

__global__ void cu_canny_g(float * NR, float * G, float * phi, int height, int width) {
	unsigned k = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned i = k / width;
	unsigned j = k % width;
	float gl[2];
	float p;

	if (i >= 2 && i < height-2 && j >= 2 && j < width-2) {
		gl[0] =	// Gx
			(1.0*NR[(i-2)*width+(j-2)] +  2.0*NR[(i-2)*width+(j-1)] +  (-2.0)*NR[(i-2)*width+(j+1)] + (-1.0)*NR[(i-2)*width+(j+2)]
			+ 4.0*NR[(i-1)*width+(j-2)] +  8.0*NR[(i-1)*width+(j-1)] +  (-8.0)*NR[(i-1)*width+(j+1)] + (-4.0)*NR[(i-1)*width+(j+2)]
			+ 6.0*NR[(i  )*width+(j-2)] + 12.0*NR[(i  )*width+(j-1)] + (-12.0)*NR[(i  )*width+(j+1)] + (-6.0)*NR[(i  )*width+(j+2)]
			+ 4.0*NR[(i+1)*width+(j-2)] +  8.0*NR[(i+1)*width+(j-1)] +  (-8.0)*NR[(i+1)*width+(j+1)] + (-4.0)*NR[(i+1)*width+(j+2)]
			+ 1.0*NR[(i+2)*width+(j-2)] +  2.0*NR[(i+2)*width+(j-1)] +  (-2.0)*NR[(i+2)*width+(j+1)] + (-1.0)*NR[(i+2)*width+(j+2)]);
		gl[1] =	// Gy
		 ((-1.0)*NR[(i-2)*width+(j-2)] + (-4.0)*NR[(i-2)*width+(j-1)] +  (-6.0)*NR[(i-2)*width+(j)] + (-4.0)*NR[(i-2)*width+(j+1)] + (-1.0)*NR[(i-2)*width+(j+2)]
		+ (-2.0)*NR[(i-1)*width+(j-2)] + (-8.0)*NR[(i-1)*width+(j-1)] + (-12.0)*NR[(i-1)*width+(j)] + (-8.0)*NR[(i-1)*width+(j+1)] + (-2.0)*NR[(i-1)*width+(j+2)]
		+    2.0*NR[(i+1)*width+(j-2)] +    8.0*NR[(i+1)*width+(j-1)] +    12.0*NR[(i+1)*width+(j)] +    8.0*NR[(i+1)*width+(j+1)] +    2.0*NR[(i+1)*width+(j+2)]
		+    1.0*NR[(i+2)*width+(j-2)] +    4.0*NR[(i+2)*width+(j-1)] +     6.0*NR[(i+2)*width+(j)] +    4.0*NR[(i+2)*width+(j+1)] +    1.0*NR[(i+2)*width+(j+2)]);

		G[k]   = normf(2, gl);	//G = √Gx²+Gy²
		p = fabs(atan2f(fabs(gl[1]),fabs(gl[0])));

		if(p<=PI/8 )
			phi[k] = 0;
		else if (p<= 3*(PI/8))
			phi[k] = 45;
		else if (p <= 5*(PI/8))
			phi[k] = 90;
		else if (p <= 7*(PI/8))
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

__global__ void cu_htkernel(const uint8_t *im, const int width, const int height, const int accu_width, const int accu_height,
		const float hough_h, const float *sin_table, const float* cos_table, uint32_t * accumulators)
{
	unsigned k = blockIdx.x * blockDim.x + threadIdx.x;
	float icenter = (float)(k/width) - height/2.0;
	float jcenter = (float)(k%width) - width/2.0;
	int theta;

	if( k < width*height && im[k] > 250 ) // Pixel is edge
	{
		for(theta=0;theta<180;theta++)
		{
			float rho = ( (jcenter * cos_table[theta]) + (icenter * sin_table[theta]));
			atomicAdd(&accumulators[ (int)((round(rho + hough_h)) * 180.0) + theta], 1);
		}
	}
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

			// TODO: Solucionar race condition
			x1_lines[*lines/* +s */] = x1;
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

	dim3 gridDim((accu_height*accu_width+TPBLK-1)/TPBLK);
	dim3 blockDim(TPBLK);

	CUDACHECK(cudaMalloc(&d_accumulators, sizeof(uint32_t)*accu_width*accu_height));
	CUDACHECK(cudaMalloc(&d_x1_lines, sizeof(int)*maxnlines));
	CUDACHECK(cudaMalloc(&d_y1_lines, sizeof(int)*maxnlines));
	CUDACHECK(cudaMalloc(&d_x2_lines, sizeof(int)*maxnlines));
	CUDACHECK(cudaMalloc(&d_y2_lines, sizeof(int)*maxnlines));
	CUDACHECK(cudaMalloc(&d_lines, sizeof(int)));
	CUDACHECK(cudaMemcpy(d_accumulators, accumulators, sizeof(uint32_t)*accu_width*accu_height, cudaMemcpyHostToDevice));
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

void cu_alloc(uint8_t ** imEdge, float **NR, float **G, float **phi, float **Gx,
		float **Gy, uint8_t **pedge, uint32_t **accum, int width, int height,
		int accu_height, int accu_width)
{
	CUDACHECK(cudaMallocHost(imEdge, sizeof(uint8_t) * width * height));
	CUDACHECK(cudaMallocHost(NR, sizeof(float) * width * height));
	CUDACHECK(cudaMallocHost(G, sizeof(float) * width * height));
	CUDACHECK(cudaMallocHost(phi, sizeof(float) * width * height));
	*Gx = *Gy = NULL;
	CUDACHECK(cudaMallocHost(pedge, sizeof(uint8_t) * width * height));

	CUDACHECK(cudaMallocHost(accum, sizeof(uint32_t) * accu_width * accu_height));
}

void line_asist_GPU(uint8_t *im, int height, int width,
	uint8_t *imEdge, float *NR, float *G, float *phi, float *Gx, float *Gy, uint8_t *pedge,
	float *sin_table, float *cos_table, 
	uint32_t *accum, int accu_height, int accu_width,
	int *x1, int *x2, int *y1, int *y2, int *nlines)
{
	int threshold;
	float level = 1000.0f;

	/*   ____    _    _   _ _   ___   __
  	 *  / ___|  / \  | \ | | \ | \ \ / /
 	 * | |     / _ \ |  \| |  \| |\ V /
 	 * | |___ / ___ \| |\  | |\  | | |
  	 *  \____/_/   \_\_| \_|_| \_| |_|
	 */
	dim3 blockDim2d(TPBLKX, TPBLKY);
	// dim3 gridDim((height*width+TPBLK - 1)/TPBLK);
	// Nota: El "-2" es por los dos píxeles del borde de abajo a la derecha que no procesaremos
	dim3 gridDim2d((width + TPBLKX - 1)/TPBLKX, (height + TPBLKY - 1)/TPBLKY);
	dim3 blockDim1d(TPBLK);
	dim3 gridDim1d((width*height+TPBLK-1)/TPBLK);

	cudaStream_t writeBackStream, houghStream;
	CUDACHECK(cudaStreamCreateWithFlags(&writeBackStream, cudaStreamNonBlocking));
	CUDACHECK(cudaStreamCreateWithFlags(&houghStream, cudaStreamNonBlocking));

	uint8_t * d_im_in, * d_im_out, * d_pedge;
	float * d_NR, *d_G, *d_phi;

	uint32_t * d_accumulators;
	float * d_sin_table, *d_cos_table;

	CUDACHECK(cudaMalloc(&d_im_in, sizeof(uint8_t)*height*width));
	CUDACHECK(cudaMalloc(&d_NR, sizeof(float)*height*width));
	CUDACHECK(cudaMalloc(&d_G, sizeof(float)*height*width));
	CUDACHECK(cudaMalloc(&d_phi, sizeof(float)*height*width));
	CUDACHECK(cudaMalloc(&d_pedge, sizeof(uint8_t)*height*width));
	CUDACHECK(cudaMalloc(&d_im_out, sizeof(uint8_t)*height*width));

	CUDACHECK(cudaMalloc(&d_accumulators, sizeof(uint32_t)*accu_width*accu_height));
	CUDACHECK(cudaMalloc(&d_sin_table, sizeof(float)*180));
	CUDACHECK(cudaMalloc(&d_cos_table, sizeof(float)*180));

	CUDACHECK(cudaMemcpyAsync(d_im_in, im, sizeof(uint8_t)*height*width, cudaMemcpyHostToDevice));

	printf("Running kernels<<<(%d, %d), (%d, %d)>>> on image size %dx%d\n",
			gridDim2d.x, gridDim2d.y, blockDim2d.x, blockDim2d.y, width, height);

	cu_canny_nr_shared<<<gridDim2d, blockDim2d, 0>>>(d_im_in, d_NR, height, width);
	// cu_canny_nr_shared2<<<dim3(gridDim.x/4, gridDim.y), blockDim>>>(d_im_in, d_NR, height, width);
	// cu_canny_nr_naive<<<(height*width+TPBLK-1)/TPBLK, TPBLK>>>(d_im_in, d_NR, height, width);

	// Vamos a aprovechar lo que tarda el canny para copiar cosas que necesitaremos para la transofrmada de hough
	CUDACHECK(cudaMemcpyAsync(d_sin_table, sin_table, sizeof(float)*180, cudaMemcpyHostToDevice, houghStream));
	CUDACHECK(cudaMemcpyAsync(d_cos_table, cos_table, sizeof(float)*180, cudaMemcpyHostToDevice, houghStream));
	CUDACHECK(cudaMemsetAsync(d_accumulators, 0, sizeof(uint32_t)*accu_width*accu_height, houghStream));

	CUDACHECK(cudaStreamSynchronize(0));
	CUDACHECK(cudaMemcpyAsync(NR, d_NR, sizeof(float)*height*width, cudaMemcpyDeviceToHost, writeBackStream));

	cu_canny_g<<<gridDim1d, blockDim1d, 0>>>(d_NR, d_G, d_phi, height, width);
	CUDACHECK(cudaStreamSynchronize(0));
	CUDACHECK(cudaMemcpyAsync(G, d_G, sizeof(float)*height*width, cudaMemcpyDeviceToHost, writeBackStream));
	CUDACHECK(cudaMemcpyAsync(phi, d_phi, sizeof(float)*height*width, cudaMemcpyDeviceToHost, writeBackStream));

	cu_canny_pedge<<<gridDim1d, blockDim1d, 0>>>(d_phi, d_G, d_pedge, height, width);
	CUDACHECK(cudaStreamSynchronize(0));
	CUDACHECK(cudaMemcpyAsync(pedge, d_pedge, sizeof(uint8_t)*height*width, cudaMemcpyDeviceToHost, writeBackStream));

	cu_canny_hthr<<<gridDim1d, blockDim1d, 0>>>(d_im_out, d_G, d_pedge, height, width, level/2, level*2);

	CUDACHECK(cudaStreamSynchronize(0));
	CUDACHECK(cudaMemcpyAsync(imEdge, d_im_out, sizeof(uint8_t)*height*width, cudaMemcpyDeviceToHost, writeBackStream));

	/*  _   _  ___  _   _  ____ _   _
 	 * | | | |/ _ \| | | |/ ___| | | |
 	 * | |_| | | | | | | | |  _| |_| |
 	 * |  _  | |_| | |_| | |_| |  _  |
 	 * |_| |_|\___/ \___/ \____|_| |_|
 	 */

	float hough_h = ((sqrt(2.0) * (float)(height>width?height:width)) / 2.0);

	printf("Running kernels<<<(%d, %d), (%d, %d)>>> on image size %dx%d\n",
		gridDim1d.x, gridDim1d.y, blockDim1d.x, blockDim1d.y, width, height);

	cu_htkernel<<<gridDim1d, blockDim1d, 0, houghStream>>>(d_im_out, width, height, accu_width, accu_height,
		hough_h, d_sin_table, d_cos_table, d_accumulators);
	CUDACHECK(cudaStreamSynchronize(houghStream));
	CUDACHECK(cudaMemcpyAsync(accum, d_accumulators, sizeof(uint32_t)*accu_width*accu_height, cudaMemcpyDeviceToHost, writeBackStream));

	if (width>height) threshold = width/6;
	else threshold = height/6;

	CUDACHECK(cudaStreamSynchronize(writeBackStream));

	getlines(threshold, accum, accu_width, accu_height, width, height,
		sin_table, cos_table,
		x1, y1, x2, y2, nlines);

	CUDACHECK(cudaFree(d_accumulators));
	CUDACHECK(cudaFree(d_sin_table));
	CUDACHECK(cudaFree(d_cos_table));

	CUDACHECK(cudaFree(d_NR));
	CUDACHECK(cudaFree(d_G));
	CUDACHECK(cudaFree(d_pedge));
	CUDACHECK(cudaFree(d_im_in));
	CUDACHECK(cudaFree(d_im_out));

	CUDACHECK(cudaStreamDestroy(writeBackStream));
	CUDACHECK(cudaStreamDestroy(houghStream));
}
