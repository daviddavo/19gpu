#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include "routinesCPU.h"
#include "routinesGPU.h"
#include "png_io.h"

/* Time */
#include <sys/time.h>
#include <sys/resource.h>


static struct timeval tv0;
double get_time()
{
	double t;
	gettimeofday(&tv0, (struct timezone*)0);
	t = ((tv0.tv_usec) + (tv0.tv_sec)*1000000);

	return (t);
}

bool array_eq(const unsigned *A, const unsigned *B, unsigned n) {
	for (unsigned i = 0; i < n; i++) {
		if (A[i] != B[i]) {
			printf("A[%d] (%d) != B[%d] (%d)\n", i, A[i], i, B[i]);
			return false;
		}
	}

	return true;
}

// TODO: Make a function where e is a percentage
bool array_eq_abs(const float *A, const float *B, unsigned n, double e) {
	for (unsigned i = 0; i < n; i++) {
		if (fabs(A[i] - B[i]) > e) {
			printf("A[%d] (%f) != B[%d] (%f), e: %lf, diff: %lf\n", i, A[i], i, B[i], e, fabsf(A[i]-B[i]));
			return false;
		}
	}

	return true;
}

bool array_eq_rel(const float*A, const float *B, unsigned n, double r) {
	for (unsigned i = 0; i < n; i++) {
		if (fabs(B[i]) > 0.000005 && fabs(A[i]/B[i]-1) > r) {
			printf("A[%d] (%f) != B[%d] (%f), r: %lf, diff: %lf\n", i, A[i], i, B[i], r, fabs(A[i]/B[i] - 1));
			return false;
		}
	}

	return true;
}

bool gpucheck(const uint8_t *im, int height, int width,
	const uint8_t *imEdge, const float *g_NR, const float *g_G, const float *g_phi,
	const float *g_Gx, const float *g_Gy, const uint8_t *g_pedge,
	const float *g_sin_table, const float *g_cos_table,
	const uint32_t *g_accum, int g_accu_height, int g_accu_width,
	const int *g_x1, const int *g_y1, const int *g_x2, const int *g_y2, const int *g_nlines)
{
	int c_x1[10], c_x2[10], c_y1[10], c_y2[10];
	float c_sin_table[180], c_cos_table[180];

	uint8_t *c_imEdge = (uint8_t *)malloc(sizeof(uint8_t) * width * height);
	float *c_NR = (float *)malloc(sizeof(float) * width * height);
	float *c_G = (float *)malloc(sizeof(float) * width * height);
	float *c_phi = (float *)malloc(sizeof(float) * width * height);
	float *c_Gx = (float *)malloc(sizeof(float) * width * height);
	float *c_Gy = (float *)malloc(sizeof(float) * width * height);
	uint8_t *c_pedge = (uint8_t *)malloc(sizeof(uint8_t) * width * height);

	//Create the accumulators
	float c_hough_h = ((sqrt(2.0) * (float)(height>width?height:width)) / 2.0);
	int c_accu_height = c_hough_h * 2.0; // -rho -> +rho
	int c_accu_width  = 180;
	uint32_t *c_accum = (uint32_t*)malloc(c_accu_width*c_accu_height*sizeof(uint32_t));

	init_cos_sin_table(c_sin_table, c_cos_table, 180);

	int c_nlines = 0;
	line_asist_CPU(im, height, width,
		c_imEdge, c_NR, c_G, c_phi, c_Gx, c_Gy, c_pedge,
		c_sin_table, c_cos_table,
		c_accum, c_accu_height, c_accu_width,
		c_x1, c_y1, c_x2, c_y2, &c_nlines);

	bool s = true;
	if (!array_eq_rel(g_NR, c_NR, height*width, 0.01f)) {
		printf("WARNING: g_NR != c_NR\n");
		s = false;
	}

	if (!array_eq_rel(g_G, c_G, height*width, 0.01f)) {
		printf("WARNING: g_G != c_G\n");
		s = false;
	}

	if(!array_eq_rel(g_phi, c_phi, height*width, 0.25f)) {
		printf("WARNING: g_phi != c_phi\n");
		s = false;
	}

	if(!array_eq(g_accum, c_accum, c_accu_height*c_accu_width)) {
		printf("WARNING: g_accum != c_accum\n");
		s = false;
	}

	if(*g_nlines != c_nlines) {
		printf("WARNING: g_nlines != c_nlines\n");
		s = false;
	}

	return s;
}

void pag_alloc(uint8_t ** imEdge, float **NR, float **G, float **phi, float **Gx,
		float **Gy, uint8_t **pedge, uint32_t **accum, int width, int height,
		int accu_height, int accu_width)
{
	*imEdge = (uint8_t *)malloc(sizeof(uint8_t) * width * height);
	*NR = (float *)malloc(sizeof(float) * width * height);
	*G = (float *)malloc(sizeof(float) * width * height);
	*phi = (float *)malloc(sizeof(float) * width * height);
	*Gx = (float *)malloc(sizeof(float) * width * height);
	*Gy = (float *)malloc(sizeof(float) * width * height);
	*pedge = (uint8_t *)malloc(sizeof(uint8_t) * width * height);

	*accum = (uint32_t *)malloc(sizeof(uint32_t) * accu_width * accu_height);
}

int main(int argc, char **argv)
{
	uint8_t *imtmp, *im;
	int width, height;

	float sin_table[180], cos_table[180];
	int nlines=0; 
	int x1[10], x2[10], y1[10], y2[10];
	int l;
	double t0, t1;


	/* Only accept a concrete number of arguments */
	if(argc < 3 || argc > 4)
	{
		printf("./exec image.png [c/g] {out.png}\n");
		exit(-1);
	}

	/* Read images */
	imtmp = read_png_fileRGB(argv[1], &width, &height);
	im    = image_RGB2BW(imtmp, height, width);

	init_cos_sin_table(sin_table, cos_table, 180);	

	// Create temporal buffers
	uint8_t *imEdge;
	float *NR, *G, *phi, *Gx, *Gy;
	uint8_t *pedge;
	uint32_t * accum;
	float hough_h = ((sqrt(2.0) * (float)(height>width?height:width)) / 2.0);
	int accu_height = hough_h * 2.0; // -rho -> +rho
	int accu_width  = 180;

	switch (argv[2][0]) {
		case 'c':
			pag_alloc(&imEdge, &NR, &G, &phi, &Gx, &Gy, &pedge, &accum,
				width, height, accu_height, accu_width);
			t0 = get_time();
			line_asist_CPU(im, height, width, 
				imEdge, NR, G, phi, Gx, Gy, pedge,
				sin_table, cos_table,
				accum, accu_height, accu_width,
				x1, y1, x2, y2, &nlines);
			t1 = get_time();
			printf("CPU Exection time %f ms.\n", t1-t0);
			break;
		case 'g':
		{
			cu_alloc(&imEdge, &NR, &G, &phi, &Gx, &Gy, &pedge, &accum,
				width, height, accu_height, accu_width);
			t0 = get_time();
			line_asist_GPU(im, height, width,
				imEdge, NR, G, phi, Gx, Gy, pedge,
				sin_table, cos_table,
				accum, accu_height, accu_width,
				x1, x2, y1, y2, &nlines);
            t1 = get_time();
			printf("GPU Exection time %f ms.\n", t1-t0);

			bool chk = gpucheck(im, height, width,
					imEdge, NR, G, phi, Gx, Gy, pedge,
					sin_table, cos_table,
					accum, accu_height, accu_width,
					x1, x2, y1, y2, &nlines);
			if (chk)
			{
				printf("CPU's png == GPU's png\n");
			} else {
				printf("CPU's png != GPU's png\n");
			}
		}
			break;
		default:
			printf("Not Implemented yet!!\n");
	}

	for (int l=0; l<nlines; l++)
		printf("(x1,y1)=(%d,%d) (x2,y2)=(%d,%d)\n", x1[l], y1[l], x2[l], y2[l]);

	draw_lines(imtmp, width, height, x1, y1, x2, y2, nlines);

	write_png_fileRGB((argc==4)?argv[3]:"out.png", imtmp, width, height);
}
