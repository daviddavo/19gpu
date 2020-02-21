#include <stdio.h>
#include <malloc.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>

#include <cuda.h>

double wtime(void)
{
        static struct timeval   tv0;
        double time_;

        gettimeofday(&tv0,(struct timezone*)0);
        time_=(double)((tv0.tv_usec + (tv0.tv_sec)*1000000));
        return( time_/1000000);
}


void Mul(float *A, float *B, int hA, int wA, int wB, float *C)
{
	int i,j,k;

	for (i=0; i<hA; i++)
		for (j=0; j<wB; j++){
			C[i*wB+j] = 0.0;
			for (k=0; k<wA; k++)
				C[i*wB+j] += A[i*wA+k]*B[k*wB+j];
		}
}

__global__ void cudaMul(float * A, float * B, int ha, int wa, int wb, float * C) {
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	int j = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < ha && j < wb)
		for (int k = 0; k < wa; k++)
			C[i*wb+j] += A[i*wa+k]*B[k*wb+j];
}

void init_matrix(float *M, int hM, int wM, float k)
{
	int i,j;

	for (i=0; i<hM; i++)
		for (j=0; j<wM; j++)
			if (i==j)
				M[i*wM+j] = k*1.0f;
			else
				M[i*wM+j] = -1.0f/(float)(wM);
}

void print_matrix(float *M, int hM, int wM)
{
	int i,j;

	for (i=0; i<hM; i++){
//		printf("Line %i: ", i);
		for (j=0; j<wM; j++)
			printf("%4.1f ", M[i*wM+j]);
		printf("\n");
	}
}

int diff(float *A, float *B, int hA, int wA, int wB, float *C)
{
	float *C_cpu;
	int size_C = wB * hA;
	C_cpu = (float*)malloc(size_C*sizeof(float));

	int i,j,k;

	double t0, t1;
	t0 = wtime();
	for (i=0; i<hA; i++)
		for (j=0; j<wB; j++){
			C_cpu[i*wB+j] = 0.0;
			for (k=0; k<wA; k++){
				C_cpu[i*wB+j] += A[i*wA+k]*B[k*wB+j];
			}
		}
	t1 = wtime();
	printf("Time CPU: %f\n", t1-t0);
	//printf("\n\nMATRIX C_cpu\n");print_matrix(C_cpu, hA, wB);

	for (i=0; i<hA; i++)
		for (j=0; j<wB; j++)
			if (fabsf(C_cpu[i*wB+j]-C[i*wB+j])>1e-5)
			{
				printf("[%i,%i]: %f!=%f\n", i, j, C_cpu[i*wB+j], C[i*wB+j]);
				return(0);
			}


	return(1);

}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv)
{
	// Matrix variables
	float *A, *B, *C;
	float *A_GPU, *B_GPU, *C_GPU;

	int hA, wA, hB, wB;
	// int i;

	setbuf(stdout, NULL);

	if (argc!=4){
		printf("./exec hA hB/WA wB\n");
		exit(-1);
	}

	hA = atoi(argv[1]);
	hB = wA = atoi(argv[2]);
	wB = atoi(argv[3]);

	// Init A and B, malloc C
	int size_A = wA * hA;
	A = (float*)malloc(size_A*sizeof(float));
	init_matrix(A, hA, wA, 1.0);

	int size_B = wB * hB;
	B = (float*)malloc(size_B*sizeof(float));
	init_matrix(B, hB, wB, 2.0);

	// We will initialize C while cudaMemCpy works
	int size_C = wB * hA;
	
	cudaMalloc(&A_GPU, size_A*sizeof(float));
	cudaMalloc(&B_GPU, size_B*sizeof(float));
	cudaMalloc(&C_GPU, size_C*sizeof(float));

	cudaMemcpy(A_GPU, A, size_A*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(B_GPU, B, size_B*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemset(C_GPU, 0.0f, size_C*sizeof(float));

	C = (float*)malloc(size_C*sizeof(float));

	// Mul(A, B, hA, wA, wB, C);
	// printf("\n\nMATRIX A\n");print_matrix(A, hA, wA);
	// printf("\n\nMATRIX B\n");print_matrix(B, hB, wB);
	// printf("\n\nMATRIX C\n");print_matrix(C, hA, wB);

	#define THX 32
	#define THY 32
	// Ahora sí que hacemos el calculo
	dim3 b(THX, THY); // Threads por bloque
	dim3 g(ceil(float(hA)/float(b.x)), ceil(float(wB)/float(b.y))); // Bloques por grid
	printf("C_SIZE: %d x %d (%d), b: %d x %d, g: %d x %d\n", wB, hA, size_C, b.x, b.y, g.x, g.y);
	double t0, t1;

	t0 = wtime();
	cudaMul<<<g, b>>>(A_GPU, B_GPU, hA, wA, wB, C_GPU);
	cudaDeviceSynchronize();
	t1 = wtime();
	printf("Time GPU: %f\n", t1-t0);

	cudaMemcpy(C, C_GPU, size_C*sizeof(float), cudaMemcpyDeviceToHost);

	if (!diff(A, B, hA, wA, wB, C))
		printf("ERROR=GPU.vs.CPU matrix mult differs\n");
	else
		printf("Everything went fine\n");

	// print Matrix
	// printf("\n\nMATRIX A\n");print_matrix(A, hA, wA);
	// printf("\n\nMATRIX B\n");print_matrix(B, hB, wB);
	// printf("\n\nMATRIX C\n");print_matrix(C, hA, wB);

	free(A); free(B); free(C);

	cudaFree(A_GPU);
	cudaFree(B_GPU);
	cudaFree(C_GPU);

	return (1);
}

