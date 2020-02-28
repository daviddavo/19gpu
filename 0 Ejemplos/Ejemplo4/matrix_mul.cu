#include <stdio.h>
#include "cublas_v2.h"
#include "matrix_mul.h"

// Host multiplication function
// Compute C = A * B
// hA is the height of A
// wA is the width of A
// wB is the width of B

extern "C"
void Mul(float* A, float* B, int hA, int wA, int wB,
	float* C)
{
	int size;

	// Load A and B to the device
	float* Ad;
	size = hA * wA * sizeof(float);
	cudaMalloc((void**)&Ad, size);
	cudaMemcpy(Ad, A, size, cudaMemcpyHostToDevice);
	float* Bd;
	size = wA * wB * sizeof(float);
	cudaMalloc((void**)&Bd, size);
	cudaMemcpy(Bd, B, size, cudaMemcpyHostToDevice);

	// Allocate C on the device
	float* Cd;
	size = hA * wB * sizeof(float);
	cudaMalloc((void**)&Cd, size);

	// Compute the execution configuration
	float alpha = 1.0, beta = 0.0;
	cublasHandle_t handle;
	cublasCreate(&handle);
	cublasSgemm(handle,
		CUBLAS_OP_N, CUBLAS_OP_N,
		hA,				/* [m] */ 
		wA,				/* [n] */  
		wB,				/* [k] */ 
		&alpha,				/* alfa */ 
		Ad, wA,			/* A[m][k], num columnas (lda) */ 
		Bd, wB,			/* B[k][n], num columnas (ldb) */
		&beta,				/* beta */
		C, wB			/* C[m][n], num columnas (ldc) */
	);
	cublasDestroy_v2(handle);

	// Read C from the device
	cudaMemcpy(C, Cd, size, cudaMemcpyDeviceToHost);

	// Free device memory
	cudaFree(Ad);
	cudaFree(Bd);
	cudaFree(Cd);
}
