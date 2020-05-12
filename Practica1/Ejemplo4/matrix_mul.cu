#include <stdio.h>
#include "cublas_v2.h"
#include "matrix_mul.h"

// Host multiplication function
// Compute C = A * B
// hA is the height of A
// wA is the width of A
// wB is the width of B

#define CUDACHECK(f) { \
    const cudaError_t error = f; \
    if (error != cudaSuccess) { \
        printf("Error: %s:%d, ", __FILE__, __LINE__); \
        printf("code: %d, reason: %s\n", error, cudaGetErrorString(error)); \
        exit(1); \
    } \
}

#define CUBLASCHECK(f) { \
    const cublasStatus_t error = f; \
    if (error != CUBLAS_STATUS_SUCCESS) { \
        printf("Error %s:%d, ", __FILE__, __LINE__); \
        printf("code: %d\n", error); \
        exit(1); \
    } \
}

extern "C"
void Mul(float* A, float* B, int hA, int wA, int wB,
	float* C)
{
	int size;

	// Load A and B to the device
	float* Ad;
	size = hA * wA * sizeof(float);
	CUDACHECK(cudaMalloc((void**)&Ad, size));
	CUDACHECK(cudaMemcpy(Ad, A, size, cudaMemcpyHostToDevice));
	float* Bd;
	size = wA * wB * sizeof(float);
	CUDACHECK(cudaMalloc((void**)&Bd, size));
	CUDACHECK(cudaMemcpy(Bd, B, size, cudaMemcpyHostToDevice));

	// Allocate C on the device
	float* Cd;
	size = hA * wB * sizeof(float);
	CUDACHECK(cudaMalloc((void**)&Cd, size));

	// Compute the execution configuration
	const float alpha = 1.0f, beta = 0.0f;
	cublasHandle_t handle;
	CUBLASCHECK(cublasCreate(&handle));
    printf("m: %d, n: %d, k: %d\n", hA, wB, wA);
	CUBLASCHECK(cublasSgemm(handle,
		CUBLAS_OP_N, CUBLAS_OP_N,
		hA,				/* [m] */ 
		wB,				/* [n] */  
		wA,				/* [k] */ 
		&alpha,				/* alfa */ 
		Ad, wA,			/* A[m][k], num columnas (lda) */ 
		Bd, wB,			/* B[k][n], num columnas (ldb) */
		&beta,				/* beta */
		Cd, wB			/* C[m][n], num columnas (ldc) */
	));
	CUBLASCHECK(cublasDestroy_v2(handle));

	// Read C from the device
	CUDACHECK(cudaMemcpy(C, Cd, size, cudaMemcpyDeviceToHost));

	// Free device memory
	CUDACHECK(cudaFree(Ad));
	CUDACHECK(cudaFree(Bd));
	CUDACHECK(cudaFree(Cd));
}
