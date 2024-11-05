//make sure to run as nvcc cublas.cu -lcublas

#include <iostream>
#include <math.h>
#include <sys/time.h>
#include <cublas_v2.h>

#define TILE_WIDTH 2

int main(){

	int width = 2 * TILE_WIDTH;
	float *x, *y, *z;
	float *hx, *hy, *hz;

	hx = (float *)malloc(sizeof(float)*width*width);
	hy = (float *)malloc(sizeof(float)*width*width);
	hz = (float *)malloc(sizeof(float)*width*width);

	cudaMallocManaged(&x, sizeof(float)*width*width);
	cudaMallocManaged(&y, sizeof(float)*width*width);
	cudaMallocManaged(&z, sizeof(float)*width*width);

	for (int i = 0; i < width; i++) {
	    for (int j = 0; j < width; j++) {
	      hx[i * width + j] = 1.0; // x[i][j]
	      hy[i * width + j] = 1.0;
	    }
	  }

	for (int i=0;i<width;i++){
		for (int j=0;j<width;j++){
			printf("%f ", hx[i*width+j]);
			
		}
		printf("\n");
	}
	printf("\n");

	cudaMemcpy(x, hx, sizeof(float)*width*width, cudaMemcpyHostToDevice);
	cudaMemcpy(y, hy, sizeof(float)*width*width, cudaMemcpyHostToDevice);

	// Setup the execution configuration
	// TILE_WIDTH is a #define constant
	
	//dim3 dimGrid(ceil((1.0*width)/TILE_WIDTH),
	//  ceil((1.0*width)/TILE_WIDTH), 1);
	//dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
	cudaDeviceSynchronize();
	 // Launch the device computation threads!

	cublasHandle_t handle;
	cublasCreate(&handle); 

	const float alf = 1;
	const float bet = 0;
	const float *alpha = &alf;
	const float *beta = &bet;

	cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, width, 
	width, width, alpha, x, width, y, width, beta, z, width);

	cublasDestroy(handle);
  	
  	printf("%s\n", cudaGetErrorString(cudaGetLastError()));

	cudaMemcpy(hz, z, sizeof(float)*width*width, cudaMemcpyDeviceToHost);
	for (int i = 0; i < width; i++) {
	    for (int j = 0; j < width; j++) {
	      if (hz[i * width + j] != width) {
	        printf("Error at z[%d][%d]: %f\n", i, j,
	         hz[i * width + j]);
	      }
	    }
	  }


	cudaFree(x);
	cudaFree(y);
	cudaFree(z);
	free(hx);
	free(hy);
	free(hz);

	return 0;
}
