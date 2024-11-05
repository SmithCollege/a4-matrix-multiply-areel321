//make sure to run as nvcc cublas.cu -lcublas

#include <iostream>
#include <math.h>
#include <sys/time.h>
#include <cublas_v2.h>

#define TILE_WIDTH 50

double get_clock() {
        struct timeval tv; int ok;
        ok = gettimeofday(&tv, (void *) 0);
        if (ok<0){
                printf("gettimeofday error\n");
        }
        return (tv.tv_sec*1.0+tv.tv_usec*1.0E-6);
}


int main(){

	int width = 2 * TILE_WIDTH;
	float *x, *y, *z;
	float *hx, *hy, *hz;

double *times = (double *)malloc(sizeof(double) * width);


                //calibrate the clock
        double t0 = get_clock();
        for (int i=0; i<width; i++){
                times[i] = get_clock();
        }
        double t1 = get_clock();
        printf("time per call: %f nx\n", (1000000000.0 * (t1-t0\
)/width));


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

	/*for (int i=0;i<width;i++){
		for (int j=0;j<width;j++){
			printf("%f ", hx[i*width+j]);
			
		}
		printf("\n");
	}*/
	printf("\n");

	cudaMemcpy(x, hx, sizeof(float)*width*width, cudaMemcpyHostToDevice);
	cudaMemcpy(y, hy, sizeof(float)*width*width, cudaMemcpyHostToDevice);

	
	cudaDeviceSynchronize();

	cublasHandle_t handle;
	cublasCreate(&handle); 

	const float alf = 1;
	const float bet = 0;
	const float *alpha = &alf;
	const float *beta = &bet;

double start = get_clock();

	cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, width, 
	width, width, alpha, x, width, y, width, beta, z, width);

double end = get_clock();

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

//print clock times
        printf("start: %f, end: %f\n", start, end);
	cudaFree(x);
	cudaFree(y);
	cudaFree(z);
	free(hx);
	free(hy);
	free(hz);
	free(times);

	return 0;
}
