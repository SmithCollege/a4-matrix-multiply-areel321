#include <iostream>
#include <math.h>
#include <sys/time.h>

#define TILE_WIDTH 50

double get_clock() {
        struct timeval tv; int ok;
        ok = gettimeofday(&tv, (void *) 0);
        if (ok<0){
                printf("gettimeofday error");
        }
        return (tv.tv_sec * 1.0 + tv.tv_usec * 1.0E-6);
}


__global__ void MatrixMulKernel(float* d_M, float* d_N, float* d_P, int Width)
{
 //calculate row index of the d_P element and d_M
 int Row = blockIdx.y*blockDim.y+threadIdx.y;
 // Calculate the column idenx of d_P and d_N
 int Col = blockIdx.x*blockDim.x+threadIdx.x;
 
 if ((Row < Width) && (Col < Width)) {
	 float Pvalue = 0.0;
	 // each thread computes one element of the block sub-matrix
	 for (int k = 0; k < Width; ++k){
	  Pvalue += d_M[Row*Width+k] * d_N[k*Width+Col];
	 }
	 d_P[Row * Width + Col] = Pvalue;
  }
}


int main(){

	int width = 2 * TILE_WIDTH;
	float *x, *y, *z;
	float *hx, *hy, *hz;

	double *times = (double *)malloc(sizeof(double)*width);
	 //calibrate clock
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

/*	for (int i=0;i<width;i++){
		for (int j=0;j<width;j++){
			printf("%f ", hx[i*width+j]);
			
		}
		printf("\n");
	}*/
	printf("\n");

	cudaMemcpy(x, hx, sizeof(float)*width*width, cudaMemcpyHostToDevice);
	cudaMemcpy(y, hy, sizeof(float)*width*width, cudaMemcpyHostToDevice);

	// Setup the execution configuration
	// TILE_WIDTH is a #define constant
	
	dim3 dimGrid(ceil((1.0*width)/TILE_WIDTH),
	  ceil((1.0*width)/TILE_WIDTH), 1);
	dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
	cudaDeviceSynchronize();

	double start_time = get_clock();

	 // Launch the device computation threads!
  	MatrixMulKernel<<<dimGrid, dimBlock>>>(x,y, z, width);

	double end_time = get_clock();
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
        printf("start: %f, end: %f\n", start_time, end_time);


	cudaFree(x);
	cudaFree(y);
	cudaFree(z);
	free(hx);
	free(hy);
	free(hz);
	free(times);

	return 0;
}
