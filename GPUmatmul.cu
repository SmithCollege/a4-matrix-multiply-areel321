#include <iostream>
#include <math.h>
#include <sys/time.h>

#define TILE_WIDTH 2

__global__ void MatrixMulKernel(float* d_M, float* d_N, float* d_P, int Width)
{
 //calculate row index of the d_P element and d_M
 int Row = blockIdx.y*blockDim.y+threadIdx.y;
 // Calculate the column idenx of d_P and d_N
 int Col = blockIdx.x*blockDim.x+threadIdx.x;
 
 if ((Row < Width) && (Col < Width)) {
	 float Pvalue = 0;
	 // each thread computes one element of the block sub-matrix
	 for (int k = 0; k < Width; ++k){
	  Pvalue += d_M[Row*Width+k] * d_N[k*Width+Col];
	 }
	 d_P[Row*Width+Col] = Pvalue;
  }
}


int main(){

	int width = 4;
	float *x, *y, *z;

	cudaMallocManaged(&x, sizeof(float)*width*width);
	cudaMallocManaged(&y, sizeof(float)*width*width);
	cudaMallocManaged(&z, sizeof(float)*width*width);

	for (int i = 0; i < width; i++) {
	    for (int j = 0; j < width; j++) {
	      x[i * width + j] = 1; // x[i][j]
	      y[i * width + j] = 1;
	    }
	  }

	// Setup the execution configuration
	 // TILE_WIDTH is a #define constant
	 dim3 dimGrid(ceil((1.0*width)/TILE_WIDTH),
	ceil((1.0*width)/TILE_WIDTH), 1);
	 dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
	 // Launch the device computation threads!
	 MatrixMulKernel<<<dimGrid, dimBlock>>>(x, y, z, width);

  for (int i = 0; i < width; i++) {
    for (int j = 0; j < width; j++) {
      if (z[i * width + j] != width) {
        printf("Error at z[%d][%d]: %f\n", i, j, z[i * width + j]);
      }
    }
  }

	return 0;
}
