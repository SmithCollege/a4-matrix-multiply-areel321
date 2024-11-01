#include <iostream>
#include <math.h>
#include <sys/time.h>

#define TILE_WIDTH 2

__global__ void MatrixMulKernel(float* M, float* N, float* P, int Width)
{
 __shared__ float subTileM[TILE_WIDTH][TILE_WIDTH];
 __shared__ float subTileN[TILE_WIDTH][TILE_WIDTH];
 int bx = blockIdx.x; int by = blockIdx.y;
 int tx = threadIdx.x; int ty = threadIdx.y;
 // Identify the row and column of the P element to work on
 int Row = by * TILE_WIDTH + ty;
 int Col = bx * TILE_WIDTH + tx;
 float Pvalue = 0;
 // Loop over the M and N tiles required to compute the P element
 // The code assumes that the Width is a multiple of TILE_WIDTH!
 for (int m = 0; m < Width/TILE_WIDTH; ++m) {
 // Collaborative loading of M and N tiles into shared memory
 subTileM[ty][tx] = M[Row*Width + m*TILE_WIDTH+tx];
 subTileN[ty][tx] = N[(m*TILE_WIDTH+ty)*Width+Col];
 __syncthreads();
 for (int k = 0; k < TILE_WIDTH; ++k)
 Pvalue += subTileM[ty][k] * subTileN[k][tx];
 __syncthreads();
 }
 P[Row*Width+Col] = Pvalue;

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
