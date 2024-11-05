#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>

#define size 100

double get_clock() {
        struct timeval tv; int ok;
        ok = gettimeofday(&tv, (void *) 0);
        if (ok<0){
                printf("gettimeofday error\n");
        }
        return (tv.tv_sec*1.0+tv.tv_usec*1.0E-6);
}

void MatrixMulOnHost(float* M, float* N, float* P, int width) {


  for (int i=0; i<width;++i) {
		for (int j=0; j<width;++j){
			float sum = 0;
			for (int k=0;k<width; ++k){
				float a = M[i*width+k];
				float b = N[k*width+j];
				sum += a*b;
			}
			P[i*width+j] = sum;
		}
	}

}

int main() {
  

  float* x = malloc(sizeof(float) * size * size);
  float* y = malloc(sizeof(float) * size * size);
  float* z = malloc(sizeof(float) * size * size);

double *times = malloc(sizeof(double) * size);


                //calibrate the clock
        double t0 = get_clock();
        for (int i=0; i<size; i++){
                times[i] = get_clock();
        }
        double t1 = get_clock();
        printf("time per call: %f nx\n", (1000000000.0 * (t1-t0\
)/size));
  
  for (int i = 0; i < size; i++) {
    for (int j = 0; j < size; j++) {
      x[i * size + j] = 1.0; // x[i][j]
      y[i * size + j] = 1.0;
    }
  }

/*
  for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            printf("%f ", x[i*size+j]);
        }
        printf("\n");
    }*/

  double start = get_clock();
  
  MatrixMulOnHost(x, y, z, size);

  double end = get_clock();
  
  for (int i = 0; i < size; i++) {
    for (int j = 0; j < size; j++) {
      if (z[i * size + j] != size) {
        printf("Error at z[%d][%d]: %f\n", i, j, z[i * size + j]);
      }
    }
  }

  //print clock times
        printf("start: %f, end: %f\n", start, end);

  return 0;
}
