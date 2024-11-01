#include <stdlib.h>
#include <stdio.h>

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
  int size = 10;

  float* x = malloc(sizeof(float) * size * size);
  float* y = malloc(sizeof(float) * size * size);
  float* z = malloc(sizeof(float) * size * size);

  for (int i = 0; i < size; i++) {
    for (int j = 0; j < size; j++) {
      x[i * size + j] = 1; // x[i][j]
      y[i * size + j] = 1;
    }
  }

  for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            printf("%d ", x[i*size+j]);
        }
        printf("\n");
    }

  
  MatrixMulOnHost(x, y, z, size);

  for (int i = 0; i < size; i++) {
    for (int j = 0; j < size; j++) {
      if (z[i * size + j] != size) {
        printf("Error at z[%d][%d]: %f\n", i, j, z[i * size + j]);
      }
    }
  }


  return 0;
}
