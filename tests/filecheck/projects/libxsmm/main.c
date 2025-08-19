#include <stdlib.h>

#define DTYPE double
#define M 2
#define N 4
#define K 4

void matmul(DTYPE* A, DTYPE* B, DTYPE* C);

int main() {
#ifdef ALIGN
  DTYPE *A, *B, *C;
  posix_memalign((void **)&A, 64, M * K * sizeof(DTYPE));
  posix_memalign((void **)&B, 64, K * N * sizeof(DTYPE));
  posix_memalign((void **)&C, 64, M * N * sizeof(DTYPE));
#else
  DTYPE A[M*K];
  DTYPE B[K*N];
  DTYPE C[M*N];
#endif
  
  matmul(A,B,C);

#ifdef ALIGN
  free(A);
  free(B);
  free(C);
#endif
  
  return 0;
}
