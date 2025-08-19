#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <stdio.h>

#define DTYPE double
#define M 2
#define N 4
#define K 4

#include "isclose.h"

void matmul(DTYPE* A, DTYPE* B, DTYPE* C);

void naive_matmul(DTYPE* A, DTYPE* B, DTYPE* C) {
  for (size_t i = 0; i < M; ++i)
    for (size_t j = 0; j < N; ++j)
      for (size_t k = 0; k < K; ++k)
        C[i * N + j] += A[i * K + k] * B[k * N + j];
}

int main() {
  int ret;
  DTYPE *A0, *A, *B0, *B, *C0, *C;
  posix_memalign((void **)&A0, 64, M * K * sizeof(DTYPE));
  posix_memalign((void **)&B0, 64, K * N * sizeof(DTYPE));
  posix_memalign((void **)&C0, 64, M * N * sizeof(DTYPE));
  posix_memalign((void **)&A, 64, M * K * sizeof(DTYPE));
  posix_memalign((void **)&B, 64, K * N * sizeof(DTYPE));
  posix_memalign((void **)&C, 64, M * N * sizeof(DTYPE));

  srand((unsigned)time(NULL));
  
  for (size_t i = 0; i < M; ++i) {
    for (size_t j = 0; j < N; ++j) {
      C[i * N + j] = 0.0;
      for (size_t k = 0; k < K; ++k) {
        A[i * K + k] = (double)rand();
        B[k * N + j] = (double)rand();
      }
    }
  }

  memcpy(A0, A, M * K * sizeof(DTYPE));
  memcpy(B0, B, K * N * sizeof(DTYPE));
  memcpy(C0, C, M * N * sizeof(DTYPE));
        
  naive_matmul(A0,B0,C0);
  matmul(A,B,C);

  if (isclose(C0, C, M * N)) {
    printf("Test Passed: The results are equal!\n");
    ret = 0;
  } else {
    printf("Test Failed: The results do not match.\n");
    ret = 1;
  }
  
  free(A0);
  free(B0);
  free(C0);
  free(A);
  free(B);
  free(C);
  
  return ret;
}
