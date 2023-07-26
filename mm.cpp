#include <stdio.h>
#include <stdlib.h>
#include "my_timer.h"
#include <x86intrin.h>
#include <omp.h>

#define NI 4096
#define NJ 4096
#define NK 4096

/* Array initialization. */
static
void init_array(float C[NI*NJ], float A[NI*NK], float B[NK*NJ])
{
  int i, j;

  for (i = 0; i < NI; i++)
    for (j = 0; j < NJ; j++)
      C[i*NJ+j] = (float)((i*j+1) % NI) / NI;
  for (i = 0; i < NI; i++)
    for (j = 0; j < NK; j++)
      A[i*NK+j] = (float)(i*(j+1) % NK) / NK;
  for (i = 0; i < NK; i++)
    for (j = 0; j < NJ; j++)
      B[i*NJ+j] = (float)(i*(j+2) % NJ) / NJ;
}

/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(float C[NI*NJ])
{
  int i, j;

  for (i = 0; i < NI; i++)
    for (j = 0; j < NJ; j++)
      printf("C[%d][%d] = %f\n", i, j, C[i*NJ+j]);
}

/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_and_valid_array_sum(float C[NI*NJ])
{
  int i, j;

  float sum = 0.0;
  float golden_sum = 27789682688.000000;
  
  for (i = 0; i < NI; i++)
    for (j = 0; j < NJ; j++)
      sum += C[i*NJ+j];

  if ( abs(sum-golden_sum)/golden_sum > 0.00001 ) // more than 0.001% error rate
    printf("Incorrect sum of C array. Expected sum: %f, your sum: %f\n", golden_sum, sum);
  else
    printf("Correct result. Sum of C array = %f\n", sum);
}


/* Main computational kernel: baseline. The whole function will be timed,
   including the call and return. DO NOT change the baseline.*/
static
void gemm_base(float C[NI*NJ], float A[NI*NK], float B[NK*NJ], float alpha, float beta)
{
  int i, j, k;

// => Form C := alpha*A*B + beta*C,
//A is NIxNK
//B is NKxNJ
//C is NIxNJ
  for (i = 0; i < NI; i++) {
    for (j = 0; j < NJ; j++) {
      C[i*NJ+j] *= beta;
    }
    for (j = 0; j < NJ; j++) {
      for (k = 0; k < NK; ++k) {
	C[i*NJ+j] += alpha * A[i*NK+k] * B[k*NJ+j];
      }
    }
  }
}

/* Main computational kernel: with tiling optimization. */
static
void gemm_tile(float C[NI*NJ], float A[NI*NK], float B[NK*NJ], float alpha, float beta)
{
  int i, j, k, ii, jj, kk;
  int TILE_SIZE = 32; // Choose an appropriate tile size (experimentally).

  // => Form C := alpha*A*B + beta*C,
  // A is NIxNK
  // B is NKxNJ
  // C is NIxNJ

  // Initialize the tiles for A, B, and C
  // float A_tile[TILE_SIZE * TILE_SIZE];
  // float B_tile[TILE_SIZE * TILE_SIZE];
  // float C_tile[TILE_SIZE * TILE_SIZE];

  // // Loop over tiles in i, j, and k dimensions
  // for (i = 0; i < NI; i += TILE_SIZE) {
  //   for (j = 0; j < NJ; j += TILE_SIZE) {
  //     // Initialize the current tile of C to zero
  //     for (ii = 0; ii < TILE_SIZE; ii++) {
  //       for (jj = 0; jj < TILE_SIZE; jj++) {
  //         C_tile[ii * TILE_SIZE + jj] = 0.0;
  //       }
  //     }

  //     for (k = 0; k < NK; k += TILE_SIZE) {
  //       // Load the current tile of A into A_tile
  //       for (ii = 0; ii < TILE_SIZE; ii++) {
  //         for (kk = 0; kk < TILE_SIZE; kk++) {
  //           A_tile[ii * TILE_SIZE + kk] = A[(i + ii) * NK + (k + kk)];
  //         }
  //       }

  //       // Load the current tile of B into B_tile
  //       for (kk = 0; kk < TILE_SIZE; kk++) {
  //         for (jj = 0; jj < TILE_SIZE; jj++) {
  //           B_tile[kk * TILE_SIZE + jj] = B[(k + kk) * NJ + (j + jj)];
  //         }
  //       }

  //       // Perform matrix multiplication on the tiles
  //       for (ii = 0; ii < TILE_SIZE; ii++) {
  //         for (jj = 0; jj < TILE_SIZE; jj++) {
  //           for (kk = 0; kk < TILE_SIZE; kk++) {
  //             C_tile[ii * TILE_SIZE + jj] += alpha * A_tile[ii * TILE_SIZE + kk] * B_tile[kk * TILE_SIZE + jj];
  //           }
  //         }
  //       }
  //     }

  //     // Store the computed tile of C back to C
  //     for (ii = 0; ii < TILE_SIZE; ii++) {
  //       for (jj = 0; jj < TILE_SIZE; jj++) {
  //         C[(i + ii) * NJ + (j + jj)] = beta * C[(i + ii) * NJ + (j + jj)] + C_tile[ii * TILE_SIZE + jj];
  //       }
  //     }
  //   }
  // }

  for(int i = 0; i < NI; i += TILE_SIZE){
    for(int j = 0; j < NJ; j += TILE_SIZE){
      for(int k = 0; k < NK; k += TILE_SIZE){
        
        if(k == 0){
          for(int x = i; x < i + TILE_SIZE && x < NI; x++){
            for(int y = j; y < j + TILE_SIZE && y < NJ; y++){
              C[x*NJ+y] *= beta;
            }
          }
        }

        for(int x = i; x < i + TILE_SIZE && x < NI; x++){
          for(int y = j; y < j + TILE_SIZE && y < NJ; y++){
            for(int z = k; z < k + TILE_SIZE && z < NK; z++){
              C[x*NJ+y] += alpha * A[x*NK+z] * B[z*NJ+y];
            }
          }
        }
      }
    }
  }
}

/* Main computational kernel: with tiling and simd optimizations. */
static
void gemm_tile_simd(float C[NI*NJ], float A[NI*NK], float B[NK*NJ], float alpha, float beta)
{
  {
    int i, j, k, x, y, z;
    int VECTOR_SIZE = 8;
    int TILE_SIZE = 32;

    // Loop over tiles in i, j, and k dimensions
    for (i = 0; i < NI; i += TILE_SIZE) {
        for (j = 0; j < NJ; j += TILE_SIZE) {
            for (k = 0; k < NK; k += TILE_SIZE) {
                // Initialize the current tile of C to zero (if k == 0)
                if (k == 0) {
                    for (x = i; x < i + TILE_SIZE && x < NI; x++) {
                        for (y = j; y < j + TILE_SIZE && y < NJ; y++) {
                            C[x * NJ + y] *= beta;
                        }
                    }
                }

                // Perform matrix multiplication on the tiles using SIMD intrinsics
                for (x = i; x < i + TILE_SIZE && x < NI; x++) {
                    for (y = j; y < j + TILE_SIZE && y < NJ; y++) {
                        for (z = k; z < k + TILE_SIZE && z < NK; z++) {
                            __m256 A_tile = _mm256_set1_ps(alpha * A[x * NK + z]);
                            __m256 B_tile = _mm256_loadu_ps(&B[z * NJ + y]);

                            __m256 mul_result = _mm256_mul_ps(A_tile, B_tile);

                            // Extract the individual elements from mul_result and add to C[x * NJ + y]
                            float tmp[8];
                            _mm256_storeu_si256((__m256*)tmp, mul_result);
                            for (int i = 0; i < 8; i++) {
                                C[x * NJ + y + i] += tmp[i];
                            }
                        }
                    }
                }
            }
        }
    }
}
}

/* Main computational kernel: with tiling, simd, and parallelization optimizations. */
static
void gemm_tile_simd_par(float C[NI*NJ], float A[NI*NK], float B[NK*NJ], float alpha, float beta)
{
  int i, j, k;

// => Form C := alpha*A*B + beta*C,
//A is NIxNK
//B is NKxNJ
//C is NIxNJ
  for (i = 0; i < NI; i++) {
    for (j = 0; j < NJ; j++) {
      C[i*NJ+j] *= beta;
    }
    for (j = 0; j < NJ; j++) {
      for (k = 0; k < NK; ++k) {
	C[i*NJ+j] += alpha * A[i*NK+k] * B[k*NJ+j];
      }
    }
  }
}

int main(int argc, char** argv)
{
  /* Variable declaration/allocation. */
  float *A = (float *)malloc(NI*NK*sizeof(float));
  float *B = (float *)malloc(NK*NJ*sizeof(float));
  float *C = (float *)malloc(NI*NJ*sizeof(float));

  /* opt selects which gemm version to run */
  int opt = 0;
  if(argc == 2) {
    opt = atoi(argv[1]);
  }
  //printf("option: %d\n", opt);
  
  /* Initialize array(s). */
  init_array (C, A, B);

  /* Start timer. */
  timespec timer = tic();

  switch(opt) {
  case 0: // baseline
    /* Run kernel. */
    gemm_base (C, A, B, 1.5, 2.5);
    /* Stop and print timer. */
    toc(&timer, "baseline time");
    break;
  case 1: // tiling
    /* Run kernel. */
    gemm_tile (C, A, B, 1.5, 2.5);
    /* Stop and print timer. */
    toc(&timer, "tiling time");
    break;
  case 2: // tiling and simd
    /* Run kernel. */
    gemm_tile_simd (C, A, B, 1.5, 2.5);
    /* Stop and print timer. */
    toc(&timer, "tiling-simd time");
    break;
  case 3: // tiling, simd, and parallelization
    /* Run kernel. */
    gemm_tile_simd_par (C, A, B, 1.5, 2.5);
    /* Stop and print timer. */
    toc(&timer, "tiling-simd-par time");
    break;
  default: // baseline
    /* Run kernel. */
    gemm_base (C, A, B, 1.5, 2.5);
    /* Stop and print timer. */
    toc(&timer, "baseline time");
  }
  /* Print results. */
  print_and_valid_array_sum(C);

  /* free memory for A, B, C */
  free(A);
  free(B);
  free(C);
  
  return 0;
}
