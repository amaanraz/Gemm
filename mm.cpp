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
  int i, j, k;
  int l, m, n;
  int TILE_SIZE = 16;

// => Form C := alpha*A*B + beta*C,
//A is NIxNK
//B is NKxNJ
//C is NIxNJ
  for (i = 0; i < NI; i+= TILE_SIZE) {
    for (j = 0; j < NJ; j += TILE_SIZE) {
      for (k = 0; k < NK; k += TILE_SIZE) {

        // *Tiling loop start
        for (l = i; l < i + TILE_SIZE && l < NI; l++) {         
          for (m = j; m < j + TILE_SIZE && m < NJ; m++) {
            if(k == 0) {
              C[l*NJ+m] *= beta;  
            }
            for (n = k; n < k + TILE_SIZE && n < NK; n++) {
              C[l*NJ+m] +=  alpha * A[l*NK+n] * B[n*NJ+m];             
            }
          }
        }
        // *Tiling loop end       
      }   
    }
  }
}

/* Main computational kernel: with tiling and simd optimizations. */
static
void gemm_tile_simd(float C[NI*NJ], float A[NI*NK], float B[NK*NJ], float alpha, float beta)
{
    int i, j, k, x, y, z;
    int VECTOR_SIZE = 8;
    int TILE_SIZE = 32;
    // Broadcast alpha to SIMD vector
    __m256 alpha_vec = _mm256_set1_ps(alpha);

    // Loop over tiles in i, j, and k dimensions
    // Loop over tiles in i, j, and k dimensions
    for (i = 0; i < NI; i += TILE_SIZE) {
        for (j = 0; j < NJ; j += TILE_SIZE) {
            for (k = 0; k < NK; k += TILE_SIZE) {
                // Tile level computation
                if (k == 0) {
                    for (x = i; x < i + TILE_SIZE && x < NI; x++) {
                        for (y = j; y < j + TILE_SIZE && y < NJ; y++) {
                            C[x * NJ + y] *= beta;
                        }
                    }
                }

                // Perform matrix multiplication on the tiles using SIMD intrinsics
                for (x = i; x < i + TILE_SIZE && x < NI; x++) {
                    for (y = j; y < j + TILE_SIZE && y < NJ; y+=8) {
                        __m256 c_vec = _mm256_loadu_ps(&C[x * NJ + y]);
                        for (z = k; z < k + TILE_SIZE && z < NK; z++) {
                            // Load A and B vectors
                            __m256 a_vec = _mm256_set1_ps(A[x * NK + z]);
                            __m256 b_vec = _mm256_loadu_ps(&B[z * NJ + y]);

                            // Multiply A, B, and alpha vectors
                            __m256 result = _mm256_mul_ps(_mm256_mul_ps(a_vec, b_vec), alpha_vec);
                            
                            c_vec = _mm256_add_ps(c_vec, result);
                        }
                        // Store the updated C vector
                        _mm256_storeu_ps(&C[x * NJ + y], c_vec);
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
 int i, j, k, x, y, z;
    const int VECTOR_SIZE = 8;
    const int TILE_SIZEI = 32;
    const int TILE_SIZEJ = 64;
    const int TILE_SIZEK = 64;
    const int NUM_THREADS = 18;
    // Broadcast alpha to SIMD vector
    __m256 alpha_vec = _mm256_set1_ps(alpha);
    
    if (beta != 1.0f) {
        for (i = 0; i < NI; i++) {
            for (j = 0; j < NJ; j++) {
                C[i * NJ + j] *= beta;
            }
        }
    }

    // omp_set_num_threads(6);
     // Parallelize the outer loops using OpenMP
    #pragma omp parallel for private(i, j, k, x, y, z)
    for (i = 0; i < NI; i += TILE_SIZEI) {
       for (j = 0; j < NJ; j += TILE_SIZEJ) {
          for (k = 0; k < NK; k += TILE_SIZEK) {
            
                // Tile level computation
                int x_end = i + TILE_SIZEI;
                int y_end = j + TILE_SIZEJ;
                int z_end = k + TILE_SIZEK;
                if (x_end > NI) x_end = NI;
                if (y_end > NJ) y_end = NJ;
                if (z_end > NK) z_end = NK;

                // Perform matrix multiplication on the tiles using SIMD
                #pragma omp parallel for private(x, y, z)
                for (x = i; x < x_end; x++) {
                    for (y = j; y < y_end; y += 4*VECTOR_SIZE) {
                        __m256 c_vec[4];
                        c_vec[0] = _mm256_loadu_ps(&C[x * NJ + y]);
                        c_vec[1] = _mm256_loadu_ps(&C[x * NJ + y + VECTOR_SIZE]);
                        c_vec[2] = _mm256_loadu_ps(&C[x * NJ + y + 2 * VECTOR_SIZE]);
                        c_vec[3] = _mm256_loadu_ps(&C[x * NJ + y + 3 * VECTOR_SIZE]);

                        for (z = k; z < z_end; z++) {
                            // Load A and B vectors
                            __m256 a_vec = _mm256_set1_ps(A[x * NK + z]);
                            __m256 b_vec[4];
                            b_vec[0] = _mm256_loadu_ps(&B[z * NJ + y]);
                            b_vec[1] = _mm256_loadu_ps(&B[z * NJ + y + VECTOR_SIZE]);
                            b_vec[2] = _mm256_loadu_ps(&B[z * NJ + y + 2 * VECTOR_SIZE]);
                            b_vec[3] = _mm256_loadu_ps(&B[z * NJ + y + 3 * VECTOR_SIZE]);


                            // Multiply A, B, and alpha vectors
                            __m256 result[4];
                            result[0] = _mm256_mul_ps(_mm256_mul_ps(a_vec, b_vec[0]), alpha_vec);
                            result[1] = _mm256_mul_ps(_mm256_mul_ps(a_vec, b_vec[1]), alpha_vec);
                            result[2] = _mm256_mul_ps(_mm256_mul_ps(a_vec, b_vec[2]), alpha_vec);
                            result[3] = _mm256_mul_ps(_mm256_mul_ps(a_vec, b_vec[3]), alpha_vec);


                            // Add the result to the C vector
                            c_vec[0] = _mm256_add_ps(c_vec[0], result[0]);
                            c_vec[1] = _mm256_add_ps(c_vec[1], result[1]);
                            c_vec[2] = _mm256_add_ps(c_vec[2], result[2]);
                            c_vec[3] = _mm256_add_ps(c_vec[3], result[3]);
                        }

                        // Store the updated C vector
                        _mm256_storeu_ps(&C[x * NJ + y], c_vec[0]);
                        _mm256_storeu_ps(&C[x * NJ + y + VECTOR_SIZE], c_vec[1]);
                        _mm256_storeu_ps(&C[x * NJ + y + 2 * VECTOR_SIZE], c_vec[2]);
                        _mm256_storeu_ps(&C[x * NJ + y + 3 * VECTOR_SIZE], c_vec[3]);
                    }
                }
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
