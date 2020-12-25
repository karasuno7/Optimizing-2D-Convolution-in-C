#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <immintrin.h>
#include "rdtsc.h"


void kernel
        ( int sizeKernel, int sizeMatrix, int sizeResult, double* restrict matrix, double* restrict result, double* restrict filter ) {
    /*
     * Assumptions:
     * matrix - stored row-wise
     * res - stored row-wise
     * filter - stored row-wise
    */

    register __m256d m0, m1, m2, m3;
    register __m256d r1, r2, r3, r4, r5, r6, r7, r8;

    register __m256d f0, f1, f2, f3;
    f0 = _mm256_loadu_pd(&filter[0]);
    f1 = _mm256_loadu_pd(&filter[4]);
    f2 = _mm256_loadu_pd(&filter[8]);
    f3 = _mm256_loadu_pd(&filter[12]);


    for (int i = 0; i < sizeMatrix - sizeKernel + 1; i = i + 4) {
        for (int j = 0; j < sizeMatrix-sizeKernel+1; j++) {

            if (i==sizeMatrix-sizeKernel){
                r1 = _mm256_setzero_pd();
                r2 = _mm256_setzero_pd();
                r3 = _mm256_setzero_pd();
                r4 = _mm256_setzero_pd();

                m0 = _mm256_loadu_pd(&matrix[(i + 0) * sizeMatrix + j]);
                m1 = _mm256_loadu_pd(&matrix[(i + 1) * sizeMatrix + j]);
                m2 = _mm256_loadu_pd(&matrix[(i + 2) * sizeMatrix + j]);
                m3 = _mm256_loadu_pd(&matrix[(i + 3) * sizeMatrix + j]);

                r1 = _mm256_fmadd_pd(m0, f0, r1);
                r1 = _mm256_fmadd_pd(m1, f1, r1);
                r1 = _mm256_fmadd_pd(m2, f2, r1);
                r1 = _mm256_fmadd_pd(m3, f3, r1);

                r1 = _mm256_hadd_pd(r1, r1);
                r2 = _mm256_permute2f128_pd(r1, r1, 1 | (2 << 4));
                r1 = _mm256_add_pd(r1, r2);

                result[(i + 0) * sizeResult + j] = r1[0];

            }else {
                r1 = _mm256_setzero_pd();
                r2 = _mm256_setzero_pd();
                r3 = _mm256_setzero_pd();
                r4 = _mm256_setzero_pd();
                r5 = _mm256_setzero_pd();
                r6 = _mm256_setzero_pd();
                r7 = _mm256_setzero_pd();
                r8 = _mm256_setzero_pd();

                m0 = _mm256_loadu_pd(&matrix[(i + 0) * sizeMatrix + j]);
                m1 = _mm256_loadu_pd(&matrix[(i + 1) * sizeMatrix + j]);
                m2 = _mm256_loadu_pd(&matrix[(i + 2) * sizeMatrix + j]);
                m3 = _mm256_loadu_pd(&matrix[(i + 3) * sizeMatrix + j]);

                r1 = _mm256_fmadd_pd(m0, f0, r1);
                r2 = _mm256_fmadd_pd(m1, f0, r2);
                r3 = _mm256_fmadd_pd(m2, f0, r3);
                r4 = _mm256_fmadd_pd(m3, f0, r4);

                r5 = _mm256_fmadd_pd(m1, f1, r5);
                r6 = _mm256_fmadd_pd(m2, f1, r6);
                r7 = _mm256_fmadd_pd(m3, f1, r7);

                m0 = _mm256_loadu_pd(&matrix[(i + 4) * sizeMatrix + j]);
                m1 = _mm256_loadu_pd(&matrix[(i + 5) * sizeMatrix + j]);

                r8 = _mm256_fmadd_pd(m0, f1, r8);

                r1 = _mm256_fmadd_pd(m2, f2, r1);
                r2 = _mm256_fmadd_pd(m3, f2, r2);
                r3 = _mm256_fmadd_pd(m0, f2, r3);
                r4 = _mm256_fmadd_pd(m1, f2, r4);
                r5 = _mm256_fmadd_pd(m3, f3, r5);
                r6 = _mm256_fmadd_pd(m0, f3, r6);
                r7 = _mm256_fmadd_pd(m1, f3, r7);

                m2 = _mm256_loadu_pd(&matrix[(i + 6) * sizeMatrix + j]);
                r8 = _mm256_fmadd_pd(m2, f3, r8);

                r1 = _mm256_add_pd(r1, r5);
                r2 = _mm256_add_pd(r2, r6);
                r3 = _mm256_add_pd(r3, r7);
                r4 = _mm256_add_pd(r4, r8);

                m0 = _mm256_setzero_pd();
                m1 = _mm256_setzero_pd();
                m2 = _mm256_setzero_pd();
                m3 = _mm256_setzero_pd();

                r1 = _mm256_hadd_pd(r1, r1);
                m0 = _mm256_permute2f128_pd(r1, r1, 1 | (2 << 4));
                r1 = _mm256_add_pd(r1, m0);

                r2 = _mm256_hadd_pd(r2, r2);
                m1 = _mm256_permute2f128_pd(r2, r2, 1 | (2 << 4));
                r2 = _mm256_add_pd(r2, m1);

                r3 = _mm256_hadd_pd(r3, r3);
                m2 = _mm256_permute2f128_pd(r3, r3, 1 | (2 << 4));
                r3 = _mm256_add_pd(r3, m2);

                r4 = _mm256_hadd_pd(r4, r4);
                m3 = _mm256_permute2f128_pd(r4, r4, 1 | (2 << 4));
                r4 = _mm256_add_pd(r4, m3);

                result[(i + 0) * sizeResult + j] = r1[0];
                result[(i + 1) * sizeResult + j] = r2[0];
                result[(i + 2) * sizeResult + j] = r3[0];
                result[(i + 3) * sizeResult + j] = r4[0];

            }

        }
    }


}

int main(int argc, char **argv){
    int sizeMatrix = atoi(argv[1]);
    int runs = atoi(argv[2]);
    double *matrix ;
    double *filter ;
    double *result ;

    int sizeKernel = 4;
    int padding = 0;
    int strides = 1;
    long long sum1 = 0;
    tsc_counter t0, t1;

    int sizeResult = (((sizeMatrix - sizeKernel + 2 * padding) / strides) + 1);
    printf("size Result: %d\n", sizeResult);

    posix_memalign((void**) &matrix, 64, sizeMatrix * sizeMatrix * sizeof(double));
    posix_memalign((void**) &filter, 64, sizeKernel * sizeKernel * sizeof(double));
    posix_memalign((void**) &result, 64, sizeResult * sizeResult * sizeof(double));

    double count = 0.0;
    for (int i = 0; i != 16; ++i){
        filter[i] = count;
        count= count+ 1.0;
    }


    for(int i = 0; i<sizeMatrix*sizeMatrix;++i){
            matrix[i] = 1.0;
    }

    for(int i = 0; i<sizeResult*sizeResult;++i){
        result[i] = 0.0;

    }

    for(unsigned int i = 0; i != runs; ++i) {
        RDTSC(t0);
        kernel(sizeKernel,sizeMatrix,sizeResult,matrix,result,filter);
        
        RDTSC(t1);
        sum1 += (COUNTER_DIFF(t1, t0, CYCLES));
    }
    printf("Average time: %lf cycles\n", ((double) (sum1 / ((double) runs))));

/**
 * To check correctness uncomment the below code
 */

    // // print filter
    // for(int i = 0; i<sizeKernel;i++){
    //     for(int j= 0; j<sizeKernel;j++){
    //         printf("%f           ",filter[i*sizeKernel + j]);
    //     }
    //     printf("\n");
    // }
    // printf("\n\n\n\n");

    // // print output 
    // for(int i = 0; i<sizeResult;i++){
    //     for(int j= 0; j<sizeResult;j++){
    //        printf("%f           ",result[i*sizeResult + j]);
    //     }
    //     printf("\n");
    // }

    free(matrix);
    free(result);
    free(filter);


}

