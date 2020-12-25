#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "rdtsc.h"
#include <immintrin.h>
#include <omp.h>

void kernel( int sizeKernel, int sizeMatrix, int sizeResult, double* restrict matrix, double* restrict result, double* restrict filter, int thread) {
    /*
     * Assumptions:
     * matrix - stored row-wise
     * res - stored row-wise
     * filter - stored row-wise
    */
    
    #pragma omp parallel num_threads(thread)
    {
    int p = omp_get_num_threads();
    int id = omp_get_thread_num();
    for (int i = id * sizeResult/p; i < (id * sizeResult/p) + sizeResult/p ; i++) {
        for (int j = 0; j < sizeResult; j++) {
            for (int kx = 0; kx < sizeKernel; kx++) {
                for (int ky = 0; ky < sizeKernel; ky++) {
                    result[i * sizeResult + j] += matrix[(i * sizeMatrix)+ j + kx + ky] * filter[kx * sizeKernel + ky];
                }
            }
        }
    }}
}


int main(int argc, char **argv) {
    int sizeMatrix = atoi(argv[1]);
    int thread = atoi(argv[2]);
    int runs = atoi(argv[3]);

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

    for (int i = 0; i != 16; ++i) {
        filter[i] = 2;
    }

    for(int i = 0; i<sizeMatrix*sizeMatrix;++i) {
            matrix[i] = 1.0;
    }

    for(int i = 0; i<sizeResult*sizeResult;++i) {
        result[i] = 0.0;
    }
    
    for (unsigned int i = 0; i != runs; ++i) {
        RDTSC(t0);
        kernel(sizeKernel,sizeMatrix,sizeResult,matrix,result,filter, thread);
        RDTSC(t1);
        sum1 += (COUNTER_DIFF(t1, t0, CYCLES));
    }
    printf("Average time: %lf cycles\n", ((double) (sum1 / ((double) runs))));

    /** To check correctness: Uncomment below code **/

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