#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <immintrin.h>
#include "rdtsc.h"

void kernel
        ( int sizeKernel, int sizeMatrix, int sizeResult, double* restrict matrix, double* restrict result, double* restrict filter )
{
    /*
     * Assumptions:
     * matrix - stored row-wise
     * res - stored row-wise
     * filter - stored row-wise
    */

    register __m256d m0,m1,m2,m3,m4,m5,m6,m7;
    register __m256d r1, r2, r3, r4;

    register __m256d f0,f1,f2,f3;
    f0 = _mm256_loadu_pd(&filter[0]);
    f1 = _mm256_loadu_pd(&filter[4]);
    f2 = _mm256_loadu_pd(&filter[8]);
    f3 = _mm256_loadu_pd(&filter[12]);


    for(int i=0 ;i<sizeMatrix-sizeKernel+1;i++){
        for(int j=0; j<sizeMatrix;j=j+4){
            r1 = _mm256_setzero_pd ();
            r2 = _mm256_setzero_pd ();
            r3 = _mm256_setzero_pd ();
            r4 = _mm256_setzero_pd ();

            if(j!=sizeMatrix-sizeKernel){

            m0 = _mm256_loadu_pd(&matrix[(i+0)*sizeMatrix+j + 0]);
            m1 = _mm256_loadu_pd(&matrix[(i+1)*sizeMatrix+j + 0]);
            m2 = _mm256_loadu_pd(&matrix[(i+2)*sizeMatrix+j + 0]);
            m3 = _mm256_loadu_pd(&matrix[(i+3)*sizeMatrix+j + 0]);
            m4 = _mm256_loadu_pd(&matrix[(i+0)*sizeMatrix+j + 1]);
            m5 = _mm256_loadu_pd(&matrix[(i+1)*sizeMatrix+j + 1]);
            m6 = _mm256_loadu_pd(&matrix[(i+2)*sizeMatrix+j + 1]);
            m7 = _mm256_loadu_pd(&matrix[(i+3)*sizeMatrix+j + 1]);



            //Position 0
            r2 = _mm256_fmadd_pd(m0,f0,r2);
            r2 = _mm256_fmadd_pd(m1,f1,r2);
            r2 = _mm256_fmadd_pd(m2,f2,r2);
            r2 = _mm256_fmadd_pd(m3,f3,r2);

            r2 = _mm256_hadd_pd(r2,r2);
            m0 = _mm256_permute2f128_pd(r2 , r2, 1|	(2	<<	4));
            r2 = _mm256_add_pd(r2, m0);

            //Position 1
            r1 = _mm256_fmadd_pd(m4,f0,r1);
            r1 = _mm256_fmadd_pd(m5,f1,r1);
            r1 = _mm256_fmadd_pd(m6,f2,r1);
            r1 = _mm256_fmadd_pd(m7,f3,r1);

            r1 = _mm256_hadd_pd(r1,r1);
            m0 = _mm256_permute2f128_pd(r1 , r1, 1|	(2	<<	4));
            r1 = _mm256_add_pd(r1, m0);


            m0 = _mm256_loadu_pd(&matrix[(i+0)*sizeMatrix+j + 2]);
            m1 = _mm256_loadu_pd(&matrix[(i+1)*sizeMatrix+j + 2]);
            m2 = _mm256_loadu_pd(&matrix[(i+2)*sizeMatrix+j + 2]);
            m3 = _mm256_loadu_pd(&matrix[(i+3)*sizeMatrix+j + 2]);
            m4 = _mm256_loadu_pd(&matrix[(i+0)*sizeMatrix+j + 3]);
            m5 = _mm256_loadu_pd(&matrix[(i+1)*sizeMatrix+j + 3]);
            m6 = _mm256_loadu_pd(&matrix[(i+2)*sizeMatrix+j + 3]);
            m7 = _mm256_loadu_pd(&matrix[(i+3)*sizeMatrix+j + 3]);

            //Position 2
            r3 = _mm256_fmadd_pd(m0,f0,r3);
            r3 = _mm256_fmadd_pd(m1,f1,r3);
            r3 = _mm256_fmadd_pd(m2,f2,r3);
            r3 = _mm256_fmadd_pd(m3,f3,r3);

            r3 = _mm256_hadd_pd(r3,r3);
            m0 = _mm256_permute2f128_pd(r3 , r3, 1|	(2	<<	4));
            r3 = _mm256_add_pd(r3, m0);

            //Position 3
            r4 = _mm256_fmadd_pd(m4,f0,r4);
            r4 = _mm256_fmadd_pd(m5,f1,r4);
            r4 = _mm256_fmadd_pd(m6,f2,r4);
            r4 = _mm256_fmadd_pd(m7,f3,r4);

            r4 = _mm256_hadd_pd(r4,r4);
            m0 = _mm256_permute2f128_pd(r4 , r4, 1|	(2	<<	4));
            r4 = _mm256_add_pd(r4, m0);

            m0 = _mm256_setzero_pd ();
            m1 = _mm256_setzero_pd ();
            m2 = _mm256_setzero_pd ();

            m0 = _mm256_shuffle_pd(r1,r2,0	|	(0	<<	1)	|	(0	<<	2)	|	(0	<<	3));
            m1 = _mm256_shuffle_pd(r4,r3,0	|	(0	<<	1)	|	(0	<<	2)	|	(0	<<	3));
            m2 = _mm256_shuffle_pd(m0,m1,1	|	(1	<<	1)	|	(0	<<	2)	|	(0	<<	3));



            _mm256_store_pd(&result[i*sizeResult+j],m2);
            }
            else{
                m0 = _mm256_loadu_pd(&matrix[(i+0)*sizeMatrix+j + 0]);
                m1 = _mm256_loadu_pd(&matrix[(i+1)*sizeMatrix+j + 0]);
                m2 = _mm256_loadu_pd(&matrix[(i+2)*sizeMatrix+j + 0]);
                m3 = _mm256_loadu_pd(&matrix[(i+3)*sizeMatrix+j + 0]);

                r2 = _mm256_fmadd_pd(m0,f0,r2);
                r2 = _mm256_fmadd_pd(m1,f1,r2);
                r2 = _mm256_fmadd_pd(m2,f2,r2);
                r2 = _mm256_fmadd_pd(m3,f3,r2);

                r2 = _mm256_hadd_pd(r2,r2);
                m0 = _mm256_permute2f128_pd(r2 , r2, 1|	(2	<<	4));
                r2 = _mm256_add_pd(r2, m0);

                result[i*sizeResult + j] = r2[0];
            }
        }

    }


}

int main(){

    double *matrix ;
    double *filter ;
    double *result ;

    int sizeMatrix = 1024;
    int sizeKernel = 4;
    int padding = 0;
    int strides = 1;

    long long sum1 = 0;
    int runs = 1;
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

    free(matrix);
    free(result);
    free(filter);


}

