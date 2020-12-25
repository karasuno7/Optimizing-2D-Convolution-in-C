# Optimizing-2D-Convolution-in-C
A SIMD and OpenMP based optimization for naive 2D convolution

The codebase has the following files:
1. base.c (Base Code)
2. Main.c (Basic Kernel Implementation)
3. Main_optimized.c (Optimized Kernel Implementation)
4. Conv_mpi.c (OpenMP-based Parallelized Implementation) 5. Makefile (Compiles and Runs all experiments)


**Compile and Run:**

Inside the MakeFile instructions on how to compile and run each model and corresponding input parameters is given:

**For OpenMP**:\
./filename.x img_size threads runs

**For Other Models:**\
./filename.x img_size runs

**Correctness:**

For checking the correctness of the implementation:

- uncomment the lines 190-195 in main_optimized.c
- uncomment the lines 73-78 in base.c
- uncomment the lines 76-81 in conv_mpi.c

**Run:**
./filename.x img_size 1 (for main_optimised.c and base.c)

./filename.x img_size 1 1 (for conv_mpi.c)


The output should be img_size-3 x img_size matrix-3 with each value = 32.0 double floating point.


Explanation:\
Image matrix is an all ones matrix.\
Kernel is a 4 x 4 matrix with all values = 2.0\
Each output cell in output matrix = 4 x 2 x 4 = 32.0
