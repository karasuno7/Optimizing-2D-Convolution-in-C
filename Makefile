all:
	# Image Sizes 1024/2048/4096:
	gcc -O2 conv_mpi.c -o conv_mpi_l.x -std=c99 -DN=22687 -DNUM_A=109410 -fopenmp
	./conv_mpi_l.x 2048 35 100

	# Image Sizes 512/256/128:
	gcc -O2 conv_mpi.c -o conv_mpi_m.x -std=c99 -DN=9877 -DNUM_A=51946 -fopenmp
	./conv_mpi_m.x 512 28 100

	# Image Sizes 8/16/32/64:
	gcc -O2 conv_mpi.c -o conv_mpi.x -std=c99 -DN=6474 -DNUM_A=25144 -fopenmp
	./conv_mpi.x 64 8 100

	# Optimised simd:
	gcc -O3 -Wall -mavx -mavx2 -mfma main_optimized.c -o main_optimized.x -std=c99
	./main_optimized.x 1024 100

	# Basecode :
	gcc -Wall -O3 base.c -o base.x -std=c99
	./base.x 1024 100

cleanup:
	rm -rf *~
	rm -rf *.x