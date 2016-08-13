#include <stdio.h>
#include "reduction_helper.hpp"

static void *alloc_gpu_memory(int size)
{
	void *p = NULL;
	cudaError_t err = cudaMalloc(&p, size);
	if (err != cudaSuccess) {
		return NULL;
	}
	return p;
}

static double *new_double_array_on_gpu(int dimx, int dimy, int dimz)
{
	double *p = (double *) alloc_gpu_memory(sizeof(double) * dimx * dimy * dimz);
	if (p == NULL) {
		printf("Failed to alloc GPU memory. Program will soon crash.");
	}
	return p;
}

double_array_gpu::double_array_gpu(int x, int y, int z) : dimx(x), dimy(y), dimz(z)
{
	array = new_double_array_on_gpu(x, y, z);
}

double_array_gpu::~double_array_gpu()
{
	cudaFree(array);
}

double double_array_gpu::sum()
{
	return 0;
}
