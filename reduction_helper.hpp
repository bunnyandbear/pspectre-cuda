#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/device_malloc.h>
#include <thrust/device_free.h>

#ifdef DEBUG
#include <iostream>

void print_memory_usage();
#endif

struct double_array_gpu {
	double_array_gpu(int dimx, int dimy, int dimz) {
		n = dimx * dimy * dimz;
#ifdef DEBUG
		std::cout << "\nConstructing double array on gpu." << std::endl;
		std::cout << "Number of elements: " << n << std::endl;
		std::cout << "Memory usage before cudaMalloc:" << std::endl;
		print_memory_usage();
#endif
		dev_ptr = thrust::device_malloc<double>(n);
		cudaMemset(dev_ptr.get(), 0, sizeof(double)*n);
#ifdef DEBUG
		std::cout << "Memory usage after cudaMalloc:" << std::endl;
		print_memory_usage();
#endif
	}
	~double_array_gpu() {
#ifdef DEBUG
		std::cout << "Destructing double array on gpu." << std::endl;
		std::cout << "Number of elements: " << n << std::endl;
		std::cout << "Memory usage before cudaFree:" << std::endl;
		print_memory_usage();
#endif
		thrust::device_free(dev_ptr);
#ifdef DEBUG
		std::cout << "Memory usage after cudaFree:" << std::endl;
		print_memory_usage();
#endif
	}
	double sum() {
		return thrust::reduce(dev_ptr, dev_ptr + n, 0.0, thrust::plus<double>());
	}
	double *ptr() {
		return thrust::raw_pointer_cast(dev_ptr);
	}
private:
	int n;
	thrust::device_ptr<double> dev_ptr;
};
