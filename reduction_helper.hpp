#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/device_malloc.h>
#include <thrust/device_free.h>

struct double_array_gpu {
	double_array_gpu(int dimx, int dimy, int dimz) {
		n = dimx * dimy * dimz;
		dev_ptr = thrust::device_malloc<double>(n);
	}
	~double_array_gpu() {
		thrust::device_free(dev_ptr);
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
