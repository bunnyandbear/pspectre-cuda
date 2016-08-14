#include <cuda_runtime.h>
#include <cufftw.h>

class fftw_complex_accessor {
	fftw_complex *ptr;
	int c_idx;
public:
	fftw_complex_accessor(fftw_complex *p, int i = 0) : ptr(p), c_idx(i) {}
	fftw_complex_accessor operator[](int i) {
		return fftw_complex_accessor(ptr, i);
	}
	operator double() {
		fftw_complex val;
		cudaMemcpy(&val, ptr, sizeof(fftw_complex), cudaMemcpyDefault);
		return val[c_idx];
	}
	void operator=(double rhs) {
		double *p = ((double *) ptr) + c_idx;
		cudaMemcpy(p, &rhs, sizeof(double), cudaMemcpyDefault);
	}
	void operator+=(double rhs) {
		*this = *this + rhs;
	}
};

class gpu_array_accessor_double {
	int i;
public:
	double *ptr;
	gpu_array_accessor_double(double *p, int i_ = 0) : ptr(p), i(i_) {}
	gpu_array_accessor_double operator[](int i) {
		return gpu_array_accessor_double(ptr, i);
	}
	operator double() {
		double val;
		cudaMemcpy(&val, ptr+i, sizeof(double), cudaMemcpyDefault);
		return val;
	}
	void operator=(double rhs) {
		cudaMemcpy(ptr + i, &rhs, sizeof(double), cudaMemcpyDefault);
	}
};

class gpu_array_accessor_fftw_complex {
public:
	fftw_complex *ptr;
	gpu_array_accessor_fftw_complex(fftw_complex *p) : ptr(p) {}
	fftw_complex_accessor operator[](int i) {
		return fftw_complex_accessor(ptr + i);
	}
};
