/*
 * SpectRE - A Spectral Code for Reheating
 * Copyright (C) 2009-2010 Hal Finkel, Nathaniel Roth and Richard Easther
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES,
 * INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY
 * AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL
 * THE AUTHORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 * WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
 * OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
 * ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "pow.hpp"
#include "field.hpp"

#include <cstdlib>
#include <cstring>
#include <cmath>

#include <iostream>
#include <thrust/device_free.h>
#include <thrust/fill.h>

using namespace std;

void print_memory_usage();

template <typename R>
void field<R>::construct(fft_worker<R> &fft_plans_)
{
	ldl = 2*(NGRIDSIZE/2+1);
	fft_plans = &fft_plans_;

#ifdef DEBUG
	cout << "\nConstructing field " << (name ? name : "unknown") << endl;
	cout << "Number of grid points: " << NGRIDSIZE << endl;
	cout << "Memory usage before cudaMalloc:" << endl;
	print_memory_usage();
#endif
	cudaError_t ret = cudaMalloc(&raw_ptr, NALLOC_SIZE);
	if (ret != cudaSuccess) {
		cout << "cudaMalloc() failed. GPUassert: "
		     << cudaGetErrorString(ret) << endl;
	}
#ifdef DEBUG
	cout << "Memory usage after cudaMalloc:" << endl;
	print_memory_usage();
#endif
	mdata = gpu_array_accessor_fftw_complex((fftw_complex *) raw_ptr);
	data = gpu_array_accessor_double(raw_ptr);
	fill0();

	// construct fft_plans (first time only)
	fft_plans->construct(NGRIDSIZE, NGRIDSIZE, NGRIDSIZE, data.ptr, mdata.ptr, false);
}

template <typename R>
field<R>::~field()
{
#ifdef DEBUG
	cout << "Destructing field " << (name ? name : "unknown") << endl;
	cout << "Memory usage before cudaFree:" << endl;
	print_memory_usage();
#endif
	cudaError_t ret = cudaFree(raw_ptr);
	if (ret != cudaSuccess) {
		cout << "cudaFree() failed. GPUassert: "
		     << cudaGetErrorString(ret) << endl;
	}
#ifdef DEBUG
	cout << "Memory usage after cudaFree:" << endl;
	print_memory_usage();
#endif
}

__global__ void divby_kernel(double *data, double v, int n)
{
	int x = blockIdx.x;
	int y = blockIdx.y;
	int z = threadIdx.x;
	int ldl = 2*(n/2+1);
	int idx = z + ldl*(y + n*x);

	data[idx] /= v;
}

template <typename R>
void field<R>::divby(R v)
{
	dim3 nr_blocks(NGRIDSIZE, NGRIDSIZE);
	dim3 nr_threads(ldl, 1);
	divby_kernel<<<nr_blocks, nr_threads>>>(data.ptr, v, NGRIDSIZE);
}

template <typename R>
void field<R>::switch_state(field_state state_)
{
	if (state_ == uninitialized) {
		state = uninitialized;
	} else if (state == uninitialized) {
		state = state_;
	} else if ((state == position) && (state_ == momentum)) {
		state = momentum;
		fft_plans->execute_p2m(data.ptr, mdata.ptr);
	} else if ((state == momentum) && (state_ == position)) {
		state = position;
		fft_plans->execute_m2p(mdata.ptr, data.ptr);
		divby(NTOTAL_GRIDPOINTS);
	}
}

template <typename R>
void field<R>::fill0()
{
	cudaError_t ret = cudaMemset(raw_ptr, 0, NALLOC_SIZE);
	if (ret != cudaSuccess) {
		cout << "fill0: cudaMemset() failed. GPUassert: "
		     << cudaGetErrorString(ret) << endl;
	}
}

template <typename R>
void field<R>::upload(fftw_complex *fld)
{
	cudaError_t ret = cudaMemcpy(raw_ptr, fld, NALLOC_SIZE, cudaMemcpyDefault);
	if (ret != cudaSuccess) {
		cout << "field::upload cudaMemcpy fail. field name = "
		     << (name ? name : "unknown") << endl;
	}
}

// Explicit instantiations
template class field<double>;
