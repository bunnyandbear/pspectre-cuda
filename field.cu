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
#include <thrust/device_malloc.h>
#include <thrust/device_free.h>
#include <thrust/fill.h>

using namespace std;

template <typename R>
void field<R>::construct(field_size &fs_)
{
	fs = fs_;
	ldl = 2*(fs.n/2+1);

	dev_ptr = thrust::device_malloc<fftw_complex>(fs.total_momentum_gridpoints);
	fftw_complex *raw_ptr = thrust::raw_pointer_cast(dev_ptr);
	mdata = gpu_array_accessor_fftw_complex(raw_ptr);
	data = gpu_array_accessor_double((double *) raw_ptr);
	fill0();

	m2p_plan.construct(fs.n, fs.n, fs.n, mdata.ptr, data.ptr, false);
	p2m_plan.construct(fs.n, fs.n, fs.n, data.ptr, mdata.ptr, false);
}

template <typename R>
field<R>::~field()
{
	thrust::device_free(dev_ptr);
}

/* (x*y) * (z)
 * (n^2) * (n/2+1)
 * BLK     THR
 *         NO-PADDING
 */
__global__ void momentum_divby_kernel(fftw_complex *mdata, double v)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	mdata[idx][0] /= v;
	mdata[idx][1] /= v;
}

/* (x*y) * (z)
 * (n^2) * (2*(n/2+1))
 * BLK     THR
 *         PADDED from n to (2*(n/2+1))
 */
__global__ void position_divby_kernel(double *data, double v, int ldl)
{
	int idx = ldl * blockIdx.x + threadIdx.x;
	data[idx] /= v;
}

template <typename R>
void field<R>::divby(R v)
{
	int n = fs.n;
	if (state == momentum) {
		momentum_divby_kernel<<<n*n, n/2+1>>>(mdata.ptr, v);
	} else if (state == position) {
		position_divby_kernel<<<n*n, n>>>(data.ptr, v, ldl);
	}
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
		p2m_plan.execute();
	} else if ((state == momentum) && (state_ == position)) {
		state = position;
		m2p_plan.execute();
		divby(fs.total_gridpoints);
	}
}

template <typename R>
void field<R>::fill0()
{
	auto ptr = thrust::device_ptr<double>(data.ptr);
	thrust::fill(ptr, ptr + 2*fs.total_momentum_gridpoints, 0.0);
}

// Explicit instantiations
template class field<double>;
