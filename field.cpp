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

#define _XOPEN_SOURCE 600

#include "pow.hpp"
#include "field.hpp"

#include <cstdlib>
#include <cstring>
#include <cmath>

#include <iostream>

using namespace std;

template <typename R>
void field<R>::construct(field_size &fs_, bool oop)
{
	fs = fs_;
	ldl = oop ? fs.n : 2*(fs.n/2+1);
	pldl = oop ? fs.n_pad_factor*fs.n : 2*((fs.n_pad_factor*fs.n)/2+1);

	size_t alloc_size = sizeof(R) * fs.total_padded_gridpoints;
	size_t c_alloc_size = 2*sizeof(R) * fs.total_padded_momentum_gridpoints;

	mdata = (typename fft_dft_c2r_3d_plan<R>::complex_t *) fft_malloc<R>(c_alloc_size);
	data = oop ? fft_malloc<R>(alloc_size) : (R *) mdata;

	// Out-of-place c2r implies FFTW_DESTROY_INPUT, data is saved to this array.
	mdata_saved = oop ? (typename fft_dft_c2r_3d_plan<R>::complex_t *) fft_malloc<R>(c_alloc_size) : 0;

	m2p_plan.construct(fs.n, fs.n, fs.n, mdata_saved ? mdata_saved : mdata, data, false);		
	p2m_plan.construct(fs.n, fs.n, fs.n, data, mdata, false);

	padded_m2p_plan.construct(fs.n_pad_factor*fs.n, fs.n_pad_factor*fs.n, fs.n_pad_factor*fs.n,
		mdata_saved ? mdata_saved : mdata, data, false);		
	padded_p2m_plan.construct(fs.n_pad_factor*fs.n, fs.n_pad_factor*fs.n, fs.n_pad_factor*fs.n,
		data, mdata, false);

	if (oop) memset(data, 0, alloc_size);
	memset(mdata, 0, c_alloc_size);
}

template <typename R>
field<R>::~field()
{
	if (!is_in_place()) {
		fft_free<R>((R *) mdata);
		fft_free<R>((R *) mdata_saved);
	}

	fft_free<R>(data);
}

template <typename R>
void field<R>::divby(R v)
{
	if (state == momentum) {

#ifdef _OPENMP
#pragma omp parallel for
#endif

		for (int idx = 0; idx < fs.total_momentum_gridpoints; ++idx) {
			mdata[idx][0] /= v;
			mdata[idx][1] /= v;
		}
	}
	else if (state == position) {

#ifdef _OPENMP
#pragma omp parallel for
#endif
		for (int x = 0; x < fs.n; ++x)
		for (int y = 0; y < fs.n; ++y)
		for (int z = 0; z < fs.n; ++z) {
			int idx = z + ldl*(y + fs.n*x);
			data[idx] /= v;
		}
	}
	else if (state == padded_momentum) {

#ifdef _OPENMP
#pragma omp parallel for
#endif

		for (int idx = 0; idx < fs.total_padded_momentum_gridpoints; ++idx) {
			mdata[idx][0] /= v;
			mdata[idx][1] /= v;
		}		
	}
	else if (state == padded_position) {

#ifdef _OPENMP
#pragma omp parallel for
#endif
		for (int x = 0; x < fs.n_pad_factor*fs.n; ++x)
		for (int y = 0; y < fs.n_pad_factor*fs.n; ++y)
		for (int z = 0; z < fs.n_pad_factor*fs.n; ++z) {
			int idx = z + pldl*(y + fs.n_pad_factor*fs.n*x);
			data[idx] /= v;
		}
	}
}

template <typename R>
void field<R>::switch_state(field_state state_, bool mmo)
{
	bool do_p2m = is_in_place() || !mmo;

	if (state_ == uninitialized) {
		state = uninitialized;
		return;
	}
	else if (state == uninitialized) {
		state = state_;
		return;
	}
	else if (state_ == state) {
		return;
	}
	else if (fs.n_pad_factor == 1 && (
		(state_ == padded_position && state == position) ||
		(state_ == position && state == padded_position) ||
		(state_ == padded_momentum && state == momentum) ||
		(state_ == momentum && state == padded_momentum))) {
		state = state_;
	}

	if (state == position) {
		state = momentum;
		if (do_p2m) p2m_plan.execute();
	}
	else if (state == padded_position) {
		state = padded_momentum;
		if (do_p2m) padded_p2m_plan.execute();
	}

switch_momentum_states:			
	if (state == momentum) {	
		if (state_ == padded_momentum || state_ == padded_position) {
			state = padded_momentum;
			pad_momentum_grid();
		}
		else if (state_ == position) {
			state = position;
			if (mdata_saved) memcpy(mdata_saved, mdata, 2*sizeof(R)*fs.total_momentum_gridpoints);
			m2p_plan.execute();
			divby(fs.total_gridpoints);
		}
	}

	if (state == padded_momentum) {
		if (state_ == momentum || state_ == position) {
			state = momentum;
			unpad_momentum_grid();
			
			if (state_ == position) {
				goto switch_momentum_states;
			}
		}
		else if (state_ == padded_position) {
			state = padded_position;
			if (mdata_saved) memcpy(mdata_saved, mdata, 2*sizeof(R)*fs.total_padded_momentum_gridpoints);
			padded_m2p_plan.execute();
			divby(fs.total_padded_gridpoints);
		}
	}
}

/*
 * The momentum-grid can be padded in place: The process works backwards.
 * CHANG: Support for padding has been removed in CUDA porting.
 */

template <typename R>
void field<R>::pad_momentum_grid()
{
	return;
}

template <typename R>
void field<R>::unpad_momentum_grid()
{
	return;
}

// Explicit instantiations
template class field<double>;
