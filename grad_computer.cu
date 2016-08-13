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

#include "grad_computer.hpp"
#include <cufftw.h>

using namespace std;

__global__ void grad_computer_kernel(
	fftw_complex *phi, fftw_complex *chi,
	fftw_complex *phigradx, fftw_complex *chigradx,
	fftw_complex *phigrady, fftw_complex *chigrady,
	fftw_complex *phigradz, fftw_complex *chigradz,
	int n, double dp)
{
	int x = blockIdx.x;
	int y = blockIdx.y;
	int z = threadIdx.x;
	int px = x <= n/2 ? x : x - n;
	int py = y <= n/2 ? y : y - n;
	int pz = z;
	int idx = z + (n/2+1)*(y + n * x);

	// The Fourier transform conventions require
	// the normalization of the position-space values by 1/N^3.
	phigradx[idx][0] = -1. * px * dp * phi[idx][1];
	phigradx[idx][1] = px * dp * phi[idx][0];

	chigradx[idx][0] = -1. * px * dp * chi[idx][1];
	chigradx[idx][1] = px * dp * chi[idx][0];

	phigrady[idx][0] = -1. * py * dp * phi[idx][1];
	phigrady[idx][1] = py * dp * phi[idx][0];

	chigrady[idx][0] = -1. * py * dp * chi[idx][1];
	chigrady[idx][1] = py * dp * chi[idx][0];

	phigradz[idx][0] = -1. * pz * dp * phi[idx][1];
	phigradz[idx][1] = pz * dp * phi[idx][0];

	chigradz[idx][0] = -1. * pz * dp * chi[idx][1];
	chigradz[idx][1] = pz * dp * chi[idx][0];
}

template <typename R>
void grad_computer<R>::compute(field_state final_state)
{
	phi.switch_state(momentum);
	chi.switch_state(momentum);

	phigradx.switch_state(uninitialized);
	phigradx.switch_state(momentum);
	
	chigradx.switch_state(uninitialized);
	chigradx.switch_state(momentum);
	
	phigrady.switch_state(uninitialized);
	phigrady.switch_state(momentum);
	
	chigrady.switch_state(uninitialized);
	chigrady.switch_state(momentum);

	phigradz.switch_state(uninitialized);
	phigradz.switch_state(momentum);
	
	chigradz.switch_state(uninitialized);
	chigradz.switch_state(momentum);

	dim3 num_blocks(fs.n, fs.n);
	dim3 num_threads(fs.n/2+1, 1);
	grad_computer_kernel<<<num_blocks, num_threads>>>(
		phi.mdata, chi.mdata,
		phigradx.mdata, chigradx.mdata,
		phigrady.mdata, chigrady.mdata,
		phigradz.mdata, chigradz.mdata,
		fs.n, mp.dp);

	phigradx.switch_state(final_state);
	chigradx.switch_state(final_state);
	
	phigrady.switch_state(final_state);
	chigrady.switch_state(final_state);

	phigradz.switch_state(final_state);
	chigradz.switch_state(final_state);	
}

// Explicit instantiations
template class grad_computer<double>;
