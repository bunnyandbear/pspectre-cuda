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
#include "integrator.hpp"
#include "reduction_helper.hpp"

#include <cufftw.h>

using namespace std;

__global__ void integrator_kernel(fftw_complex *phi, IF_CHI_ARG(fftw_complex *chi,)
				  double *total_gradient_phi, IF_CHI_ARG(double *total_gradient_chi,)
				  double dp)
{
	int x = blockIdx.x;
	int y = blockIdx.y;
	int z = threadIdx.x;
	int px = x <= NGRIDSIZE/2 ? x : x - NGRIDSIZE;
	int py = y <= NGRIDSIZE/2 ? y : y - NGRIDSIZE;
	int pz = z;
	int idx = z + (NGRIDSIZE/2+1)*(y + NGRIDSIZE*x);

	double mom2 = pow2(dp)*(pow2(px) + pow2(py) + pow2(pz));
	mom2 *= (z == 0 || z == NGRIDSIZE/2) ? 1 : 2;

	total_gradient_phi[idx] += mom2*(pow2(phi[idx][0]) + pow2(phi[idx][1]));
	IF_CHI(total_gradient_chi[idx] += mom2*(pow2(chi[idx][0]) + pow2(chi[idx][1])));
}

template <typename R>
void integrator<R>::avg_gradients(field<R> &phi, IF_CHI_ARG(field<R> &chi,)
				  R &avg_gradient_phi IF_CHI_ARG(, R &avg_gradient_chi))
{
	phi.switch_state(momentum);
	IF_CHI(chi.switch_state(momentum));

	auto total_gradient_phi_arr = double_array_gpu(NGRIDSIZE, NGRIDSIZE, NGRIDSIZE/2+1);
#if ENABLE_CHI != 0
	auto total_gradient_chi_arr = double_array_gpu(NGRIDSIZE, NGRIDSIZE, NGRIDSIZE/2+1);
#endif
	dim3 num_blocks(NGRIDSIZE, NGRIDSIZE);
	dim3 num_threads(NGRIDSIZE/2+1, 1);
	integrator_kernel<<<num_blocks, num_threads>>>(phi.mdata.ptr, IF_CHI_ARG(chi.mdata.ptr,)
						       total_gradient_phi_arr.ptr(),
						       IF_CHI_ARG(total_gradient_chi_arr.ptr(),)
						       MP_DP);

	R total_gradient_phi = total_gradient_phi_arr.sum();
	IF_CHI(R total_gradient_chi = total_gradient_chi_arr.sum());

	// Divide by total_gridpoints again to get *average* squared gradient and *average* potential energy.
	avg_gradient_phi = total_gradient_phi/pow<2, R>(NTOTAL_GRIDPOINTS);
#if ENABLE_CHI != 0
	avg_gradient_chi = total_gradient_chi/pow<2, R>(NTOTAL_GRIDPOINTS);
#endif
}

// Explicit instantiations
template class integrator<double>;
