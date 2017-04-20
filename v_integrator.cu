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

#include "reduction_helper.hpp"
#include "v_integrator.hpp"
#include <iostream>
#include <fstream>
#include <iomanip>

using namespace std;

/**
 * Returns the value of the field potential at a point given the values of the fields at that point.
 * The field values are sent in program units, and the potential is returned in program units.
 * This is equation 6.5 from the LatticeEasy manual.
 */

__global__ void v_integrator_kernel(double *phi, double *chi,
				    double *total_V,
				    double a_t)
{
	int x = blockIdx.x;
	int y = blockIdx.y;
	int z = threadIdx.x;
	int coeffx = 2 + 2 * (x & 0x1);
	int coeffy = 2 + 2 * (y & 0x1);
	int coeffz = 2 + 2 * (z & 0x1);
	int ldl = 2*(NGRIDSIZE/2+1);
	int idx = z + ldl*(y + NGRIDSIZE*x);
	int idx_V = z + NGRIDSIZE*(y + NGRIDSIZE*x);

	total_V[idx_V] = coeffx * coeffy * coeffz *
		model_params::V(phi[idx], chi[idx], a_t);
}

// Integrate the potential. Returns the average value.
template <typename R>
R v_integrator<R>::integrate(field<R> &phi, field<R> &chi, R a_t)
{
	phi.switch_state(position);
	chi.switch_state(position);

	auto total_V_arr = double_array_gpu(NGRIDSIZE, NGRIDSIZE, NGRIDSIZE);
	dim3 nr_blocks(NGRIDSIZE, NGRIDSIZE);
	dim3 nr_threads(NGRIDSIZE, 1);
	v_integrator_kernel<<<nr_blocks, nr_threads>>>(phi.data.ptr, chi.data.ptr,
						       total_V_arr.ptr(),
						       a_t);
	double total_V = total_V_arr.sum();

	return total_V / (3.0 * 3 * 3 * NTOTAL_GRIDPOINTS);
}

// Explicit instantiations
template class v_integrator<double>;
