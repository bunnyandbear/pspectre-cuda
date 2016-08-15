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

using namespace std;

/**
 * Returns the value of the field potential at a point given the values of the fields at that point.
 * The field values are sent in program units, and the potential is returned in program units.
 * This is equation 6.5 from the LatticeEasy manual.
 */

__device__ double V(double phi, double chi, double a_t)
{
	double tophys = 1./RESCALE_A * pow(a_t, -RESCALE_R);
	double phi_phys = tophys * phi;
	double chi_phys = tophys * chi;
	return pow2(RESCALE_A / RESCALE_B) * pow(a_t, -2. * RESCALE_S + 2. * RESCALE_R) *
		(
			(
				(MD_E_PHI != 0) ?
				MD_C_PHI*pow2(MD_S_PHI)*(pow(1.0 + pow2(phi_phys/MD_S_PHI), MD_E_PHI) - 1.0) :
				0.5*pow2(M_PHI * phi_phys)
				) +
			(
				(MD_E_CHI != 0) ?
				MD_C_CHI*pow2(MD_S_CHI)*(pow(1.0 + pow2(chi_phys/MD_S_CHI), MD_E_CHI) - 1.0) :
				0.5*pow2(M_CHI * chi_phys)
				) +
			0.25*LAMBDA_PHI*pow4(phi_phys) +
			0.25*LAMBDA_CHI*pow4(chi_phys) +
			0.5*pow2(MP_G * phi_phys * chi_phys) +
			GAMMA_PHI*pow6(phi_phys)/6.0 +
			GAMMA_CHI*pow6(chi_phys)/6.0
			);
}

__global__ void v_integrator_kernel(double *phi, double *chi,
				    double *total_V,
				    double a_t, int n)
{
	int x = blockIdx.x;
	int y = blockIdx.y;
	int z = threadIdx.x;
	int ldl = 2*(n/2+1);
	int idx = z + ldl*(y + n*x);
	int idx_V = z + n*(y + n*x);

	total_V[idx_V] = V(phi[idx], chi[idx], a_t);
}

// Integrate the potential. Returns the average value.
template <typename R>
R v_integrator<R>::integrate(field<R> &phi, field<R> &chi, R a_t)
{
	phi.switch_state(position);
	chi.switch_state(position);

	auto total_V_arr = double_array_gpu(fs.n, fs.n, fs.n);
	dim3 nr_blocks(fs.n, fs.n);
	dim3 nr_threads(fs.n, 1);
	v_integrator_kernel<<<nr_blocks, nr_threads>>>(phi.data.ptr, chi.data.ptr,
						       total_V_arr.ptr(),
						       a_t, fs.n);
	double total_V = total_V_arr.sum();

	return total_V / fs.total_gridpoints;
}

// Explicit instantiations
template class v_integrator<double>;
