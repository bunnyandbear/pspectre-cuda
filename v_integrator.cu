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

#define pow2(x) ((x)*(x))
#define pow4(x) ((x)*(x)*(x)*(x))
#define pow6(x) ((x)*(x)*(x)*(x)*(x)*(x))
__device__ double V(double phi, double chi, double a_t, model_params<double> mp)
{
	double tophys = 1./mp.rescale_A * pow(a_t, -mp.rescale_r);
	double phi_phys = tophys * phi;
	double chi_phys = tophys * chi;
	return pow2(mp.rescale_A / mp.rescale_B) * pow(a_t, -2. * mp.rescale_s + 2. * mp.rescale_r) *
		(
			(
				(mp.md_e_phi != 0) ?
				mp.md_c_phi*pow2(mp.md_s_phi)*(pow(1.0 + pow2(phi_phys/mp.md_s_phi), mp.md_e_phi) - 1.0) :
				0.5*pow2(mp.m_phi * phi_phys)
				) +
			(
				(mp.md_e_chi != 0) ?
				mp.md_c_chi*pow2(mp.md_s_chi)*(pow(1.0 + pow2(chi_phys/mp.md_s_chi), mp.md_e_chi) - 1.0) :
				0.5*pow2(mp.m_chi * chi_phys)
				) +
			0.25*mp.lambda_phi*pow4(phi_phys) +
			0.25*mp.lambda_chi*pow4(chi_phys) +
			0.5*pow2(mp.g * phi_phys * chi_phys) +
			mp.gamma_phi*pow6(phi_phys)/6.0 +
			mp.gamma_chi*pow6(chi_phys)/6.0
			);
}

__global__ void v_integrator_kernel(model_params<double> mp,
				    double *phi, double *chi,
				    double *total_V,
				    double a_t, int n)
{
	int x = blockIdx.x;
	int y = blockIdx.y;
	int z = threadIdx.x;
	int ldl = 2*(n/2+1);
	int idx = z + ldl*(y + n*x);
	int idx_V = z + n*(y + n*x);
	/* int fz = 2 * (1 + z & 1); */
	/* int fy = 2 * (1 + y & 1); */
	/* int fx = 2 * (1 + x & 1); */
	/* int f = fz * fy * fx; */

	total_V[idx_V] = V(phi[idx], chi[idx], a_t, mp);
}

/* __global__ void v_integrator_fix_y(double *total_V, int n) */
/* { */
/* 	int x = blockIdx.x; */
/* 	int y = blockIdx.y; */
/* 	int z = threadIdx.x; */
/* 	int idx_V = z + n*(y + n*x); */
/* 	int f = 2 * (1 + y & 1); */
/* 	total_V[idx_V] *= f; */
/* } */

/* __global__ void v_integrator_fix_x(double *total_V, int n) */
/* { */
/* 	int x = blockIdx.x; */
/* 	int y = blockIdx.y; */
/* 	int z = threadIdx.x; */
/* 	int idx_V = z + n*(y + n*x); */
/* 	int f = 2 * (1 + x & 1); */
/* 	total_V[idx_V] *= f; */
/* } */

// Integrate the potential using Simpson's rule, and assume periodic boundaries.
// Returns the average value.
template <typename R>
R v_integrator<R>::integrate(field<R> &phi, field<R> &chi, R a_t)
{
	phi.switch_state(position);
	chi.switch_state(position);

	auto total_V_arr = double_array_gpu(fs.n, fs.n, fs.n);
	dim3 nr_blocks(fs.n, fs.n);
	dim3 nr_threads(fs.n, 1);
	v_integrator_kernel<<<nr_blocks, nr_threads>>>(mp,
						       phi.data.ptr, chi.data.ptr,
						       total_V_arr.ptr(),
						       a_t, fs.n);
	/* v_integrator_fix_y<<<nr_blocks, nr_threads>>>(total_V_arr.ptr(), fs.n); */
	/* v_integrator_fix_x<<<nr_blocks, nr_threads>>>(total_V_arr.ptr(), fs.n); */
	double total_V = total_V_arr.sum();

	// The normalizing factor for Simpson's rule iterated over 3 dimensions.
	/* return (total_V * 1./(3.*3.*3.)) / fs.total_gridpoints; */
	return total_V / fs.total_gridpoints;
}

// Explicit instantiations
template class v_integrator<double>;
