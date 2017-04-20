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
#include "reduction_helper.hpp"
#include "verlet.hpp"
#include <cufftw.h>

using namespace std;

/**
 * This is where the equations of motion for the fields are actually evaluated.
 * The first and second time derivatives of the fields are computed in accordance
 * with the Klein-Gordon equation, which is written in program units and
 * transformed to momentum-space. Note that the choice of program units has eliminated
 * the first-time-derivative term from the second-time-derivative equation.
 */

__device__ double dphidotdt(double phi,
			    IF_CHI_ARG(double chi,)
			    IF_CHI_ARG(double chi2phi,)
			    IF_PHI3_ARG(double phi3,)
			    IF_PHI5_ARG(double phi5,)
			    IF_MD_PHI_ARG(double phi_md,)
			    double a_t, double adot_t,
			    double addot_t, double mom2)
{
	double md_phi = IF_MD_PHI(pow(a_t, 2. * RESCALE_R) * phi_md)
		IF_NOT_MD_PHI(pow2(M_PHI) * pow(a_t, 2. * RESCALE_R) * phi);
	return - pow(a_t, -2. * RESCALE_S - 2.) * mom2 * phi
		+ RESCALE_R * ((RESCALE_S - RESCALE_R + 2) * pow2(adot_t/a_t) + addot_t/a_t)*phi
		- pow(a_t, -2.*RESCALE_S - 2. * RESCALE_R) / pow2(RESCALE_B) * (
			md_phi
			IF_PHI3(+ LAMBDA_PHI/pow2(RESCALE_A) * phi3)
			IF_CHI(+ pow2(MP_G/RESCALE_A)*chi2phi)
			IF_PHI5(+ GAMMA_PHI/pow4(RESCALE_A) * pow(a_t, -2. * RESCALE_R) * phi5));
}

#if ENABLE_CHI != 0
__device__ double dchidotdt(double phi,
			    double chi,
			    double phi2chi,
			    IF_CHI3_ARG(double chi3,)
			    IF_CHI5_ARG(double chi5,)
			    IF_MD_CHI_ARG(double chi_md,)
			    double a_t, double adot_t,
			    double addot_t, double mom2)
{
	double md_chi = IF_MD_CHI(pow(a_t, 2. * RESCALE_R) * chi_md)
		IF_NOT_MD_CHI(pow2(M_CHI) * pow(a_t, 2. * RESCALE_R) * chi);
	return - pow(a_t, -2. * RESCALE_S - 2.) * mom2 * chi 
		+ RESCALE_R * ((RESCALE_S - RESCALE_R + 2) * pow2(adot_t/a_t) + addot_t/a_t) * chi
		- pow(a_t, -2.*RESCALE_S - 2. * RESCALE_R) / pow2(RESCALE_B) * (
			md_chi
			IF_CHI3(+ LAMBDA_CHI/pow2(RESCALE_A) * chi3)
			+ pow2(MP_G/RESCALE_A)*phi2chi
			IF_CHI5(+ GAMMA_CHI/pow4(RESCALE_A) * pow(a_t, -2. * RESCALE_R) * chi5));
}
#endif

__global__ void verlet_init_kernel(fftw_complex *phi_p,
				   IF_CHI_ARG(fftw_complex *chi_p,)
				   IF_CHI_ARG(fftw_complex *chi2phi_p,)
				   IF_CHI_ARG(fftw_complex *phi2chi_p,)
				   IF_PHI3_ARG(fftw_complex *phi3_p,)
				   IF_CHI3_ARG(fftw_complex *chi3_p,)
				   IF_PHI5_ARG(fftw_complex *phi5_p,)
				   IF_CHI5_ARG(fftw_complex *chi5_p,)
				   IF_MD_PHI_ARG(fftw_complex *phi_md_p,)
				   IF_MD_CHI_ARG(fftw_complex *chi_md_p,)
				   fftw_complex *phiddot,
				   IF_CHI_ARG(fftw_complex *chiddot,)
				   double a, double adot, double addot)
{
	int x = blockIdx.x;
	int y = blockIdx.y;
	int z = threadIdx.x;
	int px = x <= NGRIDSIZE/2 ? x : x - NGRIDSIZE;
	int py = y <= NGRIDSIZE/2 ? y : y - NGRIDSIZE;
	int pz = z;
	int idx = z + (NGRIDSIZE/2+1)*(y + NGRIDSIZE*x);
	double mom2 = pow2(MP_DP)*(pow2(px) + pow2(py) + pow2(pz));

	#pragma unroll
	for (int c = 0; c < 2; ++c) {
		phiddot[idx][c] = dphidotdt(phi_p[idx][c],
					    IF_CHI_ARG(chi_p[idx][c],)
					    IF_CHI_ARG(chi2phi_p[idx][c],)
					    IF_PHI3_ARG(phi3_p[idx][c],)
					    IF_PHI5_ARG(phi5_p[idx][c],)
					    IF_MD_PHI_ARG(phi_md_p[idx][c],)
					    a, adot,
					    addot, mom2);
#if ENABLE_CHI != 0
		chiddot[idx][c] = dchidotdt(phi_p[idx][c], chi_p[idx][c],
					    phi2chi_p[idx][c],
					    IF_CHI3_ARG(chi3_p[idx][c],)
					    IF_CHI5_ARG(chi5_p[idx][c],)
					    IF_MD_CHI_ARG(chi_md_p[idx][c],)
					    a, adot,
					    addot, mom2));
#endif
	}
}

template <typename R>
void verlet<R>::initialize()
{
	R avg_gradient_phi = 0.0;
	IF_CHI(R avg_gradient_chi = 0.0;)

	integrator<R>::avg_gradients(phi, IF_CHI_ARG(chi,)
				     avg_gradient_phi IF_CHI_ARG(, avg_gradient_chi));

	R avg_V = vi.integrate(phi, IF_CHI_ARG(chi,) ts.a);

	ts.addot = addot = mp.adoubledot(ts.t, ts.a, ts.adot,
					 avg_gradient_phi, IF_CHI_ARG(avg_gradient_chi,) avg_V);
	ddptdt = -RESCALE_S/RESCALE_B * pow(ts.a, -RESCALE_S - 1.) * ts.adot;

	nlt.transform(phi, IF_CHI_ARG(chi,) ts.a);

	phi.switch_state(momentum);
	phidot.switch_state(momentum);
	phiddot.switch_state(momentum);

	IF_CHI(chi.switch_state(momentum);
	       chidot.switch_state(momentum);
	       chiddot.switch_state(momentum));

	dim3 nr_blocks(NGRIDSIZE, NGRIDSIZE);
	dim3 nr_threads(NGRIDSIZE/2+1, 1);
	verlet_init_kernel<<<nr_blocks, nr_threads>>>(
		phi.mdata.ptr,
		IF_CHI_ARG(chi.mdata.ptr,)
		IF_CHI_ARG(nlt.chi2phi.mdata.ptr,)
		IF_CHI_ARG(nlt.phi2chi.mdata.ptr,)
		IF_PHI3_ARG(nlt.phi3.mdata.ptr,)
		IF_CHI3_ARG(nlt.chi3.mdata.ptr,)
		IF_PHI5_ARG(nlt.phi5.mdata.ptr,)
		IF_CHI5_ARG(nlt.chi5.mdata.ptr,)
		IF_MD_PHI_ARG(nlt.phi_md.mdata.ptr,)
		IF_MD_CHI_ARG(nlt.chi_md.mdata.ptr,)
		phiddot.mdata.ptr,
		IF_CHI_ARG(chiddot.mdata.ptr,)
		ts.a, ts.adot, addot);
}

__global__ void verlet_step_kernel(fftw_complex *phi_p,
				   IF_CHI_ARG(fftw_complex *chi_p,)
				   fftw_complex *phidot_p,
				   IF_CHI_ARG(fftw_complex *chidot_p,)
				   IF_CHI_ARG(fftw_complex *chi2phi_p,)
				   IF_CHI_ARG(fftw_complex *phi2chi_p,)
				   IF_PHI3_ARG(fftw_complex *phi3_p,)
				   IF_CHI3_ARG(fftw_complex *chi3_p,)
				   IF_PHI5_ARG(fftw_complex *phi5_p,)
				   IF_CHI5_ARG(fftw_complex *chi5_p,)
				   IF_MD_PHI_ARG(fftw_complex *phi_md_p,)
				   IF_MD_CHI_ARG(fftw_complex *chi_md_p,)
				   fftw_complex *phiddot,
				   IF_CHI_ARG(fftw_complex *chiddot,)
				   fftw_complex *phidot_staggered,
				   IF_CHI_ARG(fftw_complex *chidot_staggered,)
				   double a, double adot, double addot, double dt)
{
	int x = blockIdx.x;
	int y = blockIdx.y;
	int z = threadIdx.x;
	int px = x <= NGRIDSIZE/2 ? x : x - NGRIDSIZE;
	int py = y <= NGRIDSIZE/2 ? y : y - NGRIDSIZE;
	int pz = z;
	int idx = z + (NGRIDSIZE/2+1)*(y + NGRIDSIZE*x);

	double mom2 = pow2(MP_DP)*(pow2(px) + pow2(py) + pow2(pz));

	#pragma unroll
	for (int c = 0; c < 2; ++c) {
		phiddot[idx][c] = dphidotdt(phi_p[idx][c],
					    IF_CHI_ARG(chi_p[idx][c],)
					    IF_CHI_ARG(chi2phi_p[idx][c],)
					    IF_PHI3_ARG(phi3_p[idx][c],)
					    IF_PHI5_ARG(phi5_p[idx][c],)
					    IF_MD_PHI_ARG(phi_md_p[idx][c],)
					    a, adot,
					    addot, mom2);
#if ENABLE_CHI != 0
		chiddot[idx][c] = dchidotdt(phi_p[idx][c], chi_p[idx][c],
					    phi2chi_p[idx][c],
					    IF_CHI3_ARG(chi3_p[idx][c],)
					    IF_CHI5_ARG(chi5_p[idx][c],)
					    IF_MD_CHI_ARG(chi_md_p[idx][c],)
					    a, adot,
					    addot, mom2));
#endif

		phidot_p[idx][c] = phidot_staggered[idx][c] + 0.5 * phiddot[idx][c] * dt;
		IF_CHI(chidot_p[idx][c] = chidot_staggered[idx][c] + 0.5 * chiddot[idx][c] * dt);
	}
}

__global__ void step_reduction_kernel(fftw_complex *phi,
				      IF_CHI_ARG(fftw_complex *chi,)
				      fftw_complex *phidot,
				      IF_CHI_ARG(fftw_complex *chidot,)
				      fftw_complex *phiddot,
				      IF_CHI_ARG(fftw_complex *chiddot,)
				      fftw_complex *phidot_staggered,
				      IF_CHI_ARG(fftw_complex *chidot_staggered,)
				      double *total_gradient_phi,
				      IF_CHI_ARG(double *total_gradient_chi,)
				      double dp, double dt)
{
	int x = blockIdx.x;
	int y = blockIdx.y;
	int z = threadIdx.x;
	int px = x <= NGRIDSIZE/2 ? x : x - NGRIDSIZE;
	int py = y <= NGRIDSIZE/2 ? y : y - NGRIDSIZE;
	int pz = z;
	int idx = z + (NGRIDSIZE/2+1)*(y + NGRIDSIZE*x);

	double mom2 = pow2(dp)*(pow2(px) + pow2(py) + pow2(pz));

	#pragma unroll
	for (int c = 0; c < 2; ++c) {
		phidot_staggered[idx][c] = phidot[idx][c] + 0.5 * phiddot[idx][c] * dt;
		IF_CHI(chidot_staggered[idx][c] = chidot[idx][c] + 0.5 * chiddot[idx][c] * dt);

		phi[idx][c] += phidot_staggered[idx][c]*dt;
		IF_CHI(chi[idx][c] += chidot_staggered[idx][c]*dt);
	}

	mom2 *= (z == 0 || z == NGRIDSIZE/2) ? 1 : 2;

	total_gradient_phi[idx] = mom2*(pow2(phi[idx][0]) + pow2(phi[idx][1]));
	IF_CHI(total_gradient_chi[idx] = mom2*(pow2(chi[idx][0]) + pow2(chi[idx][1])));
}

template <typename R>
void verlet<R>::step()
{
	adot_staggered = ts.adot + 0.5 * addot * ts.dt;
	dptdt_staggered = dptdt + 0.5 * ddptdt * ts.dt;

	ts.a += ts.adot * ts.dt + 0.5 * addot * pow2(ts.dt);
	ts.physical_time += dptdt * ts.dt + 0.5 * ddptdt * pow2(ts.dt);

	phi.switch_state(momentum);
	phidot.switch_state(momentum);
	phiddot.switch_state(momentum);
	phidot_staggered.switch_state(momentum);

	IF_CHI(chi.switch_state(momentum);
	       chidot.switch_state(momentum);
	       chiddot.switch_state(momentum);
	       chidot_staggered.switch_state(momentum));

	auto total_gradient_phi_arr = double_array_gpu(NGRIDSIZE, NGRIDSIZE, NGRIDSIZE/2+1);
	IF_CHI(auto total_gradient_chi_arr = double_array_gpu(NGRIDSIZE, NGRIDSIZE, NGRIDSIZE/2+1));
	dim3 num_blocks(NGRIDSIZE, NGRIDSIZE);
	dim3 num_threads(NGRIDSIZE/2+1, 1);
	step_reduction_kernel<<<num_blocks, num_threads>>>(
		phi.mdata.ptr, IF_CHI_ARG(chi.mdata.ptr,)
		phidot.mdata.ptr, IF_CHI_ARG(chidot.mdata.ptr,)
		phiddot.mdata.ptr, IF_CHI_ARG(chiddot.mdata.ptr,)
		phidot_staggered.mdata.ptr, IF_CHI_ARG(chidot_staggered.mdata.ptr,)
		total_gradient_phi_arr.ptr(), IF_CHI_ARG(total_gradient_chi_arr.ptr(),)
		MP_DP, ts.dt);

	R total_gradient_phi = total_gradient_phi_arr.sum();
	IF_CHI(R total_gradient_chi = total_gradient_chi_arr.sum());

	R avg_gradient_phi = total_gradient_phi/pow<2, R>(NTOTAL_GRIDPOINTS);
#if ENABLE_CHI != 0
	R avg_gradient_chi = total_gradient_chi/pow<2, R>(NTOTAL_GRIDPOINTS);
#endif

	R avg_V = vi.integrate(phi, IF_CHI_ARG(chi,) ts.a);

	ts.addot = addot = mp.adoubledot_staggered(ts.t, ts.dt, ts.a,
						   adot_staggered, avg_gradient_phi,
						   IF_CHI_ARG(avg_gradient_chi,) avg_V);
	ts.adot = adot_staggered + 0.5 * addot * ts.dt;

	ddptdt = -RESCALE_S / RESCALE_B * pow(ts.a, -RESCALE_S - 1) * ts.adot;
	dptdt = dptdt_staggered + 0.5 * ddptdt * ts.dt;

	nlt.transform(phi, IF_CHI_ARG(chi,) ts.a);

	phi.switch_state(momentum);
	IF_CHI(chi.switch_state(momentum));

	dim3 nr_blocks(NGRIDSIZE, NGRIDSIZE);
	dim3 nr_threads(NGRIDSIZE/2+1, 1);

	verlet_step_kernel<<<nr_blocks, nr_threads>>>(
		phi.mdata.ptr,
		IF_CHI_ARG(chi.mdata.ptr,)
		phidot.mdata.ptr,
		IF_CHI_ARG(chidot.mdata.ptr,)
		IF_CHI_ARG(nlt.chi2phi.mdata.ptr,)
		IF_CHI_ARG(nlt.phi2chi.mdata.ptr,)
		IF_PHI3_ARG(nlt.phi3.mdata.ptr,)
		IF_CHI3_ARG(nlt.chi3.mdata.ptr,)
		IF_PHI5_ARG(nlt.phi5.mdata.ptr,)
		IF_CHI5_ARG(nlt.chi5.mdata.ptr,)
		IF_MD_PHI_ARG(nlt.phi_md.mdata.ptr,)
		IF_MD_CHI_ARG(nlt.chi_md.mdata.ptr,)
		phiddot.mdata.ptr,
		IF_CHI_ARG(chiddot.mdata.ptr,)
		phidot_staggered.mdata.ptr,
		IF_CHI_ARG(chidot_staggered.mdata.ptr,)
		ts.a, ts.adot, addot, ts.dt);
}

// Explicit instantiations
template class verlet<double>;
