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
#include "energy_outputter.hpp"
#include "grid_funcs.hpp"

#include <sstream>
#include <iomanip>

double compute_energy_scaling(double a, double adot);

using namespace std;

template <typename R>
energy_outputter<R>::energy_outputter(model_params &mp_, time_state<R> &ts_,
	field<R> &phi_, field<R> &chi_, field<R> &phidot_, field<R> &chidot_)
	: mp(mp_), ts(ts_), phi(phi_), chi(chi_), phidot(phidot_), chidot(chidot_),
	vi(), avg_rho_phys(0.0), avg_rho(0.0)
{
	of.open("energy.tsv");
	of << setprecision(30) << scientific;
}

__global__ void energy_sum_kernel(fftw_complex *phi, fftw_complex *chi,
				  fftw_complex *phidot, fftw_complex *chidot,
				  double *avg_phi_squared, double *avg_chi_squared,
				  double *avg_phidot_squared, double *avg_chidot_squared,
				  double *avg_gradient_phi_x, double *avg_gradient_chi_x,
				  double *avg_gradient_phi_y, double *avg_gradient_chi_y,
				  double *avg_gradient_phi_z, double *avg_gradient_chi_z,
				  double *avg_ffd_phi, double *avg_ffd_chi,
				  double dp)
{
	int x = blockIdx.x;
	int y = blockIdx.y;
	int z = threadIdx.x;
	int px = x <= NGRIDSIZE/2 ? x : x - NGRIDSIZE;
	int py = y <= NGRIDSIZE/2 ? y : y - NGRIDSIZE;
	int pz = z;
	int idx = z + (NGRIDSIZE/2+1)*(y + NGRIDSIZE*x);

	int cnt = (z == 0 || z == NGRIDSIZE/2) ? 1 : 2;

	double phi_squared = pow2(phi[idx][0]) + pow2(phi[idx][1]);
	double chi_squared = pow2(chi[idx][0]) + pow2(chi[idx][1]);
	avg_phi_squared[idx] = cnt * phi_squared;
	avg_chi_squared[idx] = cnt * chi_squared;
				
	avg_phidot_squared[idx] = cnt * (pow2(phidot[idx][0]) + pow2(phidot[idx][1]));
	avg_chidot_squared[idx] = cnt * (pow2(chidot[idx][0]) + pow2(chidot[idx][1]));
				
	avg_ffd_phi[idx] = cnt * (phi[idx][0] * phidot[idx][0] +
		phi[idx][1] * phidot[idx][1]);
	avg_ffd_chi[idx] = cnt * (chi[idx][0] * chidot[idx][0] +
		chi[idx][1] * chidot[idx][1]);
				
	double mom2x = pow2(dp) * pow2(px);
	double mom2y = pow2(dp) * pow2(py);
	double mom2z = pow2(dp) * pow2(pz);
	avg_gradient_phi_x[idx] = cnt * mom2x * phi_squared;
	avg_gradient_chi_x[idx] = cnt * mom2x * chi_squared;
	avg_gradient_phi_y[idx] = cnt * mom2y * phi_squared;
	avg_gradient_chi_y[idx] = cnt * mom2y * chi_squared;
	avg_gradient_phi_z[idx] = cnt * mom2z * phi_squared;
	avg_gradient_chi_z[idx] = cnt * mom2z * chi_squared;
}

template <typename R>
void energy_outputter<R>::output(bool no_output)
{
	R avg_V = vi.integrate(phi, chi, ts.a);
	
	phi.switch_state(momentum);
	chi.switch_state(momentum);

	phidot.switch_state(momentum);
	chidot.switch_state(momentum);
	
	auto avg_phi_squared_arr = double_array_gpu(NGRIDSIZE, NGRIDSIZE, NGRIDSIZE/2+1);
	auto avg_chi_squared_arr = double_array_gpu(NGRIDSIZE, NGRIDSIZE, NGRIDSIZE/2+1);
	auto avg_phidot_squared_arr = double_array_gpu(NGRIDSIZE, NGRIDSIZE, NGRIDSIZE/2+1);
	auto avg_chidot_squared_arr = double_array_gpu(NGRIDSIZE, NGRIDSIZE, NGRIDSIZE/2+1);
	auto avg_gradient_phi_x_arr = double_array_gpu(NGRIDSIZE, NGRIDSIZE, NGRIDSIZE/2+1);
	auto avg_gradient_chi_x_arr = double_array_gpu(NGRIDSIZE, NGRIDSIZE, NGRIDSIZE/2+1);
	auto avg_gradient_phi_y_arr = double_array_gpu(NGRIDSIZE, NGRIDSIZE, NGRIDSIZE/2+1);
	auto avg_gradient_chi_y_arr = double_array_gpu(NGRIDSIZE, NGRIDSIZE, NGRIDSIZE/2+1);
	auto avg_gradient_phi_z_arr = double_array_gpu(NGRIDSIZE, NGRIDSIZE, NGRIDSIZE/2+1);
	auto avg_gradient_chi_z_arr = double_array_gpu(NGRIDSIZE, NGRIDSIZE, NGRIDSIZE/2+1);
	auto avg_ffd_phi_arr = double_array_gpu(NGRIDSIZE, NGRIDSIZE, NGRIDSIZE/2+1);
	auto avg_ffd_chi_arr = double_array_gpu(NGRIDSIZE, NGRIDSIZE, NGRIDSIZE/2+1);

	dim3 num_blocks(NGRIDSIZE, NGRIDSIZE);
	dim3 num_threads(NGRIDSIZE/2+1, 1);
	energy_sum_kernel<<<num_blocks, num_threads>>>(
		phi.mdata.ptr, chi.mdata.ptr,
		phidot.mdata.ptr, chidot.mdata.ptr,
		avg_phi_squared_arr.ptr(), avg_chi_squared_arr.ptr(),
		avg_phidot_squared_arr.ptr(), avg_chidot_squared_arr.ptr(),
		avg_gradient_phi_x_arr.ptr(), avg_gradient_chi_x_arr.ptr(),
		avg_gradient_phi_y_arr.ptr(), avg_gradient_chi_y_arr.ptr(),
		avg_gradient_phi_z_arr.ptr(), avg_gradient_chi_z_arr.ptr(),
		avg_ffd_phi_arr.ptr(), avg_ffd_chi_arr.ptr(),
		MP_DP);

	double avg_phi_squared = avg_phi_squared_arr.sum();
	double avg_chi_squared = avg_chi_squared_arr.sum();
	double avg_phidot_squared = avg_phidot_squared_arr.sum();
	double avg_chidot_squared = avg_chidot_squared_arr.sum();
	double avg_gradient_phi_x = avg_gradient_phi_x_arr.sum();
	double avg_gradient_chi_x = avg_gradient_chi_x_arr.sum();
	double avg_gradient_phi_y = avg_gradient_phi_y_arr.sum();
	double avg_gradient_chi_y = avg_gradient_chi_y_arr.sum();
	double avg_gradient_phi_z = avg_gradient_phi_z_arr.sum();
	double avg_gradient_chi_z = avg_gradient_chi_z_arr.sum();
	double avg_ffd_phi = avg_ffd_phi_arr.sum();
	double avg_ffd_chi = avg_ffd_chi_arr.sum();

	// The first factor of 1./N^3 comes from Parseval's theorem.
	avg_phidot_squared /= 2*pow2(NTOTAL_GRIDPOINTS);
	avg_chidot_squared /= 2*pow2(NTOTAL_GRIDPOINTS);

	R fld_fac = 0.5 * pow2(RESCALE_R) * pow2(ts.adot/ts.a);
	avg_phi_squared *= fld_fac / pow2(NTOTAL_GRIDPOINTS);
	avg_chi_squared *= fld_fac / pow2(NTOTAL_GRIDPOINTS);

	R grad_fac = 0.5 * pow(ts.a, -2. * RESCALE_S - 2.);
	avg_gradient_phi_x *= grad_fac / pow2(NTOTAL_GRIDPOINTS);
	avg_gradient_chi_x *= grad_fac / pow2(NTOTAL_GRIDPOINTS);
	avg_gradient_phi_y *= grad_fac / pow2(NTOTAL_GRIDPOINTS);
	avg_gradient_chi_y *= grad_fac / pow2(NTOTAL_GRIDPOINTS);
	avg_gradient_phi_z *= grad_fac / pow2(NTOTAL_GRIDPOINTS);
	avg_gradient_chi_z *= grad_fac / pow2(NTOTAL_GRIDPOINTS);
	
	R ffd_fac = -RESCALE_R * ts.adot/ts.a;
	avg_ffd_phi *= ffd_fac / pow2(NTOTAL_GRIDPOINTS);
	avg_ffd_chi *= ffd_fac / pow2(NTOTAL_GRIDPOINTS);

	// This is the *average* energy per gridpoint.
	avg_rho_phys = avg_V + avg_phi_squared + avg_chi_squared +
		avg_phidot_squared + avg_chidot_squared +
		avg_gradient_phi_x + avg_gradient_chi_x +
		avg_gradient_phi_y + avg_gradient_chi_y +
		avg_gradient_phi_z + avg_gradient_chi_z +
		avg_ffd_phi + avg_ffd_chi;

	R avg_p_phys = -avg_V + avg_phi_squared + avg_chi_squared +
		avg_phidot_squared + avg_chidot_squared -
		(
			avg_gradient_phi_x + avg_gradient_chi_x +
			avg_gradient_phi_y + avg_gradient_chi_y +
			avg_gradient_phi_z + avg_gradient_chi_z
		)/3 +
		avg_ffd_phi + avg_ffd_chi;
	R avg_w = avg_p_phys/avg_rho_phys;

	const R es = compute_energy_scaling(ts.a, ts.adot);
	avg_rho = es * avg_rho_phys;

	if (!no_output) {
		of << setw(10) << setfill('0') <<
			ts.t << "\t" << RESCALE_B * ts.physical_time << "\t" <<
			avg_rho_phys << "\t" << avg_rho << "\t" <<
			es * avg_phidot_squared << "\t" <<
			es * avg_chidot_squared << "\t" <<
			es * avg_ffd_phi << "\t" <<
			es * avg_ffd_chi << "\t" <<
			es * avg_phi_squared << "\t" <<
			es * avg_chi_squared << "\t" <<
			es * (avg_gradient_phi_x + avg_gradient_phi_y + avg_gradient_phi_z) << "\t" <<
			es * (avg_gradient_chi_x + avg_gradient_chi_y + avg_gradient_chi_z) << "\t" <<
			es * avg_V << "\t" <<
			avg_phidot_squared << "\t" <<
			avg_chidot_squared << "\t" <<
			avg_ffd_phi << "\t" <<
			avg_ffd_chi << "\t" <<
			avg_phi_squared << "\t" <<
			avg_chi_squared << "\t" <<
			avg_gradient_phi_x + avg_gradient_phi_y + avg_gradient_phi_z << "\t" <<
			avg_gradient_chi_x + avg_gradient_chi_y + avg_gradient_chi_z << "\t" <<
			avg_V << "\t" <<

			avg_gradient_phi_x << "\t" <<
			avg_gradient_chi_x << "\t" <<
			avg_gradient_phi_y << "\t" <<
			avg_gradient_chi_y << "\t" <<
			avg_gradient_phi_z << "\t" <<
			avg_gradient_chi_z << "\t" <<

			avg_p_phys << "\t" <<
			avg_w <<

			endl;
		of.flush();
	}
}

/**
 * @page energy_tsv energy.tsv
 * energy.tsv is a tab serarated file with the following fields:
 * @li Program time
 * @li Physical time
 * @li Average physical energy (w.r.t. the rescaled length)
 * @li Average energy normalized by the Friedmann equation
 * @li Average normalized @f$ \phi'^2 @f$ energy contribution
 * @li Average normalized @f$ \chi'^2 @f$ energy contribution
 * @li Average normalized @f$ \phi\phi' @f$ energy contribution
 * @li Average normalized @f$ \chi\chi' @f$ energy contribution
 * @li Average normalized @f$ \phi^2 @f$ energy contribution
 * @li Average normalized @f$ \chi^2 @f$ energy contribution
 * @li Average normalized @f$ \nabla \phi @f$ energy contribution
 * @li Average normalized @f$ \nabla \chi @f$ energy contribution
 * @li Average normalized potential-energy contribution
 * @li Average physical @f$ \phi'^2 @f$ energy contribution
 * @li Average physical @f$ \chi'^2 @f$ energy contribution
 * @li Average physical @f$ \phi\phi' @f$ energy contribution
 * @li Average physical @f$ \chi\chi' @f$ energy contribution
 * @li Average physical @f$ \phi^2 @f$ energy contribution
 * @li Average physical @f$ \chi^2 @f$ energy contribution
 * @li Average physical @f$ \nabla \phi @f$ energy contribution
 * @li Average physical @f$ \nabla \chi @f$ energy contribution
 * @li Average physical potential-energy contribution
 * @li Average physical @f$ \nabla \phi @f$ x-direction energy contribution
 * @li Average physical @f$ \nabla \chi @f$ x-direction energy contribution
 * @li Average physical @f$ \nabla \phi @f$ y-direction energy contribution
 * @li Average physical @f$ \nabla \chi @f$ y-direction energy contribution
 * @li Average physical @f$ \nabla \phi @f$ z-direction energy contribution
 * @li Average physical @f$ \nabla \chi @f$ z-direction energy contribution
 * @li Average physical pressure
 * @li Average w (the e.o.s. parameter)
 */

// Explicit instantiations
template class energy_outputter<double>;
