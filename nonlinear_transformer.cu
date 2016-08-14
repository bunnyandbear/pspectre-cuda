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

#include "nonlinear_transformer.hpp"

using namespace std;

/* x  *  y  *  z
 * n  *  n  *  2*(n/2+1)
 * BLK(x,y) *  THR(x,1)
 */
#define pow2(p) ((p)*(p))
#define pow3(p) ((p)*(p)*(p))
#define pow5(p) ((p)*(p)*(p)*(p)*(p))
__global__ void nonlin_trans_kernel(double *phi, double *chi,
				    double *phi2chi, double *chi2phi,
				    double *phi_md, double *chi_md,
				    double *phi3, double *chi3,
				    double *phi5, double *chi5,
				    double lambda_phi, double lambda_chi,
				    double gamma_phi, double gamma_chi,
				    double md_e_phi, double md_c_phi,
				    double md_e_chi, double md_c_chi,
				    double md_s_phi, double md_s_chi,
				    double a_t, double rescale_A, double rescale_r,
				    int n, int ldl)
{
	int x = blockIdx.x;
	int y = blockIdx.y;
	int z = threadIdx.x;
	int fdx = z + ldl*(y + n*x);
	int idx = z + ldl*(y + n*x);
	double p = phi[fdx];
	double c = chi[fdx];

	phi2chi[idx] = pow2(p)*c;
	chi2phi[idx] = pow2(c)*p;

	if (lambda_phi != 0.0) {
		phi3[idx] = pow3(p);
	}

	if (lambda_chi != 0.0) {
		chi3[idx] = pow3(c);
	}

	if (gamma_phi != 0.0) {
		phi5[idx] = pow5(p);
	}

	if (gamma_chi != 0.0) {
		chi5[idx] = pow5(c);
	}

	if (md_e_phi != 0.0) {
		phi_md[idx] = 2.0 * md_c_phi * md_e_phi * p *
			pow(1.0 +
			    pow(a_t, -2. * rescale_r) * pow2(p/rescale_A) / pow2(md_s_phi),
			    md_e_phi - 1.0);
	}

	if (md_e_chi != 0.0) {
		chi_md[idx] = 2.0 * md_c_chi * md_e_chi * p *
			pow(1.0 +
			    pow(a_t, -2. * rescale_r) * pow2(p/rescale_A) / pow2(md_s_chi),
			    md_e_chi - 1.0);
	}
}

template <typename R>
void nonlinear_transformer<R>::transform(field<R> &phi, field<R> &chi, R a_t, field_state final_state)
{
	phi.switch_state(position);
	chi.switch_state(position);

	phi2chi.switch_state(uninitialized);
	phi2chi.switch_state(position);
	
	chi2phi.switch_state(uninitialized);
	chi2phi.switch_state(position);
	
	if (mp.lambda_phi != 0.0) {
		phi3.switch_state(uninitialized);
		phi3.switch_state(position);
	}
	
	if (mp.lambda_chi != 0.0) {
		chi3.switch_state(uninitialized);
		chi3.switch_state(position);
	}

	if (mp.gamma_phi != 0.0) {
		phi5.switch_state(uninitialized);
		phi5.switch_state(position);
	}
	
	if (mp.gamma_chi != 0.0) {
		chi5.switch_state(uninitialized);
		chi5.switch_state(position);
	}

	if (mp.md_e_phi != 0.0) {
		phi_md.switch_state(uninitialized);
		phi_md.switch_state(position);
	}

	if (mp.md_e_chi != 0.0) {
		chi_md.switch_state(uninitialized);
		chi_md.switch_state(position);
	}

	dim3 nr_blocks(fs.n, fs.n);
	dim3 nr_threads(fs.n, 1);
	nonlin_trans_kernel<<<nr_blocks, nr_threads>>>(phi.data.ptr, chi.data.ptr,
						       phi2chi.data.ptr, chi2phi.data.ptr,
						       phi_md.data.ptr, chi_md.data.ptr,
						       phi3.data.ptr, chi3.data.ptr,
						       phi5.data.ptr, chi5.data.ptr,
						       mp.lambda_phi, mp.lambda_chi,
						       mp.gamma_phi, mp.gamma_chi,
						       mp.md_e_phi, mp.md_c_phi,
						       mp.md_e_chi, mp.md_c_chi,
						       mp.md_s_phi, mp.md_s_chi,
						       a_t, mp.rescale_A, mp.rescale_r,
						       fs.n, phi.ldl);

	phi2chi.switch_state(final_state);
	chi2phi.switch_state(final_state);
	
	if (mp.lambda_phi != 0.0) {
		phi3.switch_state(final_state);
	}
	
	if (mp.lambda_chi != 0.0) {
		chi3.switch_state(final_state);
	}

	if (mp.gamma_phi != 0.0) {
		phi5.switch_state(final_state);
	}
	
	if (mp.gamma_chi != 0.0) {
		chi5.switch_state(final_state);
	}

	if (mp.md_e_phi != 0.0) {
		phi_md.switch_state(final_state);
	}

	if (mp.md_e_chi != 0.0) {
		chi_md.switch_state(final_state);
	}
}

// Explicit instantiations
template class nonlinear_transformer<double>;
