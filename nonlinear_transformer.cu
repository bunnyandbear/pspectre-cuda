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
__global__ void nonlin_trans_kernel(double *phi, double *chi,
				    double *phi2chi, double *chi2phi,
				    double *phi_md, double *chi_md,
				    double *phi3, double *chi3,
				    double *phi5, double *chi5,
				    double a_t)
{
	int x = blockIdx.x;
	int y = blockIdx.y;
	int z = threadIdx.x;
	int ldl = 2*(NGRIDSIZE/2+1);
	int fdx = z + ldl*(y + NGRIDSIZE*x);
	int idx = z + ldl*(y + NGRIDSIZE*x);
	double p = phi[fdx];
	double c = chi[fdx];

	phi2chi[idx] = pow2(p)*c;
	chi2phi[idx] = pow2(c)*p;

	if (LAMBDA_PHI != 0.0) {
		phi3[idx] = pow3(p);
	}

	if (LAMBDA_CHI != 0.0) {
		chi3[idx] = pow3(c);
	}

	if (GAMMA_PHI != 0.0) {
		phi5[idx] = pow5(p);
	}

	if (GAMMA_CHI != 0.0) {
		chi5[idx] = pow5(c);
	}

	if (MD_E_PHI != 0.0) {
		phi_md[idx] = 2.0 * MD_C_PHI * MD_E_PHI * p *
			pow(1.0 +
			    pow(a_t, -2. * RESCALE_R) * pow2(p/RESCALE_A) / pow2(MD_S_PHI),
			    MD_E_PHI - 1.0);
	}

	if (MD_E_CHI != 0.0) {
		chi_md[idx] = 2.0 * MD_C_CHI * MD_E_CHI * p *
			pow(1.0 +
			    pow(a_t, -2. * RESCALE_R) * pow2(p/RESCALE_A) / pow2(MD_S_CHI),
			    MD_E_CHI - 1.0);
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
	
	if (LAMBDA_PHI != 0.0) {
		phi3.switch_state(uninitialized);
		phi3.switch_state(position);
	}
	
	if (LAMBDA_CHI != 0.0) {
		chi3.switch_state(uninitialized);
		chi3.switch_state(position);
	}

	if (GAMMA_PHI != 0.0) {
		phi5.switch_state(uninitialized);
		phi5.switch_state(position);
	}
	
	if (GAMMA_CHI != 0.0) {
		chi5.switch_state(uninitialized);
		chi5.switch_state(position);
	}

	if (MD_E_PHI != 0.0) {
		phi_md.switch_state(uninitialized);
		phi_md.switch_state(position);
	}

	if (MD_E_CHI != 0.0) {
		chi_md.switch_state(uninitialized);
		chi_md.switch_state(position);
	}

	dim3 nr_blocks(NGRIDSIZE, NGRIDSIZE);
	dim3 nr_threads(NGRIDSIZE, 1);
	nonlin_trans_kernel<<<nr_blocks, nr_threads>>>(phi.data.ptr, chi.data.ptr,
						       phi2chi.data.ptr, chi2phi.data.ptr,
						       phi_md.data.ptr, chi_md.data.ptr,
						       phi3.data.ptr, chi3.data.ptr,
						       phi5.data.ptr, chi5.data.ptr,
						       a_t);

	phi2chi.switch_state(final_state);
	chi2phi.switch_state(final_state);
	
	if (LAMBDA_PHI != 0.0) {
		phi3.switch_state(final_state);
	}
	
	if (LAMBDA_CHI != 0.0) {
		chi3.switch_state(final_state);
	}

	if (GAMMA_PHI != 0.0) {
		phi5.switch_state(final_state);
	}
	
	if (GAMMA_CHI != 0.0) {
		chi5.switch_state(final_state);
	}

	if (MD_E_PHI != 0.0) {
		phi_md.switch_state(final_state);
	}

	if (MD_E_CHI != 0.0) {
		chi_md.switch_state(final_state);
	}
}

// Explicit instantiations
template class nonlinear_transformer<double>;
