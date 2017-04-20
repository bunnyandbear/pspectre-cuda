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
__global__ void nonlin_trans_kernel(double *phi,
				    IF_CHI_ARG(double *chi,)
				    IF_CHI_ARG(double *phi2chi,)
				    IF_CHI_ARG(double *chi2phi,)
				    IF_MD_PHI_ARG(double *phi_md,)
				    IF_MD_CHI_ARG(double *chi_md,)
				    IF_PHI3_ARG(double *phi3,)
				    IF_CHI3_ARG(double *chi3,)
				    IF_PHI5_ARG(double *phi5,)
				    IF_CHI5_ARG(double *chi5,)
				    double a_t)
{
	int x = blockIdx.x;
	int y = blockIdx.y;
	int z = threadIdx.x;
	int ldl = 2*(NGRIDSIZE/2+1);
	int fdx = z + ldl*(y + NGRIDSIZE*x);
	int idx = z + ldl*(y + NGRIDSIZE*x);
	double p = phi[fdx];

	IF_CHI(double c = chi[fdx];

	       phi2chi[idx] = pow2(p)*c;
	       chi2phi[idx] = pow2(c)*p);

	IF_PHI3(phi3[idx] = pow3(p));

	IF_CHI3(chi3[idx] = pow3(c));

	IF_PHI5(phi5[idx] = pow5(p));

	IF_CHI5(chi5[idx] = pow5(c));

#if ENABLE_MD_PHI != 0
	phi_md[idx] = 2.0 * MD_C_PHI * MD_E_PHI * p *
		pow(1.0 + pow(a_t, -2. * RESCALE_R) * pow2(p/RESCALE_A) / pow2(MD_S_PHI),
		    MD_E_PHI - 1.0);
#endif

#if ENABLE_MD_CHI != 0
	chi_md[idx] = 2.0 * MD_C_CHI * MD_E_CHI * p *
		pow(1.0 + pow(a_t, -2. * RESCALE_R) * pow2(p/RESCALE_A) / pow2(MD_S_CHI),
		    MD_E_CHI - 1.0);
#endif
}

template <typename R>
void nonlinear_transformer<R>::transform(field<R> &phi,
					 IF_CHI_ARG(field<R> &chi,) R a_t, field_state final_state)
{
	phi.switch_state(position);

	IF_CHI(chi.switch_state(position);

	       phi2chi.switch_state(uninitialized);
	       phi2chi.switch_state(position);

	       chi2phi.switch_state(uninitialized);
	       chi2phi.switch_state(position));
	
	IF_PHI3(phi3.switch_state(uninitialized);
		phi3.switch_state(position));
	
	IF_CHI3(chi3.switch_state(uninitialized);
		chi3.switch_state(position));

	IF_PHI5(phi5.switch_state(uninitialized);
		phi5.switch_state(position));
	
	IF_CHI5(chi5.switch_state(uninitialized);
		chi5.switch_state(position));

	IF_MD_PHI(phi_md.switch_state(uninitialized);
		  phi_md.switch_state(position));

	IF_MD_CHI(chi_md.switch_state(uninitialized);
		  chi_md.switch_state(position));

	dim3 nr_blocks(NGRIDSIZE, NGRIDSIZE);
	dim3 nr_threads(NGRIDSIZE, 1);
	nonlin_trans_kernel<<<nr_blocks, nr_threads>>>(phi.data.ptr,
						       IF_CHI_ARG(chi.data.ptr,)
						       IF_CHI_ARG(phi2chi.data.ptr,)
						       IF_CHI_ARG(chi2phi.data.ptr,)
						       IF_MD_PHI_ARG(phi_md.data.ptr,)
						       IF_MD_CHI_ARG(chi_md.data.ptr,)
						       IF_PHI3_ARG(phi3.data.ptr,)
						       IF_CHI3_ARG(chi3.data.ptr,)
						       IF_PHI5_ARG(phi5.data.ptr,)
						       IF_CHI5_ARG(chi5.data.ptr,)
						       a_t);

	IF_CHI(phi2chi.switch_state(final_state);
	       chi2phi.switch_state(final_state));
	
	IF_PHI3(phi3.switch_state(final_state));
	
	IF_CHI3(chi3.switch_state(final_state));

	IF_PHI5(phi5.switch_state(final_state));

	IF_CHI5(chi5.switch_state(final_state));

	IF_MD_PHI(phi_md.switch_state(final_state));

	IF_MD_CHI(chi_md.switch_state(final_state));
}

// Explicit instantiations
template class nonlinear_transformer<double>;
