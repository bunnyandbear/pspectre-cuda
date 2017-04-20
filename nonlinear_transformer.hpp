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

/**
 * @file
 * @brief Momentum-space representations of nonlinear field terms.
 */

#ifndef NONLINEAR_TRANSFORMER_HPP
#define NONLINEAR_TRANSFORMER_HPP

#include "model_params.hpp"
#include "field.hpp"
#include "time_state.hpp"
#include "fft.hpp"

template <typename R>
class nonlinear_transformer
{
public:
	nonlinear_transformer(time_state<R> &ts_, fft_worker<R> &fft_plans)
		: ts(ts_)
		  IF_CHI_ARG(,phi2chi("phi2chi"))
		  IF_CHI_ARG(,chi2phi("chi2phi"))
		IF_PHI3_ARG(,phi3("phi3"))
		IF_CHI3_ARG(,chi3("chi3"))
		IF_PHI5_ARG(,phi5("phi5"))
		IF_CHI5_ARG(,chi5("chi5"))
		IF_MD_PHI_ARG(,phi_md("phi_md"))
		IF_MD_CHI_ARG(,chi_md("chi_md")) {

		IF_CHI(phi2chi.construct(fft_plans);
		       chi2phi.construct(fft_plans));
		
		IF_PHI3(phi3.construct(fft_plans));

		IF_CHI3(chi3.construct(fft_plans));

		IF_PHI5(phi5.construct(fft_plans));
		
		IF_CHI5(chi5.construct(fft_plans));

		IF_MD_PHI(phi_md.construct(fft_plans));

		IF_MD_CHI(chi_md.construct(fft_plans));
	}

public:	
	void transform(field<R> &phi, IF_CHI_ARG(field<R> &chi,) R a_t,
		       field_state final_state = momentum);

protected:
	time_state<R> &ts;

public:
	IF_CHI(field<R> phi2chi;)
	IF_CHI(field<R> chi2phi;)
	IF_PHI3(field<R> phi3;)
	IF_CHI3(field<R> chi3;)
	IF_PHI5(field<R> phi5;)
	IF_CHI5(field<R> chi5;)
	IF_MD_PHI(field<R> phi_md;)
	IF_MD_CHI(field<R> chi_md;)
};

#endif // NONLINEAR_TRANSFORMER_HPP
