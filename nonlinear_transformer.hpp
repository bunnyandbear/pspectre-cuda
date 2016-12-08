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

#include "field_size.hpp"
#include "model_params.hpp"
#include "field.hpp"
#include "time_state.hpp"
#include "fft.hpp"

template <typename R>
class nonlinear_transformer
{
public:
	nonlinear_transformer(field_size &fs_, time_state<R> &ts_, fft_worker<R> &fft_plans)
		: fs(fs_), upfs(fs_.n), ts(ts_),
		phi2chi("phi2chi"), chi2phi("chi2phi"),
		phi3("phi3"), chi3("chi3"),
		phi5("phi5"), chi5("chi5")
	{
		phi2chi.construct(upfs, fft_plans);
		chi2phi.construct(upfs, fft_plans);
		
		if (LAMBDA_PHI != 0.0) {
			phi3.construct(upfs, fft_plans);
		}
		
		if (LAMBDA_CHI != 0.0) {
			chi3.construct(upfs, fft_plans);
		}

		if (GAMMA_PHI != 0.0) {
			phi5.construct(upfs, fft_plans);
		}
		
		if (GAMMA_CHI != 0.0) {
			chi5.construct(upfs, fft_plans);
		}

		if (MD_E_PHI != 0.0) {
			phi_md.construct(upfs, fft_plans);
		}

		if (MD_E_CHI != 0.0) {
			chi_md.construct(upfs, fft_plans);
		}
	}

public:	
	void transform(field<R> &phi, field<R> &chi, R a_t,
		field_state final_state = momentum);

protected:
	field_size &fs, upfs;
	time_state<R> &ts;

public:
	field<R> phi2chi, chi2phi;
	field<R> phi3, chi3;
	field<R> phi5, chi5;
	field<R> phi_md, chi_md;
};

#endif // NONLINEAR_TRANSFORMER_HPP
