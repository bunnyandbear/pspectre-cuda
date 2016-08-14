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
 * @brief FFT wrappers.
 */

#ifndef FFT_HPP
#define FFT_HPP

#include <cufftw.h>

template <typename R>
class fft_dft_c2r_3d_plan {};

template <>
class fft_dft_c2r_3d_plan<double>
{
public:
	typedef fftw_complex complex_t;
	
public:
	fft_dft_c2r_3d_plan(int n0, int n1, int n2, complex_t *in, double *out, bool estimate = true)
	{
		construct(n0, n1, n2, in, out, estimate);
	}
	
	fft_dft_c2r_3d_plan()
		: plan(0) {}
	
	~fft_dft_c2r_3d_plan()
	{
		fftw_destroy_plan(plan);
	}
	
public:
	void construct(int n0, int n1, int n2, complex_t *in, double *out, bool estimate = true)
	{
		plan = fftw_plan_dft_c2r_3d(n0, n1, n2, in, out, estimate ? FFTW_ESTIMATE : FFTW_MEASURE);
	}	

	void execute()
	{
		fftw_execute(plan);
	}

	bool constructed() {
		return plan == 0;
	}

protected:
	fftw_plan plan;
};

template <typename R>
class fft_dft_r2c_3d_plan {};

template <>
class fft_dft_r2c_3d_plan<double>
{
public:
	typedef fftw_complex complex_t;
	
public:
	fft_dft_r2c_3d_plan(int n0, int n1, int n2, double *in, complex_t *out, bool estimate = true)
	{
		construct(n0, n1, n2, in, out, estimate);
	}
	
	fft_dft_r2c_3d_plan()
		: plan(0) {}
	
	~fft_dft_r2c_3d_plan()
	{
		fftw_destroy_plan(plan);
	}

public:
	void execute()
	{
		fftw_execute(plan);
	}
	
	void construct(int n0, int n1, int n2, double *in, complex_t *out, bool estimate = true)
	{
		plan = fftw_plan_dft_r2c_3d(n0, n1, n2, in, out, estimate ? FFTW_ESTIMATE : FFTW_MEASURE);
	}

	bool constructed() {
		return plan == 0;
	}

protected:
	fftw_plan plan;
};

#endif // FFT_HPP
