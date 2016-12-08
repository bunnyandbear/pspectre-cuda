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

#include <iostream>
#include <cufftw.h>


void print_memory_usage();


template <typename R>
class fft_worker {};


template <>
class fft_worker<double>
{
public:
	typedef fftw_complex complex_t;
	
	fft_worker(int n0, int n1, int n2, double *datap, complex_t *datam, bool estimate = true)
	{
		construct(n0, n1, n2, datap, datam, estimate);
	}
	
	fft_worker()
		: m2p_plan(0), p2m_plan(0), constructed_(false) {}
	
	virtual ~fft_worker()
	{
		if (constructed_)
		{
			// free memory
			fftw_destroy_plan(m2p_plan);
			fftw_destroy_plan(p2m_plan);
			constructed_ = false;
		}
	}

	void construct(int n0, int n1, int n2, double *datap, complex_t *datam, bool estimate = true)
	{
		if (not constructed_)
		{
			#ifdef DEBUG
				std::cout << "\nConstructing fft_worker" << std::endl;
				std::cout << "Memory usage before creating fft plans (x2):" << std::endl;
				print_memory_usage();
			#endif

			// create forward plan
			p2m_plan = fftw_plan_dft_r2c_3d(n0, n1, n2, datap, datam, estimate ? FFTW_ESTIMATE : FFTW_MEASURE);

			// create reverse plan
			m2p_plan = fftw_plan_dft_c2r_3d(n0, n1, n2, datam, datap, estimate ? FFTW_ESTIMATE : FFTW_MEASURE);

			#ifdef DEBUG
				std::cout << "Memory usage after creating fft plans (x2):" << std::endl;
				print_memory_usage();
			#endif

			constructed_ = true;
		}
	}

	bool constructed()
	{
		return constructed_;
	}

	void execute_p2m(double *in, complex_t *out)
	{
		fftw_execute_dft_r2c(p2m_plan, in, out);
	}

	void execute_m2p(complex_t *in, double *out)
	{
		fftw_execute_dft_c2r(m2p_plan, in, out);
	}

protected:
	bool constructed_;
	fftw_plan m2p_plan;
	fftw_plan p2m_plan;
};

#endif // FFT_HPP
