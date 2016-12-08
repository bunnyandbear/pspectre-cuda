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
 * @brief Three-dimensional scalar fields.
 */

#ifndef FIELD_HPP
#define FIELD_HPP

#include "fft.hpp"
#include "field_size.hpp"
#include "gpu_array_accessor.hpp"

enum field_state
{
	uninitialized,
	position,
	momentum,
};

/** 
 * @brief A three-dimensional scalar field in both position and momentum space.
 */

template <typename R>
class field
{
public:
	field(field_size &fs_, fft_worker<R> &fft_plans_, const char *name_ = 0)
		: state(uninitialized), name(name_), data(0), mdata(0)
	{
		construct(fs_, fft_plans_);
	}
	
	field(const char *name_ = 0)
		: ldl(0), state(uninitialized), name(name_), data(0), mdata(0) {};

	~field();

public:
	void construct(field_size &fs_, fft_worker<R> &fft_plans_);
	void divby(R v);
	void switch_state(field_state state_);
	void fill0();
	void upload(fftw_complex *fld);

public:
	field_size fs;

	/** 
	 * @brief The position-space data.
	 *
	 * @note The inner (z) dimension is padded to a size of 2*(floor(n/2)+1).
	 */

	gpu_array_accessor_double data;

	/**
	 * @brief The length of the last dimension of the data array.
	 */

	int ldl;

	/** 
	 * @brief The momentum-space data.
	 */

	gpu_array_accessor_fftw_complex mdata;
	
protected:
	field_state state;
	fft_worker<R> *fft_plans;

private:
	double *raw_ptr;

public:
	const char *name;
};

#endif // FIELD_HPP
