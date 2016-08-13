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
 * @mainpage SpectRE - A Spectral Code for Reheating
 * SpectRE is a pseudo-spectral code for simulating a pair of interacting scalar fields
 * in a self-consistently expanding background. These fields are named phi and chi.
 *
 * @par
 * The time-dependent variable-rescaling scheme from LatticeEasy is used to eliminate the
 * first order term from the equations of motion. The fields can be initialized using either
 * the scheme from LatticeEasy or the scheme from Defrost.
 *
 * @li @ref building
 * @li @ref running
 * @li @ref outputs
 *
 * @section refs References
 * @li Gary Felder and Igor Tkachev. LATTICEEASY: A Program for Lattice Simulations of Scalar
 * Fields in an Expanding Universe. arXiv:hep-ph/0011159v1.
 * http://www.science.smith.edu/departments/Physics/fstaff/gfelder/latticeeasy/
 * @li Andrei V. Frolov. DEFROST: A New Code for Simulating Preheating after Inflation.
 * arXiv:0809.4904v2 [hep-ph]. http://www.sfu.ca/physics/cosmology/defrost/
 */

/**
 * @page building Building
 *
 * @section make Make
 * Building SpectRE requires GNU make. On systems where GNU make is not the
 * system's default make command, GNU make is often called gmake. 
 *
 * @section reqs Requirements
 * SpectRE should build and run on any POSIX-style operating system, and uses OpenMP for
 * shared-memory parallelism. It requires:
 * @li FFTW 3.
 * @li G++ (the GNU C++ compiler version 4+).
 *
 */

#define _XOPEN_SOURCE 600

#include "field.hpp"
#include "integrator.hpp"
#include "model.hpp"	

#include <cstdlib>
#include <cstring>

#include <vector>

#include <iostream>

#include <climits>
#include <cfloat>
#include <cmath>

#include <fenv.h>
#if defined(__i386__) && defined(__SSE__)
#include <xmmintrin.h>
#endif

#include <unistd.h>
#include <wordexp.h>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace std;

void print_cuda_info();

int main(int argc, char *argv[])
{
#if defined(FE_NOMASK_ENV) && !defined(__INTEL_COMPILER)
	fesetenv(FE_NOMASK_ENV);
	fedisableexcept(/* FE_OVERFLOW | */ FE_UNDERFLOW | FE_INEXACT);
#elif defined(__i386__) && defined(__SSE__)
	_MM_SET_EXCEPTION_MASK(_MM_GET_EXCEPTION_MASK() & ~(_MM_MASK_OVERFLOW|_MM_MASK_INVALID|_MM_MASK_DIV_ZERO));
#endif

	print_cuda_info();

	vector<char *> args(argv, argv + argc);
	bool first_line = true;
	wordexp_t we;
	memset(&we, 0, sizeof(we));

	// read in the parameters file and append to the argv array...
	if (argc > 1 && argv[argc-1][0] == '@') {
		args.pop_back();

		ifstream pfl(argv[argc-1]+1);
		if (!pfl) {
			cerr << "Unable to open parameters file: " << (argv[argc-1]+1) << endl;
			exit(1);
		}

		string ws = " \t";
		while (pfl) {
			string line;
			getline(pfl, line);

			if (line.length() < 1) {
				continue;
			}

			size_t wse = line.find_first_not_of(ws);
			if (wse != string::npos) {
				line = line.substr(wse);
			}

			if (line.length() < 1) {
				continue;
			}

			if (line[0] == '#') {
				continue;
			}

			if (wordexp(line.c_str(), &we, (first_line ? 0 : WRDE_APPEND) | WRDE_SHOWERR) != 0) {
				cerr << "Error parsing line: " << line << endl;
				exit(1);
			}

			first_line = false;
		}

		if (we.we_wordc) {
			char **as = we.we_wordv, **ae = we.we_wordv + we.we_wordc;
			while (as[0][0] == '\0') ++as;
			args.insert(++args.begin(), as, ae);
		}
	}

	model<double> mdl(args.size(), &args[0]);
	mdl.run();

	if (!first_line) {
		wordfree(&we);
	}

	return 0;
}

// Explicit instantiations
template struct model_params<double>;
template struct time_state<double>;

