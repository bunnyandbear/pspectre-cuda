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
 * @brief The physical, static model parameters.
 */

#ifndef MODEL_PARAMS
#define MODEL_PARAMS

#include "pow.hpp"

#include <iostream>
#include <string>
#include <set>

#include <cmath>
#include <cufftw.h>

#define NGRIDSIZE			(256)
#define NTOTAL_GRIDPOINTS		(NGRIDSIZE * NGRIDSIZE * NGRIDSIZE)
#define NTOTAL_MOMENTUM_GRIDPOINTS	(NGRIDSIZE * NGRIDSIZE * (NGRIDSIZE/2 + 1))
#define POWER_LENGTH			(int(sqrt(3)*0.5*NGRIDSIZE) + 1)

// Monodromy potential: c s^2 ([ 1 + (phi/s)^2 ]^e - 1)
// = c ( e phi^2 + (1/2) s^{-2} (e^2 - e) phi^4 + (1/6) s^{-4} (e^3 - 3 e^2 + 2 e) phi^6 + ...

#define M_PHI (0.000022139442217092964)
#define MP_LEN (3.0)
#define MP_PHI0 (0.08211141279686777)
#define MP_PHIDOT0 (0.0176648818699687)
#define RESCALE_A (1/MP_PHI0)
#define RESCALE_B (M_PHI)
#define RESCALE_S (0)
#define RESCALE_R (1.5)
#define MP_DP (2.*M_PI/MP_LEN)

#define ENABLE_PHI3	(0)
#if ENABLE_PHI3 == 0
#define IF_PHI3(x)
#define IF_PHI3_ARG(x,y)
#else
#define IF_PHI3(x)	x
#define IF_PHI3_ARG(x,y) x,y
#define LAMBDA_PHI	(0)
#endif	// ENABLE_PHI3

#define ENABLE_PHI5	(0)
#if ENABLE_PHI5 == 0
#define IF_PHI5(x)
#define IF_PHI5_ARG(x,y)
#else
#define IF_PHI5(x)	x
#define IF_PHI5_ARG(x,y) x,y
#define GAMMA_PHI	(0)
#endif	// ENABLE_PHI5

#define ENABLE_MD_PHI	(1)
#if ENABLE_MD_PHI == 0
#define IF_MD_PHI(x)
#define IF_NOT_MD_PHI(x) x
#define IF_MD_PHI_ARG(x,y)
#else
#define IF_MD_PHI(x)	x
#define IF_NOT_MD_PHI(x)
#define IF_MD_PHI_ARG(x,y) x,y
#define MD_E_PHI	(0.5)
#define MD_C_PHI	(0.5* M_PHI * M_PHI / MD_E_PHI)
#define MD_S_PHI	(0.003989422804014327)
#endif	// ENABLE_MD_PHI

#define ENABLE_CHI	(0)
#if ENABLE_CHI == 0
#define IF_CHI(x)
#define IF_CHI_ARG(x,y)
#define IF_CHI3(x)
#define IF_CHI3_ARG(x,y)
#define IF_CHI5(x)
#define IF_CHI5_ARG(x,y)
#define IF_MD_CHI(x)
#define IF_MD_CHI_ARG(x,y)
#else
#define IF_CHI(x)	x
#define IF_CHI_ARG(x,y) x,y
#define MP_G		(0)
#define M_CHI		(0)
#define MP_CHI0		(0)
#define MP_CHIDOT0	(0)

#define ENABLE_CHI3	(0)
#if ENABLE_CHI3 == 0
#define IF_CHI3(x)
#define IF_CHI3_ARG(x,y)
#else
#define IF_CHI3(x)	x
#define IF_CHI3_ARG(x,y) x,y
#define LAMBDA_CHI	(0)
#endif	// ENABLE_CHI3

#define ENABLE_CHI5	(0)
#if ENABLE_CHI5 == 0
#define IF_CHI5(x)
#define IF_CHI5_ARG(x,y)
#else
#define IF_CHI5(x)	x
#define IF_CHI5_ARG(x,y) x,y
#define GAMMA_CHI	(0)
#endif	// ENABLE_CHI5

#define ENABLE_MD_CHI	(0)
#if ENABLE_MD_CHI == 0
#define IF_MD_CHI(x)
#define IF_NOT_MD_CHI(x) x
#define IF_MD_CHI_ARG(x,y)
#else
#define IF_MD_CHI(x)	x
#define IF_NOT_MD_CHI(x)
#define IF_MD_CHI_ARG(x,y) x,y
#define MD_E_CHI	(0)
#define MD_C_CHI	(0)
#define MD_S_CHI	(1)
#endif	// ENABLE_MD_CHI

#endif	// ENABLE_CHI

#ifdef DOT0_IN_PLANCK
#define MP_PHIDOT0 (MP_PHIDOT0 * RESCALE_B)
#define MP_CHIDOT0 (MP_CHIDOT0 * RESCALE_B)
#endif

#define BEGIN_OUTPUT_TIME (350)

/*
  Below are the defaults that are compatible with DEFROST:

  len = 10.0;
  lambda_phi = 0.0;
  lambda_chi = 0.0;
  GAMMA_PHI = 0.0;
  GAMMA_CHI = 0.0;
  m_phi = (1./2.e5)/sqrt(8. * M_PI);
  m_chi = 0.0;
  g = sqrt(1e4 * 8. * M_PI * pow<2>(m_phi));
  phi0 = 1.0093430384226378929425913902459/sqrt(8. * M_PI);
  chi0 = 0.;
  phidot0 = 0.;
  chidot0 = 0.;

  md_e_phi = 0.0;
  md_e_chi = 0.0;
  md_c_phi = 0.0;
  md_c_chi = 0.0;
  md_s_phi = 1.0;
  md_s_chi = 1.0;
*/

struct model_params
{
	model_params()
	{
		using namespace std;

		pwr_exp = false;
		pwr_exp_G = 0.0;
	}

	/**
	 * Returns the value of the field potential at a point given the values of the fields at that point.
	 * The field values are sent in program units, and the potential is returned in program units.
	 * This is equation 6.5 from the LatticeEasy manual.
	 */

	__device__ __host__ static double V(double phi, IF_CHI_ARG(double chi,) double a_t)
	{
		const double tophys = 1./RESCALE_A * pow(a_t, -RESCALE_R);
		const double phi_phys = tophys * phi;
		const double md_phi = 
			IF_MD_PHI(MD_C_PHI * pow2(MD_S_PHI) *
				  (pow(1.0 + pow2(phi_phys / MD_S_PHI), MD_E_PHI) - 1.0))
			IF_NOT_MD_PHI(0.5 * pow2(M_PHI * phi_phys));
		IF_CHI(const double chi_phys = tophys * chi;
		       const double md_chi = 
		       IF_MD_CHI(MD_C_CHI * pow2(MD_S_CHI) *
				 (pow(1.0 + pow2(chi_phys / MD_S_CHI), MD_E_CHI) - 1.0))
		       IF_NOT_MD_CHI(0.5 * pow2(M_CHI * chi_phys))
			);
		return pow2(RESCALE_A / RESCALE_B) * pow(a_t, -2. * RESCALE_S + 2. * RESCALE_R) *
			(md_phi IF_CHI(+ md_chi)
			 IF_PHI3(+ 0.25*LAMBDA_PHI*pow4(phi_phys))
			 IF_CHI3(+ 0.25*LAMBDA_CHI*pow4(chi_phys))
			 IF_CHI(+ 0.5*pow2(MP_G * phi_phys * chi_phys))
			 IF_PHI5(+ GAMMA_PHI*pow6(phi_phys)/6.0)
			 IF_CHI5(+ GAMMA_CHI*pow6(chi_phys)/6.0));
	}

	/*
	 * Returns addot based on a power-law background expansion.
	 * See equation 6.46 and 6.49 of the LatticeEasy manual.
	 */

	double adoubledot_pwr_exp(double t, double a_t, double adot_t)
	{
		double f = 1./pwr_exp_G * adot_t/a_t * t + 1.;
		return (pwr_exp_G - 1.)/(pwr_exp_G*pow<2>(f)) * pow<2>(adot_t)/a_t; 
	}

	/**
	 * Returns the second time derivative of the scale factor in program units.
	 * See equation 6.26 of the LatticeEasy manual.
	 */

	double adoubledot(double t, double a_t, double adot_t,
			  double avg_gradient_phi, IF_CHI_ARG(double avg_gradient_chi,) double avg_V)
	{
		using namespace std;

		if (pwr_exp) {
			return adoubledot_pwr_exp(t, a_t, adot_t);
		}
		
		return (
			(-RESCALE_S - 2.)*pow<2>(adot_t)/a_t +
			8. * M_PI / pow<2>(RESCALE_A) * pow(a_t, -2.* RESCALE_S - 2. * RESCALE_R - 1.)*(
				1./3. * (avg_gradient_phi IF_CHI(+ avg_gradient_chi)) + pow(a_t, 2*RESCALE_S + 2.)*avg_V
			)
		);
	}

	/** Returns the second time derivative of the scale factor in program units at a half-time-step.
	 * See equation 6.35/6.36 of the LatticeEasy manual.
	 */

	double adoubledot_staggered(double t, double dt, double a_t, double adot_t,
				    double avg_gradient_phi, IF_CHI_ARG(double avg_gradient_chi,) double avg_V)
	{
		using namespace std;

		if (pwr_exp) {
			return adoubledot_pwr_exp(t, a_t, adot_t);
		}
		
		return (
			-2. * adot_t - 2. * a_t / (dt * (RESCALE_S + 2.)) *
			(1. - sqrt(1. + 2. * dt * (RESCALE_S + 2.) * adot_t/a_t +
			pow<2>(dt) * (RESCALE_S + 2.) * 8. * M_PI / pow<2>(RESCALE_A) * pow(a_t, -2.* RESCALE_S - 2. * RESCALE_R - 2.)*(
				1./3. * (avg_gradient_phi IF_CHI(+ avg_gradient_chi)) + pow(a_t, 2*RESCALE_S + 2.)*avg_V
			)))
		)/dt;
	}

	bool pwr_exp;
	double pwr_exp_G;
};

#endif // MODEL_PARAMS
