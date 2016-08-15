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

/*
__device__ double compute_chi(double a, double chi)
{
	return pow(a, -RESCALE_R) * chi/RESCALE_A;
}

__device__ double compute_phidot(double a, double adot,
				 double phi, double phidot)
{
	// f'_pr = A/B [ a^{r-s} f' + r a^{r-1-s} a' f ] =>
	// f'_pr/A = a^{r-s} f'/B + r a^{r-1-s} a'/B f =>
	// a^{r-s} f'/B = f'_pr/A - r a^{r-1-s} a'/B f =>
	// f'/B = a^{s-r} f'_pr/A - r a^{-1} a'/B f =>
	// f'/B = a^{s-r} f'_pr/A - r a^{-1-r} a'/B f_pr/A

	return pow(a, RESCALE_S - RESCALE_R)*phidot/RESCALE_A
		- RESCALE_R*pow(a, -RESCALE_R-1)*adot * phi/RESCALE_A;
}

__device__ double compute_chidot(double a, double adot,
				 double chi, double chidot)
{
	return pow(a, RESCALE_S - RESCALE_R)*chidot/RESCALE_A
		- RESCALE_R*pow(a, -RESCALE_R-1)*adot * chi/RESCALE_A;
}
*/

/*
__device__ double compute_V(double a, double adot, double phi, double chi)
{
	return compute_energy_scaling(a, adot) * compute_V_phys(a, phi, chi);
}
*/

/*
__device__ double compute_T_phi(double a, double adot, double phi, double phidot)
{
	return compute_energy_scaling(a, adot) * compute_T_phi_phys(a, adot, phi, phidot);
}
*/

/*
__device__ double compute_T_chi(double a, double adot, double chi, double chidot)
{
	return compute_energy_scaling(a, adot) * compute_T_chi_phys(a, adot, chi, chidot);
}
*/

/*
__device__ double compute_G_phi(double a, double adot, double phigradx, double phigrady, double phigradz)
{
	return compute_energy_scaling(a, adot) * compute_G_phi_phys(a, phigradx, phigrady, phigradz);
}
*/

/*
__device__ double compute_G_chi(double a, double adot, double chigradx, double chigrady, double chigradz)
{
	return compute_energy_scaling(a, adot) * compute_G_chi_phys(a, chigradx, chigrady, chigradz);
}
*/

/*
__device__ double compute_G_phi_phys_x(R a, R adot,
				      R phi, R chi, R phidot, R chidot, R phigradx, R chigradx,
				      R phigrady, R chigrady, R phigradz, R chigradz)
{
	return pow(a, -2.*RESCALE_S - 2.) * 0.5 * (
		pow2(phigradx)
		);
}

__device__ double compute_G_phi_x(R a, R adot,
				 R phi, R chi, R phidot, R chidot, R phigradx, R chigradx,
				 R phigrady, R chigrady, R phigradz, R chigradz)
{
	return compute_energy_scaling(a, adot) *
		compute_G_phi_phys_x(a, adot, phi, chi, phidot, chidot,
				     phigradx, chigradx, phigrady, chigrady, phigradz, chigradz);
}

__device__ double compute_G_chi_phys_x(R a, R adot,
				      R phi, R chi, R phidot, R chidot, R phigradx, R chigradx,
				      R phigrady, R chigrady, R phigradz, R chigradz)
{
	return pow(a, -2.*RESCALE_S - 2.) * 0.5 * (
		pow2(chigradx)
		);
}

__device__ double compute_G_chi_x(R a, R adot,
				 R phi, R chi, R phidot, R chidot, R phigradx, R chigradx,
				 R phigrady, R chigrady, R phigradz, R chigradz)
{
	return compute_energy_scaling(a, adot) *
		compute_G_chi_phys_x(a, adot, phi, chi, phidot, chidot,
				     phigradx, chigradx, phigrady, chigrady, phigradz, chigradz);
}

__device__ double compute_G_phi_phys_y(R a, R adot,
				      R phi, R chi, R phidot, R chidot, R phigradx, R chigradx,
				      R phigrady, R chigrady, R phigradz, R chigradz)
{
	return pow(a, -2.*RESCALE_S - 2.) * 0.5 * (
		pow2(phigrady)
		);
}

__device__ double compute_G_phi_y(R a, R adot,
				 R phi, R chi, R phidot, R chidot, R phigradx, R chigradx,
				 R phigrady, R chigrady, R phigradz, R chigradz)
{
	return compute_energy_scaling(a, adot) *
		compute_G_phi_phys_y(a, adot, phi, chi, phidot, chidot,
				     phigradx, chigradx, phigrady, chigrady, phigradz, chigradz);
}

__device__ double compute_G_chi_phys_y(R a, R adot,
				      R phi, R chi, R phidot, R chidot, R phigradx, R chigradx,
				      R phigrady, R chigrady, R phigradz, R chigradz)
{
	return pow(a, -2.*RESCALE_S - 2.) * 0.5 * (
		pow2(chigrady)
		);
}

__device__ double compute_G_chi_y(R a, R adot,
				 R phi, R chi, R phidot, R chidot, R phigradx, R chigradx,
				 R phigrady, R chigrady, R phigradz, R chigradz)
{
	return compute_energy_scaling(a, adot) *
		compute_G_chi_phys_y(a, adot, phi, chi, phidot, chidot,
				     phigradx, chigradx, phigrady, chigrady, phigradz, chigradz);
}

__device__ double compute_G_phi_phys_z(R a, R adot,
				      R phi, R chi, R phidot, R chidot, R phigradx, R chigradx,
				      R phigrady, R chigrady, R phigradz, R chigradz)
{
	return pow(a, -2.*RESCALE_S - 2.) * 0.5 * (
		pow2(phigradz)
		);
}

__device__ double compute_G_phi_z(R a, R adot,
				 R phi, R chi, R phidot, R chidot, R phigradx, R chigradx,
				 R phigrady, R chigrady, R phigradz, R chigradz)
{
	return compute_energy_scaling(a, adot) *
		compute_G_phi_phys_z(a, adot, phi, chi, phidot, chidot,
				     phigradx, chigradx, phigrady, chigrady, phigradz, chigradz);
}

__device__ double compute_G_chi_phys_z(R a, R adot,
				      R phi, R chi, R phidot, R chidot, R phigradx, R chigradx,
				      R phigrady, R chigrady, R phigradz, R chigradz)
{
	return pow(a, -2.*RESCALE_S - 2.) * 0.5 * (
		pow2(chigradz)
		);
}

__device__ double compute_G_chi_z(R a, R adot,
				 R phi, R chi, R phidot, R chidot, R phigradx, R chigradx,
				 R phigrady, R chigrady, R phigradz, R chigradz)
{
	return compute_energy_scaling(a, adot) *
		compute_G_chi_phys_z(a, adot, phi, chi, phidot, chidot,
				     phigradx, chigradx, phigrady, chigrady, phigradz, chigradz);
}

__device__ double compute_grad_phi_phys_x(R a, R adot,
					 R phi, R chi, R phidot, R chidot, R phigradx, R chigradx,
					 R phigrady, R chigrady, R phigradz, R chigradz)
{
	return pow(a, -RESCALE_R)/RESCALE_A * (
		phigradx
		);
}

__device__ double compute_grad_chi_phys_x(R a, R adot,
					 R phi, R chi, R phidot, R chidot, R phigradx, R chigradx,
					 R phigrady, R chigrady, R phigradz, R chigradz)
{
	return pow(a, -RESCALE_R)/RESCALE_A * (
		chigradx
		);
}

__device__ double compute_grad_phi_phys_y(R a, R adot,
					 R phi, R chi, R phidot, R chidot, R phigradx, R chigradx,
					 R phigrady, R chigrady, R phigradz, R chigradz)
{
	return pow(a, -RESCALE_R)/RESCALE_A * (
		phigrady
		);
}

__device__ double compute_grad_chi_phys_y(R a, R adot,
					 R phi, R chi, R phidot, R chidot, R phigradx, R chigradx,
					 R phigrady, R chigrady, R phigradz, R chigradz)
{
	return pow(a, -RESCALE_R)/RESCALE_A * (
		chigrady
		);
}

__device__ double compute_grad_phi_phys_z(R a, R adot,
					 R phi, R chi, R phidot, R chidot, R phigradx, R chigradx,
					 R phigrady, R chigrady, R phigradz, R chigradz)
{
	return pow(a, -RESCALE_R)/RESCALE_A * (
		phigradz
		);
}

__device__ double compute_grad_chi_phys_z(R a, R adot,
					 R phi, R chi, R phidot, R chidot, R phigradx, R chigradx,
					 R phigrady, R chigrady, R phigradz, R chigradz)
{
	return pow(a, -RESCALE_R)/RESCALE_A * (
		chigradz
		);
}

__device__ double compute_p_phys(R a, R adot,
				R phi, R chi, R phidot, R chidot, R phigradx, R chigradx,
				R phigrady, R chigrady, R phigradz, R chigradz)
{
	return -compute_V_phys(a, phi, chi) +
		compute_T_phi_phys(a, adot, phi, phidot) +
		compute_T_chi_phys(a, adot, chi, chidot) -
		compute_G_phi_phys(a, phigradx, phigrady, phigradz) -
		compute_G_chi_phys(a, chigradx, chigrady, chigradz)/3.;
}

__device__ double compute_p(R a, R adot,
			   R phi, R chi, R phidot, R chidot, R phigradx, R chigradx,
			   R phigrady, R chigrady, R phigradz, R chigradz)
{
	return compute_energy_scaling(a, adot) *
		compute_p_phys(a, adot, phi, chi, phidot, chidot,
			       phigradx, chigradx, phigrady, chigrady, phigradz, chigradz);
}
*/
