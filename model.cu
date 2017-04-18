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

#include "pow.hpp"
#include "model.hpp"
#include "integrator.hpp"
#include "verlet.hpp"
#include "initializer.hpp"
#include "le_style_initializer.hpp"
#include "grid_funcs.hpp"
#include "energy_outputter.hpp"

#include <cufftw.h>

#include <cstdlib>
#include <cstring>

#include <ctime>

#include <algorithm>

#include <iostream>
#include <fstream>
#include <iomanip>

#include <unistd.h>
#include <errno.h>

#include <sys/stat.h>
#include <sys/types.h>

using namespace std;

// Forward declaration of integrator classes...
template <typename R> class integrator;
template <typename R> class verlet;

template <typename R>
const char *precision_name()
{
	return "unknown";
}

template <>
const char *precision_name<double>()
{
	return "double";
}

template <typename R>
void model<R>::set_output_directory(const char *uodn)
{
	time_t curt = time(NULL);
	tm *curtm = localtime(&curt);

	char outdirname[2048];
	strftime(outdirname, 2048, "output-%Y%m%d%H%M%S", curtm);
	if (!uodn || uodn[0] == '\0') {
		uodn = outdirname;
	}

	mkdir(uodn, 0744);
	chdir(uodn);	
}

/**
 * @page running Running
 *
 * @section Command-Line Parameters
 * SpectRE Usage:
 * @code
 * ./pspectre [-h]
 * ./pspectre [-r] [-l [-B <real>]] [-V] [-H <name>[,<name>]*] [-N <int>] [-L <real>] [-R <int>] [-o <dir name>] [-t <real>[:<real>]] [-T <real>] [-A <real>] [-p <name>=<value>[,<name>=<value>]*] [-e] [-s <name>[,<name>]*] [-S <name>[=<value>][,<name>[=<value>]]*] [-I <name>=<value>[,<name>=<value>]*] [--long] [@<file name>]
 * @endcode
 *
 * @li -h: Display usage information and exit
 * @li -B: The base length scale (default is 1.0 to match LatticeEasy)
 * @li -V: Allow the field variance to change with L
 * @li -e: Use power-law expansion
 * @li -H \<name\>[,\<name\>]*: Use homogeneous (zero variance) initial conditions. Field names are:
 * @code
 *      phi
 *      chi
 * @endcode
 * @li -N \<int\>: The number of grid points per side of the box
 * @li -L \<real\>: The physical size of the box
 * @li -R \<int\>: The random seed
 * @li -o \<dir name\>: Set the output directory name
 * @li -t \<real\>[:\<real\>]: Set dt with an optional start time in program units
 * @li -T \<real\>: The final time in program units
 * @li -A \<real\>: The final scale factor
 * @li -p \<name\>=\<value\>[,\<name\>=\<value\>]*: Set a parameter value. Valid parameters are:
 * @code
 *	ics_scale
 *	H0
 *	ics_eff_size
 *	(a0 can be specified when H0 is specified by appending :\<a0\> to the H0 value;
 *	 Hdot0 can be similarly appended for use with power-law background expansion)
 *	(ics_eff_size is an integer <= N)
 * @endcode
 * @li -s \<name\>[,\<name\>]*: Enable slice output of a variable. Valid variables are:
 * @code
 *	phi
 *	chi
 *	phidot
 *	chidot
 *	V
 *	V_phys
 *	T_phi
 *	T_chi
 *	T_phi_phys
 *	T_chi_phys
 *	G_phi
 *	G_chi
 *	G_phi_phys
 *	G_chi_phys
 *	G_phi_x
 *	G_chi_x
 *	G_phi_phys_x
 *	G_chi_phys_x
 *	G_phi_y
 *	G_chi_y
 *	G_phi_phys_y
 *	G_chi_phys_y
 *	G_phi_z
 *	G_chi_z
 *	G_phi_phys_z
 *	G_chi_phys_z
 *	grad_phi_phys_x
 *	grad_chi_phys_x
 *	grad_phi_phys_y
 *	grad_chi_phys_y
 *	grad_phi_phys_z
 *	grad_chi_phys_z
 *	rho
 *	rho_phys
 *	p
 *	p_phys
 * @endcode
 * @li -S \<name\>[=\<value\>][,\<name\>[=\<value\>]]*: Set a slice output option value. Valid options are:
 * @code
 *	dim
 *	length
 *	skip
 *	avg
 *	fullprec
 *	(avg and fullprec do not take a value)
 * @endcode
 * @li -I \<name\>=\<value\>[:\<real\>][,\<name\>=\<value\>[:\<real\>]]*: Set an output interval with an optional start time. Valid intervals are:
 * @code
 *	scale
 *	energy
 *	screen
 *	slice
 *	all
 *	(intervals are specified as a number of iterations)
 * @endcode
 * @li \@\<file name\>: The name of a parameters file. The parameters file has the same syntax as the command
 * line except that it may be divided among multiple lines and may contain comment lines which begin with
 * a \# character.
 *
 * @par
 * The default parameters model a situation generally similar to the default model provided with DEFROST version 1.1.
 *
 * @section rexamples Examples
 *
 * The following runs the model with the default parameters except that it sets a 128^3 grid with dt = 0.0005. Also,
 * -l selects LE-style initial conditions. -I all=1 
 * sets all output intervals to 1 time step (the default is 25).
 *
 * @code
 * ./pspectre -N 128 -t 0.0005 -l -I all=1
 * @endcode
 *
 * The following runs the model with the default parameters and has binary slice outputs for the energy density, pressure
 * and gravitational potential. The slices to have a length of 32 points per side and were constructed by averaging (not skipping)
 * over every eight-point cube (since the dimension is 3).
 *
 * @code
 * ./pspectre -s rho,p -S dim=3,length=32,skip=1,avg
 * @endcode
 */

template <typename R>
model<R>::model(int argc, char *argv[])
	: fs(64), homo_ic_phi(false), homo_ic_chi(false), seed(1), tf(200.0),
	  scale_interval(25), energy_interval(25),
	  screen_interval(25), slice_interval(25),
	  scale_intervals(scale_interval, 0.0, scale_interval, "t", "scale-factor output interval"),
	  energy_intervals(energy_interval, 0.0, energy_interval, "t", "energy output interval"),
	  screen_intervals(screen_interval, 0.0, screen_interval, "t", "screen output interval"),
	  slice_intervals(slice_interval, 0.0, slice_interval, "t", "slice output interval"),
	  phi("phi"), phidot("phidot"), chi("chi"), chidot("chidot"), gc(0), som(0), ics_scale(1), len0(1.0),
	  vvwl(false), af(0.0), external_H0(false), ics_eff_size(0), phidot0pr(0.0), chidot0pr(0.0)
{
	char *subopts, *value;
	int opt;

	extern char *optarg;
	extern int optopt;
		
	const char *param_names[] = {
		"ics_scale",
		"H0", "ics_eff_size", 0
	};

	const char *interval_names[] = {
		"scale", "energy",
		"screen", "slice",
		"all", 0
	};

	const char *slice_opt_names[] = {
		"dim", "length", "skip",
		"avg", "fullprec", 0
	};
	
	const char *slice_names[] = {
		"phi", "chi", "phidot", "chidot",
		"V", "V_phys",
		"T_phi", "T_chi",
		"T_phi_phys", "T_chi_phys",
		"G_phi", "G_chi",
		"G_phi_phys", "G_chi_phys",
		"G_phi_x", "G_chi_x",
		"G_phi_phys_x", "G_chi_phys_x",
		"G_phi_y", "G_chi_y",
		"G_phi_phys_y", "G_chi_phys_y",
		"G_phi_z", "G_chi_z",
		"G_phi_phys_z", "G_chi_phys_z",
		"grad_phi_phys_x", "grad_chi_phys_x",
		"grad_phi_phys_y", "grad_chi_phys_y",
		"grad_phi_phys_z", "grad_chi_phys_z",
		"rho", "rho_phys", "p", "p_phys",
		0
	};

	const char *field_names[] = {
		"phi", "chi", 0
	};
	
	bool show_usage = false, help_requested = false;

	bool slice_phi = false, slice_chi = false,
		slice_phidot = false, slice_chidot = false,
		slice_V = false, slice_V_phys = false,
		slice_T_phi = false, slice_T_chi = false,
		slice_T_phi_phys = false, slice_T_chi_phys = false,
		slice_G_phi = false, slice_G_chi = false,
		slice_G_phi_phys = false, slice_G_chi_phys = false,
		slice_G_phi_x = false, slice_G_chi_x = false,
		slice_G_phi_phys_x = false, slice_G_chi_phys_x = false,
		slice_G_phi_y = false, slice_G_chi_y = false,
		slice_G_phi_phys_y = false, slice_G_chi_phys_y = false,
		slice_G_phi_z = false, slice_G_chi_z = false,
		slice_G_phi_phys_z = false, slice_G_chi_phys_z = false,
		slice_grad_phi_phys_x = false, slice_grad_chi_phys_x = false,
		slice_grad_phi_phys_y = false, slice_grad_chi_phys_y = false,
		slice_grad_phi_phys_z = false, slice_grad_chi_phys_z = false,
		slice_rho = false, slice_p = false,
		slice_rho_phys = false, slice_p_phys = false;

	int slicedim = 3, slicelength = 0, sliceskip = 1;
	bool sliceaverage = false, sliceflt = true;

	string odn;

	while ((opt = getopt(argc, argv, ":rlVB:hH:ON:P:L:R:p:o:t:T:A:s:S:I:z:e")) != -1) {
		switch (opt) {
		case 'h':
			help_requested = true;
			show_usage = true;
			break;
		case 'B':
			len0 = atof(optarg);
			break;
		case 'V':
			vvwl = true;
			break;
		case 'H':
			subopts = optarg;
			while (*subopts != '\0') {
				int index = getsubopt(&subopts, (char**) field_names, &value);
				if (index < 0) {
					cerr << "Invalid field specification: " << value << endl;
					show_usage = true;
				}			
				else if (!strcmp(field_names[index], "phi")) {
					homo_ic_phi = true;
				}
				else if (!strcmp(field_names[index], "chi")) {
					homo_ic_chi = true;
				}
			}
			break;
		case 'e':
			mp.pwr_exp = true;
			break;
		case 'N':
			fs.n = atoi(optarg);
			break;
		case 'P':
			if (atoi(optarg) != 1) {
				std::cerr << "Support for padding != 1 has been removed."
					  << std::endl;
				exit(1);
			}
			break;
		case 'R':
			seed = atoi(optarg);
			break;
		case 'o':
			odn = optarg;
			break;
		case 'p':
			subopts = optarg;
			while (*subopts != '\0') {
				int index = getsubopt(&subopts, (char**) param_names, &value);
				if (index < 0) {
					cerr << "Invalid parameter specification: " << value << endl;
					show_usage = true;
				}
				else if (!value) {
					cerr << "No value specified for parameter: " << param_names[index] << endl;
					show_usage = true;
				}
				else if (!strcmp(param_names[index], "ics_scale")) {
					ics_scale = atof(value);
				}
				else if (!strcmp(param_names[index], "H0")) {
					// Parse as H0:a0, where a0 is 1 by default.
					char *en;
					R H0 = strtod(value, &en);
					R a0 = 1.0, Hdot0 = 0.0;
					if (*en != 0) {
						char *en2;
						a0 = strtod(en + 1, &en2);
						if (*en2 != 0) {
							Hdot0 = atof(en2 + 1);
						}
					}

					ts.a = a0;
					ts.adot = H0 * ts.a;
					ts.addot = Hdot0 * ts.a;
					external_H0 = true;
				}
				else if (!strcmp(param_names[index], "ics_eff_size")) {
					ics_eff_size = atoi(value);
				}
			}
			break;
		case 't':
			// Parse as dt or dt:start_time
			{
				char *en;
				R dt = strtod(optarg, &en);
				R start_time = 0.0;
				if (*en != 0) {
					start_time = atof(en + 1);
				}

				ts.add_dt(start_time, dt);
			}
			break;
		case 'T':
			tf = atof(optarg);
			break;
		case 'A':
			af = atof(optarg);
			break;
		case 's':
			subopts = optarg;
			while (*subopts != '\0') {
				int index = getsubopt(&subopts, (char**) slice_names, &value);
				if (index < 0) {
					cerr << "Invalid slice output specification: " << value << endl;
					show_usage = true;
				}			
				else if (!strcmp(slice_names[index], "phi")) {
					slice_phi = true;
				}
				else if (!strcmp(slice_names[index], "chi")) {
					slice_chi = true;
				}
				else if (!strcmp(slice_names[index], "phidot")) {
					slice_phidot = true;
				}
				else if (!strcmp(slice_names[index], "chidot")) {
					slice_chidot = true;
				}
				else if (!strcmp(slice_names[index], "V")) {
					slice_V = true;
				}
				else if (!strcmp(slice_names[index], "V_phys")) {
					slice_V_phys = true;
				}
				else if (!strcmp(slice_names[index], "T_phi")) {
					slice_T_phi = true;
				}
				else if (!strcmp(slice_names[index], "T_chi")) {
					slice_T_chi = true;
				}
				else if (!strcmp(slice_names[index], "T_phi_phys")) {
					slice_T_phi_phys = true;
				}
				else if (!strcmp(slice_names[index], "T_chi_phys")) {
					slice_T_chi_phys = true;
				}
				else if (!strcmp(slice_names[index], "G_phi")) {
					slice_G_phi = true;
				}
				else if (!strcmp(slice_names[index], "G_chi")) {
					slice_G_chi = true;
				}
				else if (!strcmp(slice_names[index], "G_phi_phys")) {
					slice_G_phi_phys = true;
				}
				else if (!strcmp(slice_names[index], "G_chi_phys")) {
					slice_G_chi_phys = true;
				}
				else if (!strcmp(slice_names[index], "G_phi_x")) {
					slice_G_phi_x = true;
				}
				else if (!strcmp(slice_names[index], "G_chi_x")) {
					slice_G_chi_x = true;
				}
				else if (!strcmp(slice_names[index], "G_phi_phys_x")) {
					slice_G_phi_phys_x = true;
				}
				else if (!strcmp(slice_names[index], "G_chi_phys_x")) {
					slice_G_chi_phys_x = true;
				}
				else if (!strcmp(slice_names[index], "G_phi_y")) {
					slice_G_phi_y = true;
				}
				else if (!strcmp(slice_names[index], "G_chi_y")) {
					slice_G_chi_y = true;
				}
				else if (!strcmp(slice_names[index], "G_phi_phys_y")) {
					slice_G_phi_phys_y = true;
				}
				else if (!strcmp(slice_names[index], "G_chi_phys_y")) {
					slice_G_chi_phys_y = true;
				}
				else if (!strcmp(slice_names[index], "G_phi_z")) {
					slice_G_phi_z = true;
				}
				else if (!strcmp(slice_names[index], "G_chi_z")) {
					slice_G_chi_z = true;
				}
				else if (!strcmp(slice_names[index], "G_phi_phys_z")) {
					slice_G_phi_phys_z = true;
				}
				else if (!strcmp(slice_names[index], "G_chi_phys_z")) {
					slice_G_chi_phys_z = true;
				}
				else if (!strcmp(slice_names[index], "grad_phi_phys_x")) {
					slice_grad_phi_phys_x = true;
				}
				else if (!strcmp(slice_names[index], "grad_chi_phys_x")) {
					slice_grad_chi_phys_x = true;
				}
				else if (!strcmp(slice_names[index], "grad_phi_phys_y")) {
					slice_grad_phi_phys_y = true;
				}
				else if (!strcmp(slice_names[index], "grad_chi_phys_y")) {
					slice_grad_chi_phys_y = true;
				}
				else if (!strcmp(slice_names[index], "grad_phi_phys_z")) {
					slice_grad_phi_phys_z = true;
				}
				else if (!strcmp(slice_names[index], "grad_chi_phys_z")) {
					slice_grad_chi_phys_z = true;
				}
				else if (!strcmp(slice_names[index], "rho")) {
					slice_rho = true;
				}
				else if (!strcmp(slice_names[index], "rho_phys")) {
					slice_rho_phys = true;
				}
				else if (!strcmp(slice_names[index], "p")) {
					slice_p = true;
				}
				else if (!strcmp(slice_names[index], "p_phys")) {
					slice_p_phys = true;
				}
			}
			break;
		case 'S':
			subopts = optarg;
			while (*subopts != '\0') {
				int index = getsubopt(&subopts, (char**) slice_opt_names, &value);
				if (index < 0) {
					cerr << "Invalid slice option specification: " << value << endl;
					show_usage = true;
				}
				else if (!strcmp(slice_opt_names[index], "avg")) {
					sliceaverage = true;
				}
				else if (!strcmp(slice_opt_names[index], "fullprec")) {
					sliceflt = false;
				}
				else if (!value) {
					cerr << "No value specified for slice option: " << slice_opt_names[index] << endl;
					show_usage = true;
				}
				else if (!strcmp(slice_opt_names[index], "dim")) {
					slicedim = atoi(value);
				}
				else if (!strcmp(slice_opt_names[index], "length")) {
					slicelength = atoi(value);
				}
				else if (!strcmp(slice_opt_names[index], "skip")) {
					sliceskip = atoi(value) + 1;
				}
			}
			break;
		case 'I':
			subopts = optarg;
			while (*subopts != '\0') {
				int index = getsubopt(&subopts, (char**) interval_names, &value);
				if (index < 0) {
					cerr << "Invalid interval specification: " << value << endl;
					show_usage = true;
				}
				else if (!value) {
					cerr << "No value specified for interval: " << param_names[index] << endl;
					show_usage = true;
				}
				else {
					char *en;
					int iv = (int) strtol(value, &en, 10);
					R start_time = 0.0;
					if (*en != '0') {
						start_time = atof(en + 1);
					}

					if (!strcmp(interval_names[index], "scale")) {
						scale_intervals.add_value(start_time, iv);
					}
					else if (!strcmp(interval_names[index], "energy")) {
						energy_intervals.add_value(start_time, iv);
					}
					else if (!strcmp(interval_names[index], "screen")) {
						screen_intervals.add_value(start_time, iv);
					}
					else if (!strcmp(interval_names[index], "slice")) {
						slice_intervals.add_value(start_time, iv);
					}
					else if (!strcmp(interval_names[index], "all")) {
						scale_intervals.add_value(start_time, iv);
						energy_intervals.add_value(start_time, iv);
						screen_intervals.add_value(start_time, iv);
						slice_intervals.add_value(start_time, iv);
					}
				}
			}
			break;
		case ':':
			cerr << "Missing operand for option " << (char) optopt << endl;
			show_usage = true;
			break;
		case '?':
			cerr << "Unrecognized option: " << (char) optopt << endl;
			show_usage = true;
			break;
		}
	}

	if (optind < argc) {
		cerr << "The following options could not be parsed: \"";
		for (int i = optind; i < argc; ++i) {
			cerr << argv[i] << (i < (argc-1) ? " " : "");
		}
		cerr << "\"" << endl;
		show_usage = true;
	}

	if (show_usage) {
		ostream &hout = help_requested ? cout : cerr;
		
		hout << "SpectRE Usage:" << endl;
		hout << argv[0] << " [-h]" << endl;
		hout << argv[0] << " [-r] [-l [-B <real>]] [-V] [-H <name>[,<name>]*] [-N <int>] [-L <real>] [-R <int>] "
			"[-o <dir name>] [-t <real>[:<real>]] [-T <real>] [-A <real>] "
			"[-p <name>=<value>[,<name>=<value>]*] [-e] [-s <name>[,<name>]*] [-S <name>[=<value>][,<name>[=<value>]]*] "
			"[-I <name>=<value>[,<name>=<value>]*] "
			"[@<file name>]"
			<< endl;
		hout << endl;
		
		hout << "\t-h: Display usage information and exit" << endl;
		hout << "\t-B: The base length scale (default is 1.0 to match LatticeEasy)" << endl;
		hout << "\t-V: Allow the field variance to change with L" << endl;
		hout << "\t-e: Use power-law expansion" << endl;

		hout << "\t-H: Use homogeneous (zero variance) initial conditions. Field names are:" << endl;

		for (int i = 0; i < (int) (sizeof(field_names)/sizeof(field_names[0])) - 1; ++i) {
			hout << "\t\t" << field_names[i] << endl;
		}

		hout << "\t-N <int>: The number of grid points per side of the box" << endl;
		hout << "\t-L <real>: The physical size of the box" << endl;
		hout << "\t-R <int>: The random seed" << endl;
		hout << "\t-o <dir name>: Set the output directory name" << endl;
		hout << "\t-t <real>[:<real>]: Set dt with an optional start time in program units" << endl;
		hout << "\t-T <real>: The final time in program units" << endl;
		hout << "\t-A <real>: The final scale factor" << endl;
		hout << "\t-p <name>=<value>[,<name>=<value>]*: Set a parameter value. Valid parameters are:" << endl;
		
		for (int i = 0; i < (int) (sizeof(param_names)/sizeof(param_names[0])) - 1; ++i) {
			hout << "\t\t" << param_names[i] << endl;
		}
		hout << "\t\t(a0 can be specified when H0 is specified by appending :<a0> to the H0 value" << endl;
		hout << "\t\t Hdot0 can be similarly appended for use with power-law background expansion)" << endl;
		hout << "\t\t(ics_eff_size is an integer <= N)" << endl;

		hout << "\t-s <name>[,<name>]*: Enable slice output of a variable. Valid variables are:" << endl;
		
		for (int i = 0; i < (int) (sizeof(slice_names)/sizeof(slice_names[0])) - 1; ++i) {
			hout << "\t\t" << slice_names[i] << endl;
		}

		hout << "\t-S <name>[=<value>][,<name>[=<value>]]*: Set a slice output option value. Valid options are:" << endl;
		
		for (int i = 0; i < (int) (sizeof(slice_opt_names)/sizeof(slice_opt_names[0])) - 1; ++i) {
			hout << "\t\t" << slice_opt_names[i] << endl;
		}
		hout << "\t\t(avg and fullprec do not take a value)" << endl;

		hout << "\t-I <name>=<value>[:<real>][,<name>=<value>[:<real>]]*: Set an output interval with an optional start time. Valid intervals are:" << endl;
		
		for (int i = 0; i < (int) (sizeof(interval_names)/sizeof(interval_names[0])) - 1; ++i) {
			hout << "\t\t" << interval_names[i] << endl;
		}
		hout << "\t\t(intervals are specified as a number of iterations)" << endl;

		hout << "\t@<file name>: The name of a parameters file" << endl;

		exit(help_requested ? 0 : 1);
	}

	ts.finalize_dts();

	scale_intervals.finalize_values();
	energy_intervals.finalize_values();
	screen_intervals.finalize_values();
	slice_intervals.finalize_values();

	srand48(seed);
	
	cout << "Using " << precision_name<R>() << " precision." << endl;

	fs.calculate_size_totals();

	phi.construct(fs, fft_plans);
	phidot.construct(fs, fft_plans);
	chi.construct(fs, fft_plans);
	chidot.construct(fs, fft_plans);

	char *swd = 0; std::size_t swdl = 1024;
	do {
		delete [] swd;
		swdl *= 2;
		swd = new char[swdl];
	} while (!getcwd(swd, swdl) && errno == ERANGE);
	start_wd = swd;
	delete [] swd;

	set_output_directory(odn.c_str());

	gc = new grad_computer<R>(fs, phi, chi, fft_plans);
	som = new slice_output_manager<R>(fs, ts, phi, chi, phidot, chidot, *gc,
					  slicedim, slicelength, sliceskip, sliceaverage, sliceflt);
}

template <typename R>
model<R>::~model()
{
	delete gc;
	delete som;
}

__global__ void field_init_kernel(fftw_complex *mdata, int n, int effpmax, int effpmin)
{
	int x = blockIdx.x;
	int y = blockIdx.y;
	int z = threadIdx.x;
	int px = x <= n/2 ? x : x - n;
	int py = y <= n/2 ? y : y - n;
	int pz = z;
	int idx = z + (n/2+1) * (y + n*x);
	if (px > effpmax || py > effpmax || pz > effpmax ||
	    px < effpmin || py < effpmin || pz < effpmin) {
		mdata[idx][0] = mdata[idx][1] = 0;
	}
}

template <typename R>
void model<R>::set_initial_conditions()
{
	// This is the initial value of adot, and since a = 1 at t = 0, this is the initial value of H.
	// See equation 6.91 in the LatticeEasy manual.

	// Note that the relationship between phidot0 in physical and program units is bit complicated:
	// f_pr = A a^r f => f'_pr = d/dt_pr f_pr = d/dt_pr A a^r f = 1/B a^{-s} d/dt ( A a^r f ) =>
	// f'_pr = A/B a^{-s} d/dt ( a^r f ) = A/B a^{-s} [ a^r f' + r a^{r-1} a' f ] =>
	// f'_pr = A/B [ a^{r-s} f' + r a^{r-1-s} a' f ]
	// So setting a' depends on f'_pr and vice versa, so we'll iterate to convergence...

	if (!external_H0) {
		phidot0pr = RESCALE_A*MP_PHIDOT0;
		chidot0pr = RESCALE_A*MP_CHIDOT0;

		const R adot_thrsh = 1e-14;
		R adot_prev;
		int adot_iters = 0;

		ts.adot = 0.0;
		do {
			adot_prev = ts.adot;

			R hf = 3. * pow<2>(RESCALE_A)/(4. * M_PI);
			R h0norm = 1. / (hf - pow<2>(RESCALE_R*RESCALE_A) * (pow<2>(MP_PHI0) + pow<2>(MP_CHI0)));
			for (int s = -1; s <= 1; s += 2) {
				ts.adot = h0norm * (
					-RESCALE_R*pow<2>(RESCALE_A)*((phidot0pr/RESCALE_A)*MP_PHI0 + (chidot0pr/RESCALE_A)*MP_CHI0) +
					s*sqrt(
						hf * (pow<2>(phidot0pr) + pow<2>(chidot0pr)) +
						2. * model_params::V(RESCALE_A * MP_PHI0, RESCALE_A * MP_CHI0, 1.) *
							(hf - pow<2>(RESCALE_R*RESCALE_A)*(pow<2>(MP_PHI0) + pow<2>(MP_CHI0)))
					)
				);

				if (ts.adot >= 0) {
					break;				
				}
			}

			// Assuming here that a = 1.
			phidot0pr = RESCALE_A*(MP_PHIDOT0 + RESCALE_R*ts.adot*MP_PHI0);
			chidot0pr = RESCALE_A*(MP_CHIDOT0 + RESCALE_R*ts.adot*MP_CHI0);

			++adot_iters;
		} while (adot_iters < 2 || fabs(ts.adot - adot_prev) > ts.adot*adot_thrsh);

		cout << "Initial homogeneous adot (to be corrected later) = " << ts.adot << " (converged in " << adot_iters << " iteration(s))" << endl;
	}
	else {
		phidot0pr = RESCALE_A*(
			pow(ts.a, RESCALE_R - RESCALE_S)*MP_PHIDOT0 +
			RESCALE_R*pow(ts.a, RESCALE_R-RESCALE_S-1)*ts.adot*MP_PHI0
		);
		chidot0pr = RESCALE_A*(
			pow(ts.a, RESCALE_R - RESCALE_S)*MP_CHIDOT0 +
			RESCALE_R*pow(ts.a, RESCALE_R-RESCALE_S-1)*ts.adot*MP_CHI0
		);
	}

	if (!homo_ic_phi || !homo_ic_chi) {
		initializer<R> *init = (initializer<R> *) new le_style_initializer<R>
			(fs, phi, phidot, chi, chidot, ts.adot, len0);

		init->initialize();

		if (vvwl) {
			const R cf = pow(MP_LEN/(le_init ? len0 : R(1.0)), R(1.5))/pow<3>(2 * M_PI/(MP_LEN));

			phi.switch_state(momentum);
			phidot.switch_state(momentum);

			phi.divby(cf);
			phidot.divby(cf);

			chi.switch_state(momentum);
			chidot.switch_state(momentum);

			chi.divby(cf);
			chidot.divby(cf);
		}

		if (homo_ic_phi) {
			phi.switch_state(momentum);
			phidot.switch_state(momentum);
			phi.fill0();
			phidot.fill0();
		}

		if (homo_ic_chi) {
			chi.switch_state(momentum);
			chidot.switch_state(momentum);
			chi.fill0();
			chidot.fill0();
		}
	}
	else {
		phi.switch_state(momentum);
		chi.switch_state(momentum);
		phidot.switch_state(momentum);
		chidot.switch_state(momentum);
	}

	phi.divby(ics_scale);
	chi.divby(ics_scale);
	phidot.divby(ics_scale);
	chidot.divby(ics_scale);
		
	// Note that the 0-mode in Fourier space is the sum over all points in position space.
	phi.mdata[0][0] = RESCALE_A * pow(ts.a, RESCALE_R) * fs.total_gridpoints * MP_PHI0;
	phi.mdata[0][1] = 0.;
	phidot.mdata[0][0] = fs.total_gridpoints * phidot0pr;
	phidot.mdata[0][1] = 0.;
	chi.mdata[0][0] = RESCALE_A * pow(ts.a, RESCALE_R) * fs.total_gridpoints * MP_CHI0;
	chi.mdata[0][1] = 0.;
	chidot.mdata[0][0] = fs.total_gridpoints * chidot0pr;
	chidot.mdata[0][1] = 0.;

	if (ics_eff_size > 0) {
		phi.switch_state(momentum);
		chi.switch_state(momentum);
		phidot.switch_state(momentum);
		chidot.switch_state(momentum);

		// Note that F_{x,y,z} = F*_{N-x,N-y,N-z}
		// What does ics_eff_size mean? In means that all momentum modes will be zeroed out which
		// would not be on a grid of size ics_eff_size.
		int effpmax = ics_eff_size/2;
		int effpmin = -effpmax+1;

		dim3 num_blocks(fs.n, fs.n);
		dim3 num_threads(fs.n/2+1, 1);
		field_init_kernel<<<num_blocks, num_threads>>>(phi.mdata.ptr, fs.n, effpmax, effpmin);
		field_init_kernel<<<num_blocks, num_threads>>>(chi.mdata.ptr, fs.n, effpmax, effpmin);
		field_init_kernel<<<num_blocks, num_threads>>>(phidot.mdata.ptr, fs.n, effpmax, effpmin);
		field_init_kernel<<<num_blocks, num_threads>>>(chidot.mdata.ptr, fs.n, effpmax, effpmin);
	}

	ofstream phiout("phiout-set_init_condition");
	phiout << setprecision(30);
	phiout << scientific;

	for (int x = 0; x < fs.n; x += 8) {
		for (int y = 0; y < fs.n; y += 8) {
			for (int z = 0; z < fs.n; z += 8) {
				int ldl = 2*(fs.n/2+1);
				int idx = z + ldl*(y + fs.n*x);
				phiout << phi.data[idx] << endl;
			}
		}
	}

}

/**
 * @page outputs Output Files
 * All output files generated by SpectRE are placed into a directory named output-YYYYMMDDHHMMSS
 * where YYYY is the current year, etc.
 *
 * @li @ref info_txt
 * @li @ref sf_tsv
 * @li @ref energy_tsv
 * @li @ref slices
 *
 */

/**
 * @page info_txt info.txt
 * The info.txt contains a human-readable summary of the run parameters (both physical and numerical).
 */

/**
 * @page sf_tsv sf.tsv
 * sf.tsv is a tab serarated file with the following fields:
 * @li Program time
 * @li Physical time
 * @li a
 * @li H
 */

template <typename R>
void model<R>::evolve(integrator<R> *ig)
{
	int counter = 0;
	energy_outputter<R> eo(fs, mp, ts, phi, chi, phidot, chidot);
	ofstream scaleof("sf.tsv");
	scaleof << setprecision(30) << fixed;

	if (!external_H0) {
		// Make H self-consistent.
		R adot1 = 0, adot_homo = ts.adot;
		const R avg_rho_thrsh = 1e-14;
		int avg_rho_iters = 0;
		do {
			cout << "here" << endl;
			eo.output(true);
			cout << "avg_rho_phys: " << eo.avg_rho_phys << endl;
			ts.adot = ts.a *
				sqrt( 8./(3. * pow<2>(RESCALE_A) * pow(ts.a, 2. * RESCALE_R)) * M_PI * eo.avg_rho_phys);
			if (!avg_rho_iters) adot1 = ts.adot;
			++avg_rho_iters;
		} while (fabs(eo.avg_rho - R(1.0)) > avg_rho_thrsh);
		cout << "Initial adot converged in " << avg_rho_iters << " iteration(s) to " << ts.adot << ": <rho> = " << eo.avg_rho
			<< " (homo. delta = " << ts.adot - adot_homo << ", from " << adot_homo << ")" 
			<< " (1st iter delta = " << ts.adot - adot1 << ", from " << adot1 << ")" << endl;
	}
	else {
		cout << "User-provided adot = " << ts.adot << ", a = " << ts.a << endl;
	}

	if (mp.pwr_exp) {
		cout << "Using power-law background expansion..." << endl;
		// addot = (G-1)/G 1/a adot^2 1/f^2, f = 1 =>
		// a * addot/adot^2 = 1-1/G =>
		// 1 - a * addot/adot^2 = 1/G
		mp.pwr_exp_G = 1./(1. - ts.a*ts.addot/pow<2>(ts.adot));
		cout << "\tG = " << mp.pwr_exp_G << endl;

		// G = gamma/(gamma*s + 1) =>
		// 1/G = (gamma*s + 1)/gamma =>
		// 1/G = s + 1/gamma =>
		// 1/gamma = 1/G - s
		R gamma = 1./(1./mp.pwr_exp_G - RESCALE_S);
		cout << "Effective power-law exponent: " << gamma << endl;

		// gamma = 2/(3(1+alpha)) =>
		// 2/(3*gamma) = 1 + alpha =>
		// alpha = 2/(3*gamma) - 1
		R alpha = 2./(3.*gamma) - R(1);
		cout << "Effective E.o.S.: p = " << alpha << "*rho" << endl;
	}

	ig->initialize();

	while (ts.t <= tf) {
		if (af > 0.0 && ts.a > af) {
			cout << "Exiting because the scale factor is now " << ts.a << endl;
			break;
		}

		if (counter % scale_interval == 0) {
			scaleof << ts.t << "\t" << RESCALE_B * ts.physical_time << "\t" <<
				ts.a << "\t" << ts.adot/ts.a << "\t" << ts.addot/ts.a << endl;
			scaleof.flush();
		}
		
		if (counter % energy_interval == 0) {
			eo.output();
		}

		if (counter % slice_interval == 0) {
			som->output();
		}

		ig->step();

		ts.advance();
		scale_intervals.advance(ts.t);
		energy_intervals.advance(ts.t);
		screen_intervals.advance(ts.t);
		slice_intervals.advance(ts.t);

		if (counter % screen_interval == 0) {
			cout << ts.t/tf * 100 << " %" << endl;
		}

		++counter;
	}
}

template <typename R>
void model<R>::run()
{
	write_info_file();

	set_initial_conditions();
	
	integrator<R> *ig = (integrator<R> *) new verlet<R>(fs, mp, ts, phi, phidot, chi, chidot, fft_plans);

	cout << "Beginning field evolution..." << endl;
	evolve(ig);
	delete ig;			
}

template <typename R>
void model<R>::write_info_file()
{
	ofstream info_file("info.txt");
	info_file << setprecision(30);
	info_file << scientific;
	
	info_file << "N: " << fs.n << endl;
	info_file << "final time: " << tf << endl;
	info_file << "gamma_phi: " << GAMMA_PHI << endl;
	info_file << "gamma_chi: " << GAMMA_CHI << endl;
	info_file << "lambda_phi: " << LAMBDA_PHI << endl;
	info_file << "lambda_chi: " << LAMBDA_CHI << endl;
	info_file << "m_phi: " << M_PHI << endl;
	info_file << "m_chi: " << M_CHI << endl;
	info_file << "g: " << MP_G << endl;
	info_file << "monodromy_exp_phi: " << MD_E_PHI << endl;
	info_file << "monodromy_exp_chi: " << MD_E_CHI << endl;
	info_file << "monodromy_scale_phi: " << MD_S_PHI << endl;
	info_file << "monodromy_scale_chi: " << MD_S_CHI << endl;
	info_file << "L: " << MP_LEN << endl;
	info_file << "phi0: " << MP_PHI0 << endl;
	info_file << "chi0: " << MP_CHI0 << endl;
	info_file << "phidot0: " << MP_PHIDOT0 << endl;
	info_file << "chidot0: " << MP_CHIDOT0 << endl;

	info_file << endl;

	info_file << "rescale A: " << RESCALE_A << endl;
	info_file << "rescale B: " << RESCALE_B << endl;
	info_file << "rescale r: " << RESCALE_R << endl;
	info_file << "rescale s: " << RESCALE_S << endl;
	if (mp.pwr_exp) {
		info_file << "power-law expansion: yes" <<  endl;
	}

	info_file << endl;

	info_file << "precision: " << precision_name<R>() << endl;
	info_file << "integrator: " << "verlet" << endl;
	info_file << "initial conditions: " <<
		(
			(homo_ic_phi && homo_ic_chi) ? "homogeneous" : "latticeeasy"
		) << endl;
	if (le_init) {
		info_file << "base length scale: " << len0 << endl;
	}
	info_file << "homogeneous initial conditions for phi: " << (homo_ic_phi ? "yes" : "no") << endl;
	info_file << "homogeneous initial conditions for chi: " << (homo_ic_chi ? "yes" : "no") << endl;
	info_file << "initial conditions scale: " << ics_scale << endl;
	info_file << "effective-size cutoff: "; if (ics_eff_size > 0) info_file << ics_eff_size; else info_file << "none"; info_file << endl;
	info_file << "vary field variance with L: " << (vvwl ? "yes" : "no") << endl;

	info_file << "parallelize: cuda" << endl;

	info_file << "N pad factor: " << 1 << endl;
	info_file << "random seed: " << seed << endl;
	
	info_file << endl;

	ts.dt_summary(info_file);

	info_file << endl;
}

// Explicit instantiations
template class model<double>;
