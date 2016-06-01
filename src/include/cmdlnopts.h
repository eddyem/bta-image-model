/*
 * cmdlnopts.h - comand line options for parceargs
 *
 * Copyright 2013 Edward V. Emelianoff <eddy@sao.ru>
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
 * MA 02110-1301, USA.
 */

#pragma once
#ifndef __CMDLNOPTS_H__
#define __CMDLNOPTS_H__

#include "parseargs.h"
#include "mkHartmann.h"
#include "saveimg.h"

typedef struct{
	int S_dev;		// size of initial array of surface deviations
	int S_interp;	// size of interpolated S0
	int S_image;	// resulting image size
	int N_phot;		// amount of photons falled to one pixel of S1 by one iteration
	int N_iter;		// iterations number
	int randMask;	// add to mask random numbers
	float randAmp;	// amplitude of added random noice
	float CCDW;		// CCD width
	float CCDH;		//           and height (in millimeters)
	imtype it;		// output image type
	char *dev_filename;// input deviations file name
	char *holes_filename;// input holes file name
	char *outfile;	// output file name
	mirPar *Mirror;	// mirror parameters
} glob_pars;

// default parameters
extern glob_pars const Gdefault;
extern mirPar const Mdefault;

glob_pars *parse_args(int argc, char **argv);

#endif // __CMDLNOPTS_H__
