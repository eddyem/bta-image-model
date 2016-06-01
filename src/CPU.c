/*
 * CPU.c - CPU variants (if possible - on OpenMP) of main functions
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

#define CPU_C
#include "mkHartmann.h"
#include "usefull_macros.h"
#include "wrapper.h"

// RANDOM NUMBER generation block  ============================================>
static int rand_initialized = 0;
/*
 * Fills linear array of size sz by random float numbers
 * name: fillrandarr
 * @param sz  - size of array
 * @param arr - array data (allocated outside)
 * @param Amp - amplitude of array
 * @return 0 if failed
 */
int CPUfillrandarr(size_t sz, float *arr, float Amp){
	FNAME();
	if(!rand_initialized){
		srand48(throw_random_seed());
		rand_initialized = 1;
	}
	size_t i;
	OMP_FOR()
	for(i = 0; i < sz; i++)
		arr[i] = (float)drand48() * Amp;
	return 1;
}
// <=====================================  end of random number generation block

// BICUBIC interpolation block  ===============================================>
inline float p_ta(float t, float t2, float t3,
						float a,float b,float c,float d){
	return (2*b + t*(-a+c) + t2*(2*a-5*b+4*c-d) + t3*(-a+3*b-3*c+d)) / 2.f;
}

int CPUbicubic_interp(float *out, float *in,
								size_t oH, size_t oW, size_t iH, size_t iW){
	FNAME();
	float fracX = (float)(iW-1)/(oW-1), fracY = (float)(iH-1)/(oH-1);
	size_t X, Y, ym1,y1,y2, xm1,x1,x2;  // pixel coordinates on output
	size_t Ym = iH - 1, Xm = iW - 1, Pcur;
	float x,y; // coordinates on output in value of input
	OMP_FOR()
	for(Y = 0; Y < oH; Y++){
		// we can't do "y+=fracY" because of possible cumulative error
		y = (float)Y * fracY;
		int y0 = floor(y);
		float pty = y - (float)y0, pty2 = pty*pty, pty3 = pty*pty2;
		ym1 = y0-1; y1 = y0 + 1; y2 = y0 + 2;
		if(y0 == 0) ym1 = 0;
		if(y1 > Ym) y1 = Ym;
		if(y2 > Ym) y2 = Ym;
		for(X = 0, Pcur = Y * oW; X < oW; X++, Pcur++){
			// we can't do "x+=fracX" because of possible cumulative error
			x = (float)X * fracX;
			int x0 = floor(x);
			float ptx = x - (float)x0, ptx2 = ptx*ptx, ptx3 = ptx*ptx2;
			xm1 = x0-1; x1 = x0 + 1; x2 = x0 + 2;
			if(x0 == 0) xm1 = 0;
			if(x1 > Xm) x1 = Xm;
			if(x2 > Xm) x2 = Xm;
			#define TEX(x,y) (in[iW*y + x])
			#define TX(y) p_ta(ptx, ptx2, ptx3, TEX(xm1,y), \
						TEX(x0,y), TEX(x1,y), TEX(x2,y))
			out[Pcur] = p_ta(
				pty, pty2, pty3,
				TX(ym1), TX(y0), TX(y1), TX(y2));
			#undef TX
			#undef TEX
		}
	}
	return 1;
}
// <========================================= end of BICUBIC interpolation block

/**
 * Compute matrices of mirror surface deviation
 * @param map, mapWH - square matrix of surface deviations (in meters) and its size
 * @param mirDia - mirror diameter
 * @param mirDev - deviations:
 * 		.mirWH - size of output square matrices (mirWH x mirWH)
 * 		.mirZ  - matrix of mirror Z variations (add to mirror Z)
 * 		.mirDX, .mirDY - partial derivatives dZ/dx & dZ/dy (add to mirror der.)
 * @return 0 if fails
 */
int CPUmkmirDeviat(float *map, size_t mapWH, float mirDia,
						mirDeviations * mirDev){
	FNAME();
	size_t mirWH = mirDev->mirWH;
	if(!CPUbicubic_interp(mirDev->mirZ,map,mirWH,mirWH,mapWH,mapWH)) return 0;
	;
	return 0;
}

int CPUgetPhotonXY(float *xout _U_, float *yout _U_, int R _U_ ,mirDeviations *D _U_,
				mirPar *mirParms _U_, size_t N_photons _U_, BBox *box _U_){
	FNAME();
	return 0;
}

