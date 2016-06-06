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
	OMP_FOR(shared(arr))
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
	float mirPixSZ = mirDia/((float)(mirWH-1)) * GRAD_WEIGHT;
	float *dZdX = mirDev->mirDX, *dZdY = mirDev->mirDY, *Z = mirDev->mirZ;
	int x, y, maxsz = mirWH - 1;
	// simple mnemonics
	size_t ul=-1-mirWH, ur=1-mirWH, dl=-1+mirWH, dr=1+mirWH, r=1, l=-1, u=-mirWH, d=mirWH;
	OMP_FOR(shared(dZdX, dZdY, Z))
	for(y = 1; y < maxsz; ++y){
		size_t s = mirWH*y+1;
		float *dzdx = &dZdX[s], *dzdy = &dZdY[s], *z = &Z[s];
		for(x = 1; x < maxsz; ++x, ++dzdx, ++dzdy, ++z){
			float dif1 = z[ur]-z[dl], dif2 = z[dr]-z[ul];
			*dzdx = (z[r]-z[l] + DIVSQ2*(dif1 + dif2))/mirPixSZ;
			// REMEMBER!!! Axe Y looks up, so we must change the sign
			*dzdy = -(z[u]-z[d] + DIVSQ2*(dif1 - dif2))/mirPixSZ;
		}
	}
	float *dzdx = dZdX, *dzdy = dZdY;
	// now simply copy neighbours to top row and left column
	for(x = 0; x < mirWH; ++x, ++dzdx, ++dzdy){
		*dzdx = dzdx[mirWH]; *dzdy = dzdy[mirWH];
	}
	dzdx = dZdX, dzdy = dZdY;
	for(y = 0; y < mirWH; ++y, dzdx+=mirWH, dzdy+=mirWH){
		*dzdx = dzdx[1]; *dzdy = dzdy[1];
	}
	*dZdX = (dZdX[1]+dZdX[mirWH])/2;
	*dZdY = (dZdY[1]+dZdY[mirWH])/2;
	return 1;
}

/**
 * Don't use it: without linear interpolation the results are wrong!
 */
int CPUgetPhotonXY(float *xout, float *yout, int R, mirDeviations *D,
				mirPar *mirParms, size_t N_photons, BBox *box){
	FNAME();
	return 0;
	float z0 = mirParms->foc, SZ = sin(mirParms->objZ);
	float F = mirParms->F, _2F = 2.f* F, Rmir = mirParms->D / 2.f, Rmir_2 = Rmir*Rmir;
	float x0 = box->x0, y0 = box->y0, w = box->w, h = box->h;
	int i;
	if(R){ // create random arrays X & Y
		CPUfillrandarr(N_photons, xout, 1.);
		CPUfillrandarr(N_photons, yout, 1.);
	}
	float A = mirParms->Aincl, Z = mirParms->Zincl;
	size_t mirWH = D->mirWH;
	float cA, sA, cZ, sZ; // sin/cos A&Z
	sincosf(A, &sA, &cA);
	sincosf(Z, &sZ, &cZ);
	// light direction vector
	float f[3] = {-SZ*sinf(mirParms->objA), -SZ*cosf(mirParms->objA), -cosf(mirParms->objZ)};
	int rot = 0;
	if((fabs(A) > FLT_EPSILON) || (fabs(Z) > FLT_EPSILON)) rot = 1;
	float pixSZ = mirParms->D / ((float)(mirWH-1)); // "pixel" size on mirror's normales matrix
	OMP_FOR(shared(xout, yout))
	for(i = 0; i < N_photons; ++i){
		float x = x0 + xout[i] * w, y = y0 + yout[i] * h, r2 = x*x + y*y;
		if(r2 > Rmir_2){xout[i] = 1e10f; yout[i] = 1e10f; continue;}
		float z = r2 / F;
		// coordinates on deviation matrix, don't forget about y-mirroring!
		int xOnMat = (x + Rmir) / pixSZ, yOnMat = (y + R) / pixSZ; //yOnMat = (R - y) / pixSZ;
		size_t idx = xOnMat + yOnMat * mirWH;
		// now add z-deviations, nearest interpolation of pre-computed matrix
		z += D->mirZ[idx];
		float normal[3] = { D->mirDX[idx] - x/_2F, D->mirDY[idx] - y/_2F, 1.f};
		float point[3] = {x, y, z};
		void inline rotate(float Mat[3][3], float vec[3]){
			float tmp[3] = {Mat[0][0]*vec[0]+Mat[0][1]*vec[1]+Mat[0][2]*vec[2],
							Mat[1][0]*vec[0]+Mat[1][1]*vec[1]+Mat[1][2]*vec[2],
							Mat[2][0]*vec[0]+Mat[2][1]*vec[1]+Mat[2][2]*vec[2]};
			memmove(vec, tmp, 3*sizeof(float));
		}
		if(rot){ // rotate mirror
			float M[3][3] = {{cA, sA*cZ, sA*sZ},  // rotation matrix
				{-sA, cA*cZ, cA*sZ},
				{0.f, -sZ, cZ}};
			rotate(M, point);
			rotate(M, normal);
		}
		// normalize normal
		{float L = sqrtf(normal[0]*normal[0]+normal[1]*normal[1]+normal[2]*normal[2]);
			normal[0] /= L; normal[1] /= L; normal[2] /= L;}
		// calculate reflection direction vector
		float fn = 2.f*fabs(f[0]*normal[0]+f[1]*normal[1]+f[2]*normal[2]);
		float refl[3] = {fn*normal[0]+f[0], fn*normal[1]+f[1], fn*normal[2]+f[2]};
		float K;
		if(mirParms->dia && mirParms->dia->Nholes){ // there is a diaphragm - test it
			Diaphragm *D = mirParms->dia;
			int S = D->mask->WH; // size of matrix mask
			pixSZ = mirParms->D / (float)S;
			K = (D->Z - point[2]) / refl[2]; // scale to convert normal to vector
			int curX = (int)((x + Rmir) / pixSZ + 0.5f); // coords on mirror mask
			int curY = (int)((y + Rmir) / pixSZ + 0.5f);
			if(curX < 0 || curX >= S || curY < 0 || curY >= S)
				{xout[i] = 1e10f; yout[i] = 1e10f; continue;}
			uint16_t mark = D->mask->data[curY*S + curX];
			if(!mark){xout[i] = 1e10f; yout[i] = 1e10f; continue;}
			x = point[0] + K*refl[0]; // coords on diaphragm
			y = point[1] + K*refl[1];
			do{
				int t = D->holes[mark-1].type;
				BBox *b = &D->holes[mark-1].box;
				float rx = b->w/2.f, ry = b->h/2.f;
				float xc = b->x0 + rx, yc=b->y0 + ry;
				float sx = x - xc, sy = y - yc;
				switch(t){
					case H_SQUARE:
						if(fabs(sx) > rx || fabs(sy) > ry) mark = 0;
					break;
					case H_ELLIPSE:
						if(sx*sx/rx/rx+sy*sy/ry/ry > 1.f) mark = 0;
					break;
					default:
						mark = 0;
				}
			}while(0);
			if(!mark){xout[i] = 1e10f; yout[i] = 1e10f; continue;};
		}
		// OK, test is passed, calculate position on Z0:
		K = (z0 - point[2]) / refl[2];
		x = point[0] + K*refl[0];
		y = point[1] + K*refl[1];
		xout[i] = x;
		yout[i] = y;
	}
	return 1;
}

