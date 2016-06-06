/*
 * wrapper.h - CPU/GPU wrapper
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
#ifndef __WRAPPER_H__
#define __WRAPPER_H__

#ifndef EXTERN
	#define EXTERN extern
#endif // EXTERN

#include "mkHartmann.h"

// 1./sqrt(2)
#define DIVSQ2  0.7071068f
// 2*(1+2/sqrt(2)) -- (1+2/sqrt(2)) taken from linear gradient:
//         1 = (2 + 4/sqrt(2)) / (2*x) ==> x = 1 + 2/sqrt(2)
#define GRAD_WEIGHT 4.82842712f

static const size_t MB = 1024*1024; // convert bytes to MB

void noCUDA();
void tryCUDA();
void forceCUDA();
int CUsuccess();
#ifdef CUDA_FOUND
	#define CUDAavailable()		1
#else
	#define CUDAavailable()		0
#endif

double dtime();
void getprops();
EXTERN long throw_random_seed();
EXTERN int fillImage(float *phX, float *phY, size_t ph_sz,
				float *image, size_t imW, size_t imH, BBox *imbox);

mirMask *makeDmask(Diaphragm *d, size_t minSz, mirPar *M, mirDeviations *D);
void freeDmask(mirMask *m);


#define Fn1(A,B) A(x1)
#define Df1(A,B) A(B x1)
#define Fn2(A,B,C) A(x1, x2)
#define Df2(A,B,C) A(B x1, C x2)
#define Fn3(A,B,C,D) A(x1, x2, x3)
#define Df3(A,B,C,D) A(B x1, C x2, D x3)
#define Fn4(A,B,C,D,E) A(x1, x2, x3, x4)
#define Df4(A,B,C,D,E) A(B x1, C x2, D x3, E x4)
#define Fn5(A,B,C,D,E,F) A(x1, x2, x3, x4, x5)
#define Df5(A,B,C,D,E,F) A(B x1, C x2, D x3, E x4, F x5)
#define Fn6(A,B,C,D,E,F,G) A(x1, x2, x3, x4, x5, x6)
#define Df6(A,B,C,D,E,F,G) A(B x1, C x2, D x3, E x4, F x5, G x6)
#define Fn7(A,B,C,D,E,F,G,H) A(x1, x2, x3, x4, x5, x6, x7)
#define Df7(A,B,C,D,E,F,G,H) A(B x1, C x2, D x3, E x4, F x5, G x6, H x7)
#define Fn8(A,B,C,D,E,F,G,H,I) A(x1, x2, x3, x4, x5, x6, x7, x8)
#define Df8(A,B,C,D,E,F,G,H,I) A(B x1, C x2, D x3, E x4, F x5, G x6, H x7, I x8)
#define Fn9(A,B,C,D,E,F,G,H,I,J) A(x1, x2, x3, x4, x5, x6, x7, x8, x9)
#define Df9(A,B,C,D,E,F,G,H,I,J) A(B x1, C x2, D x3, E x4, F x5, G x6, H x7, I x8, J x9)
#define Fn10(A,B,C,D,E,F,G,H,I,J,K) A(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10)
#define Df10(A,B,C,D,E,F,G,H,I,J,K) A(B x1, C x2, D x3, E x4, F x5, G x6, H x7, I x8, J x9, K x10)

#define DEF(N, ...) int Df ## N(__VA_ARGS__)
#define CONCAT(A, B) A ## B
#define FN(N, ...) Fn ## N(__VA_ARGS__)
#define DF(N, ...) Df ## N(__VA_ARGS__)
#define XFUNC(T, X) CONCAT(T, X)
#define FUNC(T, ...) XFUNC(T, FN(__VA_ARGS__))
#define DFUNC(T,...) EXTERN int XFUNC(T, DF(__VA_ARGS__))

#ifdef WRAPPER_C
// even when using cuda in case of fail CUDA init use CPU
static int Only_CPU =
#ifdef CUDA_FOUND
	0
#else
	1
#endif
;

static int CUnoerr, CUforce = 0;

EXTERN int CUgetprops();
EXTERN int CUgetMEM(size_t memsz, size_t *Free, size_t *Total);
EXTERN int CUallocaTest(size_t memsz);

#ifdef CUDA_FOUND
#define SET_F(...) DEF(__VA_ARGS__){					\
	if(!Only_CPU){ CUnoerr = 1;							\
		if(FUNC(CU, __VA_ARGS__)) return 1;				\
		else CUnoerr = 0;								\
	}if(!CUforce && FUNC(CPU, __VA_ARGS__)) return 1;	\
	return 0;											\
}
#else
#define SET_F(...) DEF(__VA_ARGS__){					\
	if(FUNC(CPU, __VA_ARGS__)) return 1;				\
	return 0;											\
}
#endif // CUDA_FOUND
#else
	#define SET_F(...)
#endif // WRAPPER_C

#ifdef CPU_C // file included from CPU.c
	#define BOTH(...) DFUNC(CPU, __VA_ARGS__);
	//#pragma message "CPUC"
#elif defined CUDA_CU //file included from CUDA.cu
	#define BOTH(...) DFUNC(CU, __VA_ARGS__);
#elif defined WRAPPER_C // wrapper.c needs names of both wariants
	#ifndef CUDA_FOUND
		#define BOTH(...) DFUNC(CPU, __VA_ARGS__);
	#else
		#define BOTH(...) DFUNC(CU, __VA_ARGS__); DFUNC(CPU, __VA_ARGS__);
	#endif // CUDA_FOUND
#else // file included from something else - just define a function
	#define BOTH(...) DFUNC(, __VA_ARGS__);
#endif

#define DFN(...) BOTH(__VA_ARGS__) SET_F(__VA_ARGS__)


DFN(3, fillrandarr, size_t, float *, float)

DFN(6, bicubic_interp, float *, float *, size_t, size_t, size_t, size_t)

DFN(4, mkmirDeviat, float *, size_t, float, mirDeviations *)

DFN(7, getPhotonXY, float *, float *, int, mirDeviations *, mirPar *, size_t, BBox *)

//DFN(7, fillImage, float *, float *, size_t, float *, size_t, size_t, BBox *)
#endif // __WRAPPER_H__
