/*
 * mkHartmann.h - main header file with common definitions
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
#ifndef __MKHARTMANN_H__
#define __MKHARTMANN_H__

#ifndef _GNU_SOURCE
	#define _GNU_SOURCE
#endif
#include <features.h>
#include <assert.h>			// assert
#include <stdlib.h>			// malloc/free etc
#include <stdio.h>			// printf etc
#include <stdint.h>			// int types
#include <stdarg.h>			// vprintf
#include <string.h>			// memset etc
#include <unistd.h>			// file operations
#include <libintl.h>		// gettext
#include <limits.h>			// LONG_MAX
#include <sys/time.h>		// gettimeofday
#include <sys/types.h>		// open
#include <sys/stat.h>		// open
#include <fcntl.h>			// open
#include <math.h>			// floor & other math
#include <err.h>			// err
#include <errno.h>
#include <sys/mman.h>		// munmap
#include <time.h>			// time, ctime

#ifndef FLT_EPSILON
	#define FLT_EPSILON 1.19209E-07
#endif

#ifndef GETTEXT_PACKAGE
	#define GETTEXT_PACKAGE "mkHartmann"
#endif
#ifndef PACKAGE_VERSION
	#define PACKAGE_VERSION "0.0.1"
#endif
#ifndef LOCALEDIR
	#define LOCALEDIR "/usr/share/locale/"
#endif

#define _(String)				gettext(String)
#define gettext_noop(String)	String
#define N_(String)				gettext_noop(String)

#define _U_    __attribute__((__unused__))

#ifndef THREAD_NUMBER
	#define THREAD_NUMBER 2
#endif

#ifdef OMP
	#ifndef OMP_NUM_THREADS
		#define OMP_NUM_THREADS THREAD_NUMBER
	#endif
	#define Stringify(x) #x
	#define OMP_FOR(x) _Pragma(Stringify(omp parallel for x))
#else
	#define OMP_FOR(x)
#endif // OMP

extern int globErr;
#define ERR(...) do{globErr=errno; _WARN(__VA_ARGS__); exit(-1);}while(0)
#define ERRX(...) do{globErr=0; _WARN(__VA_ARGS__); exit(-1);}while(0)
#define WARN(...) do{globErr=errno; _WARN(__VA_ARGS__);}while(0)
#define WARNX(...) do{globErr=0; _WARN(__VA_ARGS__);}while(0)

// debug mode, -DEBUG
#ifdef EBUG
	#define FNAME() fprintf(stderr, "\n%s (%s, line %d)\n", __func__, __FILE__, __LINE__)
	#define DBG(...) do{fprintf(stderr, "%s (%s, line %d): ", __func__, __FILE__, __LINE__); \
					fprintf(stderr, __VA_ARGS__);			\
					fprintf(stderr, "\n");} while(0)
#else
	#define FNAME()	 do{}while(0)
	#define DBG(...) do{}while(0)
#endif //EBUG

#define ALLOC(type, var, size)  type * var = ((type *)my_alloc(size, sizeof(type)))
#define MALLOC(type, size) ((type *)my_alloc(size, sizeof(type)))
#define FREE(ptr)			do{free(ptr); ptr = NULL;}while(0)

#ifndef EXTERN  // file wasn't included from CUDA.cu
	#define EXTERN extern
#endif

#define RAD 57.2957795130823
#define D2R(x) ((x) / RAD)
#define R2D(x) ((x) * RAD)

// STRUCTURES definition

// bounding box
typedef struct{
	float x0; // left border
	float y0; // lower border
	float w;  // width
	float h;  // height
} BBox;

// mirror deviation parameters
typedef struct{
	float *mirZ; // Z
	float *mirDX;// dZ/dX
	float *mirDY;// dZ/dY
	size_t mirWH;// size: mirWH x mirWH
}mirDeviations;

// type of hole in diaphragm
typedef enum{
	 H_SQUARE  // square hole
	,H_ELLIPSE // elliptic hole
	,H_UNDEF   // error: can't define type
} HoleType;

// hole in diaphragm
typedef struct{
	BBox box; // bounding box of hole
	int type; // type, in case of round hole borders of box are tangents to hole
} aHole;

// mirror mask for given diaphragm
typedef struct{
	size_t WH;       // size of mask: WH x WH
	uint16_t *data;  // mask data
}mirMask;

// diaphragm itself
typedef struct{
	BBox box;       // bounding box of diaphragm, must be a little larger of its contents
	aHole *holes;   // array of holes
	int Nholes;     // size of array
	float Z;        // z-coordinate of diaphragm
	mirMask *mask;
} Diaphragm;

// parameters of mirror
typedef struct{
	float D;     // diameter
	float F;     // focus
	float Zincl; // inclination from Z axe (radians)
	float Aincl; // azimuth of inclination (radians)
	float objA;  // azimuth of object (radians)
	float objZ;  // zenith of object (radians)
	float foc;   // Z-coordinate of light receiver
	Diaphragm *dia; // diaphragm or NULL
} mirPar;

// functions for color output in tty & no-color in pipes
EXTERN int (*red)(const char *fmt, ...);
EXTERN int (*_WARN)(const char *fmt, ...);
EXTERN int (*green)(const char *fmt, ...);
void * my_alloc(size_t N, size_t S);


#endif // __MKHARTMANN_H__

/*
 *
 *
 * <===========================================================================>
 *
 *
 */

