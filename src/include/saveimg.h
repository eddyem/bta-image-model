/*
 * saveimg.h
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
#ifndef __SAVEIMG_H__
#define __SAVEIMG_H__

#include <stdint.h>
#include "wrapper.h"

typedef uint8_t imtype;
enum{
	IT_FITS = (imtype)(1),
	IT_PNG  = (imtype)(1<<1),
	IT_JPEG = (imtype)(1<<2),
	IT_TIFF = (imtype)(1<<3)
};

int writeimg(char *name, imtype t, size_t width, size_t height, BBox *imbox,
				mirPar *mirror, float *data);

char *createfilename(char* outfile, char* suffix);

#endif // __SAVEIMG_H__
