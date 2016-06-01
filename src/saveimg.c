/*
 * saveimg.c - functions to save data in png and FITS formats
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

#include "mkHartmann.h"
#include "usefull_macros.h"
#include "cmdlnopts.h" // for flag "-f", which will tell to rewrite existing file
#include "saveimg.h"
#if defined __PNG && __PNG == TRUE
	#include <png.h>
#endif // PNG
#if defined __JPEG && __JPEG == TRUE
	#include <jpeglib.h>
#endif // JPEG
#if defined __TIFF && __TIFF == TRUE
	#include <tiffio.h>
#endif // TIFF
#include <fitsio.h>

//char *strdup(const char *s);

#define TRYFITS(f, ...)						\
do{int status = 0; f(__VA_ARGS__, &status);	\
	if (status){ ret = 0;					\
		fits_report_error(stderr, status);	\
		goto returning;}					\
}while(0)
#define WRITEKEY(...)						\
do{ int status = 0;							\
	fits_write_key(fp, __VA_ARGS__, &status);\
	if(status)								\
		fits_report_error(stderr, status);	\
}while(0)

int rewrite_ifexists = 0; // don't rewrite existing files
/**
 * Create filename as   outfile + number + "." + suffix
 * 		number -- any number from 1 to 9999
 * This function simply returns "outfile.suffix" when "-f" option is set
 *
 * @param outfile - file name
 * @param suffix  - file suffix
 * @return created filename or NULL
 */
char *createfilename(char* outfile, char* suffix){
	FNAME();
	struct stat filestat;
	char buff[256], sfx[32];
	if(suffix) snprintf(sfx, 31, ".%s", suffix);
	else sfx[0] = 0; // no suffix
	if(rewrite_ifexists){ // there was key "-f": simply return copy of filename
		if(snprintf(buff, 255, "%s%s", outfile, sfx) < 1){
			DBG("error");
			return NULL;
		}
		DBG("(-f present) filename: %s", buff);
		return strdup(buff);
	}
	int num;
	if(!outfile) outfile = "";
	for(num = 1; num < 10000; num++){
		if(snprintf(buff, 255, "%s_%04d%s", outfile, num, sfx) < 1){
			DBG("error");
			return NULL;
		}
		if(stat(buff, &filestat)){ // || !S_ISREG(filestat.st_mode)) // OK, file not exists
			DBG("filename: %s", buff);
			return strdup(buff);
		}
	}
	DBG("n: %s\n", buff);
	WARN("Oops! All  numbers are busy or other error!");
	return NULL;
}

typedef struct{
	float *image;
	double min;
	double max;
	double avr;
	double std;
} ImStat;

static ImStat glob_stat;

/**
 * compute basics image statictics
 * @param img - image data
 * @param size - image size W*H
 * @return
 */
void get_stat(float *img, size_t size){
	FNAME();
	if(glob_stat.image == img) return;
	size_t i;
	double pv, sum=0., sum2=0., sz=(double)size;
	double max = -1., min = 1e15;
	for(i = 0; i < size; i++){
		pv = (double) *img++;
		sum += pv;
		sum2 += (pv * pv);
		if(max < pv) max = pv;
		if(min > pv) min = pv;
	}
	glob_stat.image = img;
	glob_stat.avr = sum/sz;
	glob_stat.std = sqrt(fabs(sum2/sz - glob_stat.avr*glob_stat.avr));
	glob_stat.max = max;
	glob_stat.min = min;
	DBG("Image stat: max=%g, min=%g, avr=%g, std=%g", max, min, glob_stat.avr, glob_stat.std);
}

/**
 * Save data to fits file
 * @param filename - filename to save to
 * @param width, height - image size
 * @param imbox - image bounding box
 * @data  image data
 * @return 0 if failed
 */
int writefits(char *filename, size_t width, size_t height, BBox *imbox,
				mirPar *mirror, float *data){
	FNAME();
	long naxes[2] = {width, height};
	static char* newname = NULL;
	char buf[80];
	int ret = 1;
	double dX, dY;
	if(imbox){
		dX = imbox->w / (double)(width - 1);
		dY = imbox->h / (double)(height - 1);
	}
	time_t savetime = time(NULL);
	fitsfile *fp;
	assert(filename);
	newname = realloc(newname, strlen(filename + 2));
	sprintf(newname, "!%s", filename); // say cfitsio that file could be rewritten
	TRYFITS(fits_create_file, &fp, newname);
	TRYFITS(fits_create_img, fp, FLOAT_IMG, 2, naxes);
	// FILE / Input file original name
	WRITEKEY(TSTRING, "FILE", filename, "Input file original name");
	WRITEKEY(TSTRING, "DETECTOR", "Hartmann model", "Detector model");
	if(imbox){
		snprintf(buf, 79, "%.2g x %.2g", dX * 1e6, dY * 1e6);
		// PXSIZE / pixel size
		WRITEKEY(TSTRING, "PXSIZE", buf, "Pixel size in mkm");
		// XPIXELSZ, YPIXELSZ -- the same
		WRITEKEY(TDOUBLE, "XPIXELSZ", &dX, "X pixel size in m");
		WRITEKEY(TDOUBLE, "YPIXELSZ", &dY, "Y pixel size in m");
		// LBCX, LBCY / Coordinates of left bottom corner
		WRITEKEY(TFLOAT, "LBCX", &imbox->x0, "X of left bottom corner");
		WRITEKEY(TFLOAT, "LBCY", &imbox->y0, "Y of left bottom corner");
	}
	// IMAGETYP / object, flat, dark, bias, scan, eta, neon, push
	WRITEKEY(TSTRING, "IMAGETYP", "object", "Image type");
	// DATAMAX, DATAMIN / Max,min pixel value
	WRITEKEY(TDOUBLE, "DATAMAX", &glob_stat.max, "Max data value");
	WRITEKEY(TDOUBLE, "DATAMIN", &glob_stat.min, "Min data value");
	// Some Statistics
	WRITEKEY(TDOUBLE, "DATAAVR", &glob_stat.avr, "Average data value");
	WRITEKEY(TDOUBLE, "DATASTD", &glob_stat.std, "Standart deviation of data value");
	// DATE / Creation date (YYYY-MM-DDThh:mm:ss, UTC)
	strftime(buf, 79, "%Y-%m-%dT%H:%M:%S", gmtime(&savetime));
	WRITEKEY(TSTRING, "DATE", buf, "Creation date (YYYY-MM-DDThh:mm:ss, UTC)");
	// DATE-OBS / DATE OF OBS.
	WRITEKEY(TSTRING, "DATE-OBS", buf, "DATE OF OBS. (YYYY-MM-DDThh:mm:ss, local)");
	// OBJECT  / Object name
	WRITEKEY(TSTRING, "OBJECT", "Modeled object", "Object name");
	// BINNING / Binning
	WRITEKEY(TSTRING, "XBIN", "1", "Horizontal binning");
	WRITEKEY(TSTRING, "YBIN", "1", "Vertical binning");
	// PROG-ID / Observation program identifier
	WRITEKEY(TSTRING, "PROG-ID", "BTA Hartmann modeling", "Observation program identifier");
	// AUTHOR / Author of the program
	WRITEKEY(TSTRING, "AUTHOR", "Edward V. Emelianov", "Author of the program");
	if(mirror){
		WRITEKEY(TFLOAT, "MIRDIAM",  &mirror->D,    "Mirror diameter");
		WRITEKEY(TFLOAT, "MIRFOC",   &mirror->F,    "Mirror focus ratio");
		WRITEKEY(TFLOAT, "MIRZINCL", &mirror->Zincl,"Mirror inclination from Z axe");
		WRITEKEY(TFLOAT, "MIRAINCL", &mirror->Aincl,"Azimuth of mirror inclination");
		WRITEKEY(TFLOAT, "A",        &mirror->objA, "Object's azimuth");
		WRITEKEY(TFLOAT, "Z",        &mirror->objZ, "Object's zenith distance");
		WRITEKEY(TFLOAT, "FOCUS",    &mirror->foc,  "Z-coordinate of light receiver");
	}

	TRYFITS(fits_write_img, fp, TFLOAT, 1, width * height, data);
	TRYFITS(fits_close_file, fp);

returning:
	return ret;
}

static uint8_t *rowptr = NULL;
uint8_t *processRow(float *irow, size_t width, float min, float wd){
	FREE(rowptr);
	//float umax = ((float)(UINT16_MAX-1));
	rowptr = MALLOC(uint8_t, width * 3);
	OMP_FOR()
	for(size_t i = 0; i < width; i++){
		double gray = ((double)(irow[i] - min))/((double)wd);
		if(gray == 0.) continue;
		int G = (int)(gray * 4.);
		double x = 4.*gray - (double)G;
		uint8_t *ptr = &rowptr[i*3];
		uint8_t r = 0, g = 0, b = 0;
		switch(G){
			case 0:
				g = (uint8_t)(255. * x + 0.5);
				b = 255;
			break;
			case 1:
				g = 255;
				b = (uint8_t)(255. * (1. - x) + 0.5);
			break;
			case 2:
				r = (uint8_t)(255. * x + 0.5);
				g = 255;
			break;
			case 3:
				r = 255;
				g = (uint8_t)(255. * (1. - x) + 0.5);
			break;
			default:
				r = 255;
		}
		ptr[0] = r; ptr[1] = g; ptr[2] = b;
		//ptr[0] = ptr[1] = ptr[2] = gray*255;
	}
	return rowptr;
}

int writepng(char *filename, size_t width, size_t height, BBox *imbox,
				mirPar *mirror, float *data){
	FNAME();
	int ret = 1;
#if defined __PNG && __PNG == TRUE
	FILE *fp = NULL;
	png_structp pngptr = NULL;
	png_infop infoptr = NULL;
	float min = glob_stat.min, wd = glob_stat.max - min;
	float *row;

	if ((fp = fopen(filename, "w")) == NULL){
		perror("Can't open png file");
		ret = 0;
		goto done;
	}
	if ((pngptr = png_create_write_struct(PNG_LIBPNG_VER_STRING,
							NULL, NULL, NULL)) == NULL){
		perror("Can't create png structure");
		ret = 0;
		goto done;
	}
	if ((infoptr = png_create_info_struct(pngptr)) == NULL){
		perror("Can't create png info structure");
		ret = 0;
		goto done;
	}
	png_init_io(pngptr, fp);
	png_set_compression_level(pngptr, 1);
	png_set_IHDR(pngptr, infoptr, width, height, 8, PNG_COLOR_TYPE_RGB,//16, PNG_COLOR_TYPE_GRAY,
				PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT,
				PNG_FILTER_TYPE_DEFAULT);
	png_write_info(pngptr, infoptr);
	png_set_swap(pngptr);
	for(row = &data[width*(height-1)]; height > 0; row -= width, height--)
		png_write_row(pngptr, (png_bytep)processRow(row, width, min, wd));
	png_write_end(pngptr, infoptr);
done:
	if(fp) fclose(fp);
	if(pngptr) png_destroy_write_struct(&pngptr, &infoptr);
#else
	WARN("Save to PNG doesn't supported");
#endif // PNG
	return ret;
}

int writejpg(char *filename, size_t width, size_t height, BBox *imbox,
				mirPar *mirror, float *data){
	FNAME();
	int ret = 1;
#if defined __JPEG && __JPEG == TRUE
	float min = glob_stat.min, wd = glob_stat.max - min;
	float *row;
	FILE* outfile = fopen(filename, "w");
	if(!outfile){
		perror("Can't open jpg file");
		ret = 0;
		goto done;
	}
	struct jpeg_compress_struct cinfo;
	struct jpeg_error_mgr       jerr;
	cinfo.err = jpeg_std_error(&jerr);
	jpeg_create_compress(&cinfo);
	jpeg_stdio_dest(&cinfo, outfile);
	cinfo.image_width      = width;
	cinfo.image_height     = height;
	cinfo.input_components = 3;
	cinfo.in_color_space   = JCS_RGB;
	jpeg_set_defaults(&cinfo);
	jpeg_set_quality (&cinfo, 99, 1);
	jpeg_start_compress(&cinfo, 1);
	JSAMPROW row_pointer;
	for(row = &data[width*(height-1)]; height > 0; row -= width, height--){
		row_pointer = (JSAMPROW)processRow(row, width, min, wd);
		jpeg_write_scanlines(&cinfo, &row_pointer, 1);
	}
	jpeg_finish_compress(&cinfo);
done:
	if(outfile) fclose(outfile);
#else
	WARN("Save to JPEG doesn't supported");
#endif // JPEG
	return ret;
}

int writetiff(char *filename, size_t width, size_t height, BBox *imbox,
				mirPar *mirror, float *data){
	FNAME();
	int ret = 1;
#if defined __TIFF && __TIFF == TRUE
	float min = glob_stat.min, wd = glob_stat.max - min;
	float *row;
	TIFF *image = TIFFOpen(filename, "w");
	if(!image){
		perror("Can't open tiff file");
		ret = 0;
		goto done;
	}
	TIFFSetField(image, TIFFTAG_IMAGEWIDTH, width);
	TIFFSetField(image, TIFFTAG_IMAGELENGTH, height);
	TIFFSetField(image, TIFFTAG_BITSPERSAMPLE, 8);
	TIFFSetField(image, TIFFTAG_SAMPLESPERPIXEL, 3);
	TIFFSetField(image, TIFFTAG_ROWSPERSTRIP, 1);
	TIFFSetField(image, TIFFTAG_ORIENTATION, ORIENTATION_BOTLEFT);
	TIFFSetField(image, TIFFTAG_COMPRESSION, COMPRESSION_DEFLATE);
	TIFFSetField(image, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_RGB);
	TIFFSetField(image, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
	TIFFSetField(image, TIFFTAG_RESOLUTIONUNIT, RESUNIT_NONE);
	tstrip_t strip = 0;
	for(row = data; strip < height; row += width, strip++){
		TIFFWriteEncodedStrip(image, strip, processRow(row, width, min, wd), width * 3);
		//TIFFWriteScanline
	}
done:
	if(image) TIFFClose(image);
#else
	WARN("Save to TIFF doesn't supported");
#endif // TIFF
	return ret;
}


typedef struct{
	imtype t;
	char *s;
	int (*writefn)(char *, size_t, size_t, BBox *, mirPar *, float *);
}itsuff;

static itsuff suffixes[] = {
	{IT_FITS,	"fits", writefits},
	{IT_PNG,	"png",  writepng},
	{IT_JPEG,	"jpeg", writejpg},
	{IT_TIFF,	"tiff", writetiff},
	{0,			NULL,   NULL}
};
/**
 * Save data to image file[s] with format t
 * @param name - filename prefix or NULL to save to "outXXXX.format"
 * @param t - image[s] type[s]
 * @param width, height - image size
 * @param imbox - image bounding box (for FITS header)
 * @param mirror - mirror parameters (for FITS header)
 * @param data  image data
 * @return number of saved images
 */
int writeimg(char *name, imtype t, size_t width, size_t height, BBox *imbox,
				mirPar *mirror, float *data){
	FNAME();
	char *filename = NULL, *suffix;
	int ret = 0;
	itsuff *suf = suffixes;
	get_stat(data, width*height);
	while(t && suf->t){
		if(!(t & suf->t)){
			suf++;
			continue;
		}
		t ^= suf->t;
		suffix = suf->s;
		if(name)
			filename = createfilename(name, suffix);
		else
			filename = createfilename("out", suffix);
		DBG("Filename: %s", filename);
		if(!filename){
			fprintf(stderr, "Create file with name %s and suffix %s failed,\n", name, suffix);
			continue;
		}
		if(suf->writefn(filename, width, height, imbox, mirror, data)) ret++;
		FREE(filename);
	}
	return ret;
}
