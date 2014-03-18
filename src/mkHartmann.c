/*
 * mkHartmann.c - main file for mkHartmann utilite produsing hartmanogramms
 * 					of BTA telescope
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
#include "cmdlnopts.h"
#include "wrapper.h"
#include "saveimg.h"
#include "diaphragm.h"

/*
 * Coloured messages output
 */
#define RED			"\033[1;31;40m"
#define GREEN		"\033[1;32;40m"
#define OLDCOLOR	"\033[0;0;0m"


int globErr = 0; // errno for WARN/ERR
// pointers to coloured output printf
int (*red)(const char *fmt, ...);
int (*green)(const char *fmt, ...);
int (*_WARN)(const char *fmt, ...);

/*
 * format red / green messages
 * name: r_pr_, g_pr_
 * @param fmt ... - printf-like format
 * @return number of printed symbols
 */
int r_pr_(const char *fmt, ...){
	va_list ar; int i;
	printf(RED);
	va_start(ar, fmt);
	i = vprintf(fmt, ar);
	va_end(ar);
	printf(OLDCOLOR);
	return i;
}
int g_pr_(const char *fmt, ...){
	va_list ar; int i;
	printf(GREEN);
	va_start(ar, fmt);
	i = vprintf(fmt, ar);
	va_end(ar);
	printf(OLDCOLOR);
	return i;
}
/*
 * print red error/warning messages (if output is a tty)
 * @param fmt ... - printf-like format
 * @return number of printed symbols
 */
int r_WARN(const char *fmt, ...){
	va_list ar; int i = 1;
	fprintf(stderr, RED);
	va_start(ar, fmt);
	if(globErr){
		errno = globErr;
		vwarn(fmt, ar);
		errno = 0;
		globErr = 0;
	}else
		i = vfprintf(stderr, fmt, ar);
	va_end(ar);
	i++;
	fprintf(stderr, OLDCOLOR "\n");
	return i;
}

const char stars[] = "****************************************";
/*
 * notty variants of coloured printf
 * name: s_WARN, r_pr_notty
 * @param fmt ... - printf-like format
 * @return number of printed symbols
 */
int s_WARN(const char *fmt, ...){
	va_list ar; int i;
	i = fprintf(stderr, "\n%s\n", stars);
	va_start(ar, fmt);
	if(globErr){
		errno = globErr;
		vwarn(fmt, ar);
		errno = 0;
		globErr = 0;
	}else
		i = +vfprintf(stderr, fmt, ar);
	va_end(ar);
	i += fprintf(stderr, "\n%s\n", stars);
	i += fprintf(stderr, "\n");
	return i;
}
int r_pr_notty(const char *fmt, ...){
	va_list ar; int i;
	i = printf("\n%s\n", stars);
	va_start(ar, fmt);
	i += vprintf(fmt, ar);
	va_end(ar);
	i += printf("\n%s\n", stars);
	return i;
}

/*
 * safe memory allocation for macro ALLOC
 * @param N - number of elements to allocate
 * @param S - size of single element (typically sizeof)
 * @return pointer to allocated memory area
 */
void *my_alloc(size_t N, size_t S){
	void *p = calloc(N, S);
	if(!p) ERR("malloc");
	//assert(p);
	return p;
}

char *outpfile = NULL; // filename for data output in octave text format
int printDebug = 0;    // print tab
bool firstRun = TRUE;  // first run: create new file
/**
 * Print tabular on screen (if outpfile == NULL) or to outpfile
 * 		in octave text format
 * 	Function run only if printDebug == TRUE or outpfile != NULL
 *
 * @param W, H (i)  - matrix width & height
 * @param data      - data to print
 * @param mask      - format to print (if on screen)
 * @param comment   -
 * @return
 */
void printTAB(size_t W, size_t H, float *data, char *mask, char *comment){
	size_t x,y;
	if(!printDebug && !outpfile) return; // don't print debug info if no flag debug &/| outpfile
	if(!outpfile){ // simply print to stdout
		if(comment) printf("%s\n", comment);
		if(mask) printf("All units *1e-6\n");
		for(y = 0; y < H; y++){
			for(x = 0; x < W; x++){
				float d = data[H*(H-y-1) + x];
				mask ? printf(mask, d*1e6) : printf("%6g ", d);
			}
			printf("\n");
		}
		return;
	}
	// print to file
	struct stat statbuf;
	FILE *f = NULL;
	#define PR(...) do{if(fprintf(f, __VA_ARGS__) < 0) ERR(_("Can't write to %s"), outpfile);}while(0)
	if(firstRun){
		if(stat(outpfile, &statbuf)){
			if(ENOENT != errno) // file not exist but some error occured
				ERR(_("Can't stat %s"), outpfile);
			// OK, file not exist: use its name
		}else{ // file exists, create new name
			outpfile = createfilename(outpfile, NULL); // create new file name
		}
		firstRun = FALSE;
		if(!outpfile) ERRX(_("No output filename given"));
		f = fopen(outpfile, "w"); // create or truncate
		time_t T = time(NULL);
		PR("# Created by %s, %s\n", __progname, ctime(&T)); // add header
	}else{ // simply open file for adding info
		f = fopen(outpfile, "a");
	}
	if(!f) ERR(_("Can't open file %s"), outpfile);
	// print standard octave header for matrix
	if(comment) PR("# name: %s\n", comment);
	else PR("# name: tmp_%ju\n", (uintmax_t)time(NULL)); // or create temporary name
	PR("# type: matrix\n# rows: %zd\n# columns: %zd\n", H, W); // dimentions
	// now put out matrix itself (upside down - for octave/matlab)
	for(y = 0; y < H; y++){
		for(x = 0; x < W; x++){
			PR(" %g", *data++);
		}
		PR("\n");
	}
	PR("\n\n");
	#undef PR
	if(fclose(f)) ERR(_("Can't close file %s"), outpfile);
}

/**
 * Read array with deviations from file
 * 		if filename is NULL it will just generate zeros (Size x Size)
 *		ALL DEVIATIONS in file are IN MICROMETERS!!!
 *
 * @param filename (i)  - name of file or NULL for zeros
 * @param Size     (io) - size of output square array
 * @return allocated array
 */
float *read_deviations(char *filename, size_t *Size){
	float *ret = NULL;
	int W = 0, W0 = 0, H0 = 0, i;
	size_t Mlen;
	char *Mem = NULL, *endptr, *ptr;
	if(!filename){
		assert(Size);
		ret = MALLOC(float, (*Size) * (*Size)); // allocate matrix with given size
		assert(ret);
		return ret;
	}
	// there was filename given: try to read data from it
	Mem = My_mmap(filename, &Mlen); // from diaphragm.c
	ptr = Mem;
	do{
		errno = 0;
		strtof(ptr, &endptr);
		if(errno || (endptr == ptr && *ptr))
			ERRX(_("Wrong file: should be matrix of float data separated by spaces"));
		W++;
		if(endptr >= Mem + Mlen) break; // eptr out of range - EOF?
		if(*endptr == '\n'){
			H0++;
			ptr = endptr + 1;
			if(!W0) W0 = W; // update old width counter
			else if(W != W0) // check it
				ERRX(_("All rows must contain equal number of columns"));
			W = 0;
		}else ptr = endptr;
	}while(endptr && endptr < Mem + Mlen);
	if(W > 1) H0++; // increase number of rows if there's no trailing '\n' in last line
	if(W0 != H0)
		ERRX(_("Matrix must be square"));
	*Size = W0;
	DBG("here");
	ret = MALLOC(float, W0*W0);
	DBG("there");
	ptr = Mem;
	for(i = 0, H0 = 0; H0 < W0; H0++)
		for(W = 0; W < W0; W++, i++){
			DBG("%d ", i);
			ret[W0*(W0-H0-1) + W] = strtof(ptr, &endptr) * 1e-6;
			if(errno || (endptr == ptr && *ptr) || endptr >= Mem + Mlen)
				ERRX(_("Input file was modified in runtime!"));
			ptr = endptr;
		}
	W0 *= W0;
	if(i != W0)
		ERRX(_("Error reading data: read %d numbers instaed of %d"), W-1, W0);
	munmap(Mem, Mlen);
	return ret;
}

int save_images = 0;
/*
 * N ph per pix of mask			TIME, s
 * // 29400 non-zero pts in mask
 * 10000						50
 * 50000						73
 * 100000						102
 * 200000						160
 * 500000						303
 * 1000000						556
 * 2000000						1128
 * 5000000						2541
 * 10000000						4826
 */
int main(int argc, char **argv){
	glob_pars *G = NULL; // default parameters see in cmdlnopts.c
	mirPar *M = NULL;    // default mirror parameters
	int x, y _U_;
	// setup coloured output
	if(isatty(STDOUT_FILENO)){ // make color output in tty
		red = r_pr_; green = g_pr_;
	}else{ // no colors in case of pipe
		red = r_pr_notty; green = printf;
	}
	if(isatty(STDERR_FILENO)) _WARN = r_WARN;
	else _WARN = s_WARN;
	// Setup locale
	setlocale(LC_ALL, "");
	setlocale(LC_NUMERIC, "C");
	bindtextdomain(GETTEXT_PACKAGE, LOCALEDIR);
	textdomain(GETTEXT_PACKAGE);
	G = parce_args(argc, argv);
	M = G->Mirror;
	// Run simple initialisation of CUDA and/or memory test
	getprops();
	size_t S0 = G->S_dev, S1 = G->S_interp, Sim = G->S_image, N_phot = G->N_phot;
	size_t masksize = S1 * S1;
	// type of image to save: fits, png, jpeg or tiff
	imtype imt = G->it;
	// bounding box of mirror
	BBox box;
	box.x0 = box.y0 = -M->D/2; box.w = box.h = M->D;
	float *idata = read_deviations(G->dev_filename, &S0);
	printTAB(S0, S0, idata, "%5.2f ", "input_deviations");
	G->S_dev = S0; // update size
	// memory allocation
	ALLOC(float, mirZ,  masksize); // mirror Z coordinate
	ALLOC(float, mirDX, masksize); // projections of normale to mirror
	ALLOC(float, mirDY, masksize);
	if(G->randMask || G->randAmp != Gdefault.randAmp){ // add random numbers to mask
		if(!fillrandarr(S0*S0, idata, G->randAmp))
			/// "Не могу построить матрицу случайных чисел"
			ERR(_("Can't build random matrix"));
	}
	// initialize structure of mirror deviations
	mirDeviations mD;
	mD.mirZ = mirZ; mD.mirDX = mirDX; mD.mirDY = mirDY;
	mD.mirWH = S1;
	if(!mkmirDeviat(idata, S0, M->D, &mD))
		ERRX(_("Can't build mirror dZ arrays"));
	if(save_images) writeimg("interp_deviations", imt, S1, S1, &box, M, mirZ);
	printTAB(S1, S1, mirZ, "%5.3f ", "interpolated_deviations");
	printTAB(S1, S1, mirDX, "%5.2f ", "dev_dZdX");
	printTAB(S1, S1, mirDY, "%5.2f ", "dev_dZdY");

	/*aHole holes[3] = {
		{{-0.35, -0.35, 0.1, 0.1}, H_ELLIPSE},
		{{-0.2, 0.2, 0.7, 0.4}, H_ELLIPSE},
		{{-0.1,-0.45,0.6,0.6}, H_ELLIPSE}}; */

	// Hartmann mask
	Diaphragm dia; //{{-0.5, -0.5, 1., 1.}, NULL, 0, 20., NULL};
	mirMask *mask;
	readHoles("holes.json", &dia);
	#ifdef EBUG
	green("Dia: ");
	printf("(%g, %g, %g, %g); %d holes, Z = %g\n", dia.box.x0, dia.box.y0,
		dia.box.w, dia.box.h, dia.Nholes, dia.Z);
	#endif
	if(!(mask = makeDmask(&dia, 128, M, &mD))) ERR("Can't make dmask!");
	M->dia = &dia;
	if(save_images){
		int S = (int)dia.mask->WH, S2=S*S;
		float *dm = MALLOC(float, S2);
		for(x=0; x<S2; x++) dm[x] = (float)dia.mask->data[x];
		writeimg("mirror_mask", imt, S, S, NULL, M, dm);
		free(dm);
	}
	// coordinates of photons
	ALLOC(float, xout, N_phot);
	ALLOC(float, yout, N_phot);
	// resulting image
	ALLOC(float, image, Sim*Sim);
/*
//for(int i = 0; i < 100; i++){
	box.x0 = -3.; box.y0 = -3.; box.w = 6.; box.h = 6.;
	if(!getPhotonXY(xout, yout, 1, &mD, M, N_phot, &box))
		ERR("Can't build photon map");
	box.x0 = -15e-3; box.y0 = -15e-3; box.w = 30e-3; box.h = 30e-3;
	//box.x0 = -5e-3; box.y0 = .8365; box.w = 10e-3; box.h = 10e-3;
	if(!fillImage(xout, yout, N_phot, image, Sim, Sim, &box))
		ERR("Can't fill output image");
//}
	writeimg("image", imt, Sim, Sim, &box, M, image);
	FREE(xout); FREE(yout); FREE(image);
*/
	// CCD bounding box
	BBox CCD = {-15e-3, -15e-3, 30e-3, 30e-3};
	for(x = 0; x < 100; x++){
		if(!getPhotonXY(xout, yout, 1, &mD, M, N_phot, &box))
			ERR("Can't build photon map");
		if(!fillImage(xout, yout, N_phot, image, Sim, Sim, &CCD))
			ERR("Can't fill output image");
	}
/*	int S = mask->WH;
	double R = M->D / 2., scale = M->D / (double)S;
	uint16_t *dptr = mask->data;
	box.w = box.h = scale;
	// check mask's pixels & throw photons to holes
	for(y = 0; y < S; y++){
		for(x = 0; x < S; x++, dptr++){
			if(!*dptr) continue;
			DBG("x = %d, Y=%d\n", x,y);
			box.x0 = -R + scale*(double)x;
			box.y0 = -R + scale*(double)y;
			if(!getPhotonXY(xout, yout, 1, &mD, M, N_phot, &box))
				ERR("Can't build photon map");
			if(!fillImage(xout, yout, N_phot, image, Sim, Sim, &CCD))
				ERR("Can't fill output image");
		}
	}
*/
	writeimg("image", imt, Sim, Sim, &CCD, M, image);
	FREE(xout); FREE(yout); FREE(image);
	// if rand() is good, amount of photons on image should be 785398 on every 1000000
	//printTAB(Sim, Sim, image, NULL, "\n\nResulting image:");
/*	for(x = 0; x < N_phot; x++)
		if(fabs(xout[x]) < M->D/2. && fabs(yout[x]) < M->D/2.)
			printf("photon #%4d:\t\t(%g, %g)\n", x, xout[x]*1e6, yout[x]*1e6);*/

/*	FILE *F = fopen("TESTs", "w");
	if(!F) ERR("Can't open");
	fprintf(F,"S1\tGPU\t\tCPU\n");


	for(S1 = 100; ; S1 += S1*drand48()){
	float *odata = my_alloc(S1*S1, sizeof(float));
	double t0;
	fprintf(F,"%zd", S1);
	forceCUDA();
	*/
/*	int x, y; float *ptr = idata;
	printf("Original array:\n");
	for(y = 0; y < 5; y++){
		for(x = 0; x < 5; x++){
			*ptr *= 2.;
			*ptr += x;
			printf("%4.3f ", (*ptr++));
		}
		printf("\n");
	}
	t0 = dtime();
	if(!bicubic_interp(odata, idata, S1,S1, S0,S0)) fprintf(F,"\tnan");
	else fprintf(F,"\t%g", dtime() - t0);

	printf("Enlarged array:\n");
	ptr = odata;
	for(y = 0; y < 20; y++){
		for(x = 0; x < 20; x++)
			printf("%4.3f ", (*ptr++));
		printf("\n");
	}
	*
	noCUDA();
	t0 = dtime();
	/// "Не могу построить интерполяцию"
	if(!bicubic_interp(odata, idata, S1,S1, S0,S0)) ERR(_("Can't do interpolation"));
	fprintf(F,"\t%g\n", dtime() - t0);
	fflush(F);
	free(odata);
	}*/
	return 0;
}
