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
#include "usefull_macros.h"


char *outpfile = NULL; // filename for data output in octave text format
int printDebug = 0;    // print tab
bool firstRun = TRUE;  // first run: create new file
int forceCPU = 0;

void signals(int sig){exit (sig);}


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
			//PR(" %g", *data++);
			float d = data[H*(H-y-1) + x];
			PR(" %g", d);
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
	mmapbuf *map;
	char *endptr, *ptr, *dstart;
	if(!filename){
		assert(Size);
		ret = MALLOC(float, (*Size) * (*Size)); // allocate matrix with given size
		assert(ret);
		return ret;
	}
	// there was filename given: try to read data from it
	map = My_mmap(filename);
	ptr = dstart = map->data;
	Mlen = map->len;
	do{
		errno = 0;
		strtof(ptr, &endptr);
		if(errno || (endptr == ptr && *ptr))
			ERR(_("Wrong file: should be matrix of float data separated by spaces"));
		W++;
		if(endptr >= dstart + Mlen) break; // eptr out of range - EOF?
		if(*endptr == '\n'){
			H0++;
			ptr = endptr + 1;
			if(!W0) W0 = W; // update old width counter
			else if(W != W0) // check it
				ERRX(_("All rows must contain equal number of columns"));
			W = 0;
		}else ptr = endptr;
	}while(endptr && endptr < dstart + Mlen);
	if(W > 1) H0++; // increase number of rows if there's no trailing '\n' in last line
	if(W0 != H0)
		ERRX(_("Matrix must be square"));
	*Size = W0;
	ret = MALLOC(float, W0*W0);
	ptr = dstart;
	for(i = 0, H0 = 0; H0 < W0; H0++)
		for(W = 0; W < W0; W++, i++){
			ret[W0*(W0-H0-1) + W] = strtof(ptr, &endptr) * 1e-6;
			if(errno || (endptr == ptr && *ptr))
				ERR(_("File modified in runtime?"));
			if(endptr > dstart + Mlen) goto ex_for;
			ptr = endptr;
		}
ex_for:
	W0 *= W0;
	if(i != W0)
		ERRX(_("Error reading data: read %d numbers instead of %d"), i, W0);
	My_munmap(map);
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
	initial_setup();
	G = parse_args(argc, argv);
	M = G->Mirror;
	if(forceCPU) noCUDA();
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
	if(G->randMask || G->randAmp > 0.){ // add random numbers to mask
		size_t Ss = S0*S0;
		ALLOC(float, tmpf,  Ss);
		if(!fillrandarr(Ss, tmpf, G->randAmp))
			/// "Не могу построить матрицу случайных чисел"
			ERR(_("Can't build random matrix"));
		OMP_FOR()
		for(x = 0; x < Ss; x++) idata[x] += tmpf[x];
		FREE(tmpf);
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
	Diaphragm dia = {{-0.5, -0.5, 1., 1.}, NULL, 0, 20., NULL};
	mirMask *mask;
	if(G->holes_filename) readHoles(G->holes_filename, &dia);
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
		FREE(dm);
	}
	// coordinates of photons
	ALLOC(float, xout, N_phot);
	ALLOC(float, yout, N_phot);
	// resulting image
	ALLOC(float, image, Sim*Sim);

	// CCD bounding box
	BBox CCD = {-5e-4*G->CCDW, -5e-4*G->CCDH, 1e-3*G->CCDW, 1e-3*G->CCDH};

	DBG("obj A=%f, Z=%f",M->objA, M->objZ);
	green("Make %d iterations by %d photons on each", G->N_iter, N_phot);
	printf("\n");
	for(x = 0; x < G->N_iter; ++x){
		if(x%1000 == 999) printf("Iteration %d\n", x+1);
		if(!getPhotonXY(xout, yout, 1, &mD, M, N_phot, &box))
			ERR("Can't build photon map");
		if(!fillImage(xout, yout, N_phot, image, Sim, Sim, &CCD))
			ERR("Can't fill output image");
	}
	FREE(xout); FREE(yout);

	writeimg(G->outfile, imt, Sim, Sim, &CCD, M, image);
	FREE(image);
	return 0;
}
