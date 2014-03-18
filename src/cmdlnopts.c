/*
 * cmdlnopts.c - the only function that parce cmdln args and returns glob parameters
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
/*
#include <stdio.h>
#include <stdlib.h>
#include <string.h>*/
#include "cmdlnopts.h"


// global variables for parsing
glob_pars  G;
mirPar     M;
int help;
// don't rewrite existing files: for saveimg.c
extern int rewrite_ifexists;
/*
 * variables for mkHartmann.c
 */
extern char *outpfile; // output file name for saving matrixes
extern int printDebug; // print tabulars with data on screen or to file
extern int save_images;// save intermediate results to images

// suboptions structure for get_suboptions
// in array last MUST BE {0,0,0}
typedef struct{
	float *val;		// pointer to result
	char *par;		// parameter name (CASE-INSENSITIVE!)
	bool isdegr;	// == TRUE if parameter is an angle in format "[+-][DDd][MMm][SS.S]"
} suboptions;

//            DEFAULTS
// default global parameters
glob_pars const Gdefault = {
	8,		// size of initial array of surface deviations
	100,	// size of interpolated S0
	1000,	// resulting image size
	10000,	// amount of photons falled to one pixel of S1 by one iteration
	0,		// add to mask random numbers
	1e-8,	// amplitude of added random noice
	IT_FITS,// output image type
	NULL,	// input deviations file name
	NULL	// mirror
};
//default mirror parameters
mirPar const Mdefault = {
	6.,		// diameter
	24.024,	// focus
	0.,		// inclination from Z axe (radians)
	0.,		// azimuth of inclination (radians)
	0.,		// azimuth of object (radians)
	0.,		// zenith of object (radians)
	23.9,	// Z-coordinate of light receiver
	NULL	// diaphragm
};

bool get_mir_par(void *arg, int N);
bool get_imtype(void *arg, int N);

char MirPar[] = N_("set mirror parameters, arg=[diam=num:foc=num:Zincl=ang:Aincl=ang:Ao=ang:Zo=ang:C=num]\n" \
		"\t\t\tALL DEGREES ARE IN FORMAT [+-][DDd][MMm][SS.S] like -10m13.4 !\n" \
		"\t\tdiam  - diameter of mirror\n" \
		"\t\tfoc   - mirror focus ratio\n" \
		"\t\tZincl - inclination from Z axe\n" \
		"\t\tAincl - azimuth of inclination\n" \
		"\t\tAobj  - azimuth of object\n" \
		"\t\tZobj  - zenith of object\n" \
		"\t\tccd   - Z-coordinate of light receiver");

//	name	has_arg	flag	val		type		argptr			help
myoption cmdlnopts[] = {
	{"help",	0,	NULL,	'h',	arg_int,	APTR(&help),		N_("show this help")},
	{"dev-size",1,	NULL,	'd',	arg_int,	APTR(&G.S_dev),		N_("size of initial array of surface deviations")},
	{"int-size",1,	NULL,	'i',	arg_int,	APTR(&G.S_interp),	N_("size of interpolated array of surface deviations")},
	{"image-size",1,NULL,	'I',	arg_int,	APTR(&G.S_image),	N_("resulting image size")},
	{"N-photons",1,NULL,	'N',	arg_int,	APTR(&G.N_phot), 	N_("amount of photons falled to one pixel of matrix by one iteration")},
	{"mir-parameters",1,NULL,'M',	arg_function,APTR(&get_mir_par),MirPar},
	{"add-noice",0,	&G.randMask,1,	arg_none,	NULL,				N_("add random noice to mirror surface deviations")},
	{"noice-amp",1,	NULL,	'a',	arg_float,	APTR(&G.randAmp),	N_("amplitude of random noice (default: 1e-8)")},
	{"dev-file", 1,	NULL,	'F',	arg_string,	APTR(&G.dev_filename),N_("filename for mirror surface deviations (in microns!)")},
	{"force",	0,	&rewrite_ifexists,1,arg_none,NULL,				N_("rewrite output file if exists")},
	{"log-file",1,	NULL,	'l',	arg_string,	APTR(&outpfile),	N_("save matrices to file arg")},
	{"print-matr",0,&printDebug,1,	arg_none,	NULL,				N_("print matrices on screen")},
	{"save-images",0,&save_images,1,arg_none,	NULL,				N_("save intermediate results to images")},
	{"image-type",1,NULL,	'T',	arg_function,APTR(&get_imtype),	N_("image type, arg=[jfpt] (Jpeg, Fits, Png, Tiff)")},
	end_option
};

/**
 * Safely convert data from string to float
 *
 * @param num (o) - float number read from string
 * @param str (i) - input string
 * @return TRUE if success
 */
bool myatof(float *num, const char *str){
	float res;
	char *endptr;
	assert(str);
	res = strtof(str, &endptr);
	if(endptr == str || *str == '\0' || *endptr != '\0'){
		WARNX(_("Wrong float number format!"));
		return FALSE;
	}
	*num = res;
	return TRUE;
}

/**
 * Convert string "[+-][DDd][MMm][SS.S]" into radians
 *
 * @param ang (o) - angle in radians or exit with help message
 * @param str (i) - string with angle
 * @return TRUE if OK
 */
bool get_radians(float *ret, char *str){
	float val = 0., ftmp, sign = 1.;
	char *ptr;
	assert(str);
	switch(*str){ // check sign
		case '-':
			sign = -1.;
		case '+':
			str++;
	}
	if((ptr = strchr(str, 'd'))){ // found DDD.DDd
		*ptr = 0; if(!myatof(&ftmp, str)) return FALSE;
		ftmp = fabs(ftmp);
		if(ftmp > 360.){
			WARNX(_("Degrees should be less than 360"));
			return FALSE;
		}
		val += ftmp;
		str = ptr + 1;
	}
	if((ptr = strchr(str, 'm'))){ // found DDD.DDm
		*ptr = 0; if(!myatof(&ftmp, str)) return FALSE;
		ftmp = fabs(ftmp);
		/*if(ftmp >= 60.){
			WARNX(_("Minutes should be less than 60"));
			return FALSE;
		}*/
		val += ftmp / 60.;
		str = ptr + 1;
	}
	if(strlen(str)){ // there is something more
		if(!myatof(&ftmp, str)) return FALSE;
		ftmp = fabs(ftmp);
		/*if(ftmp >= 60.){
			WARNX(_("Seconds should be less than 60"));
			return FALSE;
		}*/
		val += ftmp / 3600.;
	}
	DBG("Angle: %g degr", val*sign);
	*ret = D2R(val * sign); // convert degrees to radians
	return TRUE;
}

/**
 * Parse string of suboptions (--option=name1=var1:name2=var2... or -O name1=var1,name2=var2...)
 * Suboptions could be divided by colon or comma
 *
 * !!NAMES OF SUBOPTIONS ARE CASE-UNSENSITIVE!!!
 *
 * @param arg (i)    - string with description
 * @param V (io) - pointer to suboptions array (result will be stored in sopts->val)
 * @return TRUE if success
 */
bool get_suboptions(void *arg, suboptions *V){
	char *tok, *val, *par;
	int i;
	tok = strtok(arg, ":,");
	do{
		if((val = strchr(tok, '=')) == NULL){ // wrong format
			WARNX(_("Wrong format: no value for keyword"));
			return FALSE;
		}
		*val++ = '\0';
		par = tok;
		for(i = 0; V[i].val; i++){
			if(strcasecmp(par, V[i].par) == 0){ // found parameter
				if(V[i].isdegr){ // DMS
					if(!get_radians(V[i].val, val)) // wrong angle
						return FALSE;
					DBG("Angle: %g rad\n", *(V[i].val));
				}else{ // simple float
					if(!myatof(V[i].val, val)) // wrong number
						return FALSE;
					DBG("Float val: %g\n", *(V[i].val));
				}
				break;
			}
		}
		if(!V[i].val){ // nothing found - wrong format
			WARNX(_("Bad keyword!"));
			return FALSE;
		}
	}while((tok = strtok(NULL, ":,")));
	return TRUE;
}

/**
 * Parse string of mirror parameters (--mir-diam=...)
 *
 * @param arg (i) - string with description
 * @param N (i)   - number of selected option (unused)
 * @return TRUE if success
 */
bool get_mir_par(void *arg, int N _U_){
	suboptions V[] = { // array of mirror parameters and string keys for cmdln pars
		{&M.D,		"diam",		FALSE},
		{&M.F,		"foc",		FALSE},
		{&M.Zincl,	"zincl",	TRUE},
		{&M.Aincl,	"aincl",	TRUE},
		{&M.objA,	"aobj",		TRUE},
		{&M.objZ,	"zobj",		TRUE},
		{&M.foc,	"ccd",		FALSE},
		{0,0,0}
	};
	return get_suboptions(arg, V);
}

/**
 * Parce command line options and return dynamically allocated structure
 * 		to global parameters
 * @param argc - copy of argc from main
 * @param argv - copy of argv from main
 * @return allocated structure with global parameters
 */
glob_pars *parce_args(int argc, char **argv){
	int i;
	void *ptr;
	ptr = memcpy(&G, &Gdefault, sizeof(G)); assert(ptr);
	ptr = memcpy(&M, &Mdefault, sizeof(M)); assert(ptr);
	G.Mirror = &M;
	// format of help: "Usage: progname [args]\n"
	change_helpstring("Usage: %s [args]\n\n\tWhere args are:\n");
	// parse arguments
	parceargs(&argc, &argv, cmdlnopts);
	if(help) showhelp(-1, cmdlnopts);
	if(argc > 0){
		printf("\nIgnore argument[s]:\n");
		for (i = 0; i < argc; i++)
			printf("\t%s\n", argv[i]);
	}
	return &G;
}

/**
 * Get image type from command line parameter:
 * 		F/f for FITS
 * 		P/p for PNG
 * 		J/j for JPEG
 * 		T/t for TIFF
 * @param arg (i) - string with parameters
 * @return FALSE if fail
 */
bool get_imtype(void *arg, int N _U_){
	assert(arg);
	G.it = 0;
	do{
		switch(*((char*)arg)){
			case 'F': case 'f':
				G.it |= IT_FITS;
			break;
			case 'P': case 'p':
				G.it |= IT_PNG;
			break;
			case 'J': case 'j':
				G.it |= IT_JPEG;
			break;
			case 'T' : case 't':
				G.it |= IT_TIFF;
			break;
			default:
				WARNX(_("Wrong format of image type: %c"), *((char*)arg));
				return FALSE;
		}
	}while(*((char*)++arg));
	return TRUE;
}
