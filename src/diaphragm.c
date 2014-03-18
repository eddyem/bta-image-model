/*
 * diaphragm.c - read diaphragm parameters from file
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
#include "wrapper.h"
#include "diaphragm.h"
#include <sys/mman.h>
#include <json/json.h>

/**
 * Get double value from object
 * @param jobj (i) - object
 * @return double value
 */
double get_jdouble(json_object *jobj){
	enum json_type type = json_object_get_type(jobj);
	double val;
	switch(type){
		case json_type_double:
			val = json_object_get_double(jobj);
		break;
		case json_type_int:
			val = json_object_get_int(jobj);
		break;
		default:
			ERRX(_("Wrong value! Get non-number!\n"));
	}
	return val;
}

/**
 * Fill double array from json array object
 *
 * @param  jobj  (i) - JSON array
 * @param getLen (o) - array length or NULL
 * @return filled array
 */
double *json_get_array(json_object *jobj, int *getLen){
	enum json_type type;
	json_object *jarray = jobj;
	int arraylen = json_object_array_length(jarray);
	double *arr = calloc(arraylen, sizeof(double));
	int L = 0;
	int i;
	json_object *jvalue;
	for (i=0; i< arraylen; i++){
		jvalue = json_object_array_get_idx(jarray, i);
		type = json_object_get_type(jvalue);
		if(type == json_type_array){ // nested arrays is error
			ERRX(_("Invalid file format! Found nested arrays!\n"));
		}
		else if (type != json_type_object){
			arr[L++] = get_jdouble(jvalue);
		}
		else{ // non-numerical data?
			ERRX(_("Invalid file format! Non-numerical data in array!\n"));
		}
	}
	if(L == 0) FREE(arr);
	if(getLen) *getLen = L;
	return arr;
}

/**
 * Read bounding box from JSON object
 *
 * @param B    (o) - output BBox (allocated outside)
 * @param jobj (i) - JSON object (array with BBox params)
 */
void json_get_bbox(BBox *B, json_object *jobj){
	double *arr = NULL;
	int Len;
	assert(B); assert(jobj);
	json_object *o = json_object_object_get(jobj, "bbox");
	if(!o) return;
	if(!(arr = json_get_array(o, &Len)) || Len != 4){
		ERRX(_("\"bbox\" must contain an array of four doubles!\n"));
	}
	B->x0 = arr[0]; B->y0 = arr[1]; B->w = arr[2]; B->h = arr[3];
	FREE(arr);
}

aHole globHole; // global parameters for all holes (in beginning of diafragm JSON)
/**
 * Fills aHole object by data in JSON
 * @param jobj (i) - JSON data for hole
 * @param H    (o) - output hole object
 */
void get_obj_params(json_object *jobj, aHole *H){
	double *arr = NULL;
	int Len;
	enum json_type type;
	if(!H){
		ERRX( _("Error: NULL instead of aHole structure!\n"));
	}
	memcpy(H, &globHole, sizeof(aHole)); // initialize hole by global values
	json_object *o = json_object_object_get(jobj, "shape");
	if(o){
		const char *ptr = json_object_get_string(o);
		if(strcmp(ptr, "square") == 0) H->type = H_SQUARE;
		else if(strcmp(ptr, "round") == 0 || strcmp(ptr, "ellipse") == 0 )  H->type = H_ELLIPSE;
		else H->type = H_UNDEF;
	}
	o = json_object_object_get(jobj, "radius");
	if(o){
		type = json_object_get_type(o);
		if(type == json_type_int || type == json_type_double){ // circle / square
			double R = json_object_get_double(o);
			H->box.w = H->box.h = R * 2.;
		}else if(type == json_type_array){ // ellipse / rectangle
			if(!(arr = json_get_array(o, &Len)) || Len != 2){
				ERRX(_("\"radius\" array must consist of two doubles!\n"));
			}
			H->box.w = arr[0] * 2.; H->box.h = arr[1] * 2.;
			FREE(arr);
		}else{
			ERRX(_("\"radius\" must be a number or an array of two doubles!\n"));
		}
	}
	o = json_object_object_get(jobj, "center");
	if(o){
		if(!(arr = json_get_array(o, &Len)) || Len != 2){
			ERRX(_("\"center\" must contain an array of two doubles!\n"));
		}
		H->box.x0 = arr[0] - H->box.w/2.;
		H->box.y0 = arr[1] - H->box.h/2.;
		FREE(arr);
	}else{
		json_get_bbox(&H->box, jobj);
	/*	o = json_object_object_get(jobj, "bbox");
		if(o){
			if(!(arr = json_get_array(o, &Len)) || Len != 4){
				ERRX(_("\"bbox\" must contain an array of four doubles!\n"));
			}
			H->box.x0 = arr[0]; H->box.y0 = arr[1]; H->box.w = arr[2]; H->box.h = arr[3];
			FREE(arr);
		}*/
	}
}

/**
 * Fill array of holes from JSON data
 *
 * @param jobj   (i) - JSON array with holes data
 * @param getLen (o) - length of array or NULL
 * @return holes array
 */
aHole *json_parse_holesarray(json_object *jobj, int *getLen){
	enum json_type type;
	json_object *jarray = jobj;
	int arraylen = json_object_array_length(jarray), i;
	aHole *H = calloc(arraylen, sizeof(aHole));
	json_object *jvalue;
	for (i=0; i < arraylen; i++){
		jvalue = json_object_array_get_idx(jarray, i);
		type = json_object_get_type(jvalue);
		if(type == json_type_object){
			get_obj_params(jvalue, &H[i]);
		}else{
			ERRX(_("Invalid holes array format!\n"));
		}
	}
	if(getLen) *getLen = arraylen;
	return H;
}

#ifdef EBUG
char *gettype(aHole *H){
	char *ret;
	switch(H->type){
		case H_SQUARE:
			ret = "square";
		break;
		case H_ELLIPSE:
			ret = "ellipse";
		break;
		default:
			ret = "undefined";
	}
	return ret;
}
#endif

/**
 * Try to mmap a file
 *
 * @param filename (i) - name of file to mmap
 * @return pointer with mmap'ed file or die
 */
char *My_mmap(char *filename, size_t *Mlen){
	int fd;
	char *ptr;
	struct stat statbuf;
	if(!filename) ERRX(_("No filename given!"));
	if((fd = open(filename, O_RDONLY)) < 0)
		ERR(_("Can't open %s for reading"), filename);
	if(fstat (fd, &statbuf) < 0)
		ERR(_("Can't stat %s"), filename);
	*Mlen = statbuf.st_size;
	if((ptr = mmap (0, *Mlen, PROT_READ, MAP_PRIVATE, fd, 0)) == MAP_FAILED)
		ERR(_("Mmap error for input"));
	if(close(fd)) ERR(_("Can't close mmap'ed file"));
	return  ptr;
}

/**
 * Read holes array for diaphragm structure from file
 *
 * file should
 *
 * @param filename - name of file
 * @return readed structure or NULL if failed
 */
aHole *readHoles(char *filename, Diaphragm *dia){
	char *ptr;
	enum json_type type;
	json_object *o, *jobj;
	int i, HolesNum;
	aHole *HolesArray;
	BBox Dbox = {100.,100.,-200.,-200.}; // bounding box of diaphragm
	size_t Mlen;
	if(dia) memset(dia, 0, sizeof(Diaphragm));
	ptr = My_mmap(filename, &Mlen);
	jobj = json_tokener_parse(ptr);
	get_obj_params(jobj, &globHole); // read global parameters
	// now try to find diaphragm bounding box & Z-coordinate
	json_get_bbox(&Dbox, jobj);
	// check for Z-coordinate
	if(dia){
		o = json_object_object_get(jobj, "Z");
		if(!o) o = json_object_object_get(jobj, "maskz");
		if(!o)
			ERRX(_("JSON file MUST contain floating point field \"Z\" or \"maskz\" with mask's coordinate"));
		dia->Z = get_jdouble(o); // read mask Z
	}
	o = json_object_object_get(jobj, "holes");
	if(!o)
		ERRX(_("Corrupted file: no holes found!"));
	type = json_object_get_type(o);
	if(type == json_type_object){ // single hole
		HolesArray = calloc(1, sizeof(aHole));
		assert(HolesArray);
		HolesNum = 1;
		get_obj_params(o, HolesArray);
	}else{ // array of holes
		HolesArray = json_parse_holesarray(o, &HolesNum);
	}
	if(!HolesArray || HolesNum < 1)
		ERRX(_("Didn't find any holes in json file!"));
	DBG("Readed %d holes", HolesNum);
	// check bbox of diafragm (or make it if none)
	float minx=100., miny=100., maxx=-100., maxy=-100.;
	for(i = 0; i < HolesNum; i++){
		BBox *B = &HolesArray[i].box;
		float L=B->x0, R=L+B->w, D=B->y0, U=D+B->h;
		if(minx > L) minx = L;
		if(maxx < R) maxx = R;
		if(miny > D) miny = D;
		if(maxy < U) maxy = U;
#ifdef EBUG
		green("Hole %4d:", i);
		printf(" type =%9s, bbox = {%7.4f, %7.4f, %7.4f, %7.4f }\n", gettype(&HolesArray[i]),
				B->x0, B->y0, B->w, B->h);
#endif
	}
	float wdth = maxx - minx + 0.1, hght = maxy - miny + 0.1; // width & height of bbox
	// now correct bbox (or fill it if it wasn't in JSON)
	if(Dbox.x0 > minx) Dbox.x0 = minx;
	if(Dbox.y0 > miny) Dbox.y0 = miny;
	if(Dbox.w < wdth) Dbox.w = wdth;
	if(Dbox.h < hght) Dbox.h = hght;
	munmap(ptr, Mlen);
	if(dia){
		dia->holes = HolesArray;
		dia->Nholes = HolesNum;
		memcpy(&dia->box, &Dbox, sizeof(BBox));
	}
	return HolesArray;
}

