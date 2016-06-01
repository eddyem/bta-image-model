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
#include "usefull_macros.h"
#include "wrapper.h"
#include "diaphragm.h"
#include <sys/mman.h>
#include "json.h"

/**
 * Get double value from object
 * @param jobj (i) - object
 * @return double value
 */
double get_jdouble(json_pair *val){
	if(val->type != json_type_number) ERRX(_("Wrong value! Get non-number!\n"));
	return json_pair_get_number(val);
}

/**
 * Fill double array from json array object
 *
 * @param  jobj  (i) - JSON array
 * @param getLen (o) - array length or NULL
 * @return filled array
 */
double *json_get_array(json_pair *pair, int *getLen){
	if(pair->type != json_type_data_array) return NULL;
	size_t i, arraylen = pair->len;
	if(arraylen < 1) return NULL;
	double *arr = MALLOC(double, arraylen);
	int L = 0;
	char *jvalue;
	for (i = 0; i< arraylen; i++){
		jvalue = json_array_get_data(pair, i);
		//DBG("get arr val[%zd]: %s",i, jvalue);
		if(!jvalue) break;
		arr[L++] = strtod(jvalue, NULL);
		FREE(jvalue);
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
 * @return 0 if all OK
 */
int json_get_bbox(BBox *B, json_object *jobj){
	double *arr = NULL;
	int Len;
	assert(B); assert(jobj);
	json_pair *o = json_object_get_pair(jobj, "bbox");
	if(!o) return 1;
	if(!(arr = json_get_array(o, &Len)) || Len != 4){
		ERRX(_("\"bbox\" must contain an array of four doubles!\n"));
	}
	B->x0 = arr[0]; B->y0 = arr[1]; B->w = arr[2]; B->h = arr[3];
	FREE(arr);
	return 0;
}

aHole globHole; // global parameters for all holes (in beginning of diafragm JSON)
/**
 * Fills aHole object by data in JSON
 * @param jobj (i) - JSON data for hole
 * @param H    (o) - output hole object
 * @return 0 if all OK
 */
int get_obj_params(json_object *jobj, aHole *H){
	double *arr = NULL;
	if(!jobj) return 1;
	int Len;
	if(!H){
		ERRX( _("Error: NULL instead of aHole structure!\n"));
	}
	memcpy(H, &globHole, sizeof(aHole)); // initialize hole by global values
	json_pair *o = json_object_get_pair(jobj, "shape");
	if(o){
		char *ptr = json_pair_get_string(o);
		if(!ptr) ERRX(_("Wrong \"shape\" value"));
		if(strcmp(ptr, "square") == 0) H->type = H_SQUARE;
		else if(strcmp(ptr, "round") == 0 || strcmp(ptr, "ellipse") == 0 )  H->type = H_ELLIPSE;
		else H->type = H_UNDEF;
		//DBG("shape: %s", ptr);
	}
	o = json_object_get_pair(jobj, "radius");
	if(o){
		//DBG("radius: %s", o->value);
		if(o->type == json_type_number){ // circle / square
			double R = strtod(o->value, NULL);
			H->box.w = H->box.h = R * 2.;
		}else if(o->type == json_type_data_array){ // ellipse / rectangle
			if(!(arr = json_get_array(o, &Len)) || Len != 2){
				ERRX(_("\"radius\" array must consist of two doubles!\n"));
			}
			H->box.w = arr[0] * 2.; H->box.h = arr[1] * 2.;
			FREE(arr);
		}else{
			ERRX(_("\"radius\" must be a number or an array of two doubles!\n"));
		}
	}
	o = json_object_get_pair(jobj, "center");
	if(o){
		//DBG("center");
		if(!(arr = json_get_array(o, &Len)) || Len != 2){
			ERRX(_("\"center\" must contain an array of two doubles!\n"));
		}
		H->box.x0 = arr[0] - H->box.w/2.;
		H->box.y0 = arr[1] - H->box.h/2.;
		FREE(arr);
	}else{
		return json_get_bbox(&H->box, jobj);
	}
	return 0;
}

/**
 * Fill array of holes from JSON data
 *
 * @param jobj   (i) - JSON array with holes data
 * @param getLen (o) - length of array or NULL
 * @return holes array
 */
aHole *json_parse_holesarray(json_pair *jobj, int *getLen){
	int arraylen = jobj->len, i;
	if(!arraylen || jobj->type != json_type_obj_array) return NULL;
	aHole *H = calloc(arraylen, sizeof(aHole));
	json_object *jvalue;
	for (i = 0; i < arraylen; i++){
		jvalue = json_array_get_object(jobj, i);
		if(!jvalue) break;
		if(get_obj_params(jvalue, &H[i]))
			ERRX(_("Invalid format for hole #%d!\n"), i);
		json_free_obj(&jvalue);
	}
	if(getLen) *getLen = i;
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
 * Read holes array for diaphragm structure from file
 *
 * file should
 *
 * @param filename - name of file
 * @return read structure or NULL if failed
 */
aHole *readHoles(char *filename, Diaphragm *dia){
	FNAME();
	mmapbuf *map;
	json_pair *o;
	json_object *jobj;
	int i, HolesNum = -1;
	aHole *HolesArray = NULL;
	BBox Dbox = {100.,100.,-200.,-200.}; // bounding box of diaphragm
	if(dia) memset(dia, 0, sizeof(Diaphragm));
	map = My_mmap(filename);
	jobj = json_tokener_parse(map->data);
	if(!jobj) ERRX(_("Wrong JSON file"));
	get_obj_params(jobj, &globHole); // read global parameters
	// now try to find diaphragm bounding box & Z-coordinate
	json_get_bbox(&Dbox, jobj);
	// check for Z-coordinate
	if(dia){
		o = json_object_get_pair(jobj, "Z");
		if(!o) o = json_object_get_pair(jobj, "maskz");
		if(!o)
			ERRX(_("JSON file MUST contain floating point field \"Z\" or \"maskz\" with mask's coordinate"));
		dia->Z = get_jdouble(o); // read mask Z
	}
	o = json_object_get_pair(jobj, "holes");
	if(!o)
		ERRX(_("Corrupted file: no holes found!"));
	if(o->type == json_type_object){ // single hole
		HolesArray = MALLOC(aHole, 1);
		HolesNum = 1;
		json_object *obj = json_pair_get_object(o);
		if(!obj) ERRX(_("Wrong hole descriptor"));
		get_obj_params(obj, HolesArray);
		json_free_obj(&obj);
	}else if(o->type == json_type_obj_array){ // array of holes
		//DBG("array of holes");
		HolesArray = json_parse_holesarray(o, &HolesNum);
	}else{
		ERRX(_("Corrupted file: bad holes format!"));
	}
	if(!HolesArray || HolesNum < 1)
		ERRX(_("Didn't find any holes in json file!"));
	//DBG("Read %d holes", HolesNum);
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
	My_munmap(map);
	if(dia){
		dia->holes = HolesArray;
		dia->Nholes = HolesNum;
		memcpy(&dia->box, &Dbox, sizeof(BBox));
	}
	json_free_obj(&jobj);
	return HolesArray;
}

