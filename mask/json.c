#include <sys/mman.h>
#include <json/json.h>

#define FREE(ptr)			do{free(ptr); ptr = NULL;}while(0)

typedef struct{
	float x0; // left border
	float y0; // lower border
	float w;  // width
	float h;  // height
} BBox;
typedef enum{
	 H_SQUARE  // square hole
	,H_ELLIPSE // elliptic hole
	,H_UNDEF
} HoleType;
typedef struct{
	BBox box; // bounding box of hole
	int type; // type, in case of round hole borders of box are tangents to hole
} aHole;

aHole globHole;

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
			fprintf(stderr, "Wrong value! Get non-number!\n");
			exit(-1);
	}
	return val;
}

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
			fprintf(stderr, "Invalid file format! Found nested arrays!\n");
			exit(-1);
		}
		else if (type != json_type_object){
			arr[L++] = get_jdouble(jvalue);
		}
		else{ // non-numerical data?
			fprintf(stderr, "Invalid file format! Non-numerical data in array!\n");
			exit(-1);
		}
	}
	if(L == 0) FREE(arr);
	if(getLen) *getLen = L;
	return arr;
}

void get_obj_params(json_object *jobj, aHole *H){
	double *arr = NULL;
	int Len;
	enum json_type type;
	if(!H){
		fprintf(stderr, "Error: NULL instead of aHole structure!\n");
		exit(-1);
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
				fprintf(stderr, "\"radius\" array must consist of two doubles!\n");
				exit(-1);
			}
			H->box.w = arr[0] * 2.; H->box.h = arr[1] * 2.;
			FREE(arr);
		}else{
			fprintf(stderr, "\"radius\" must be a number or an array of two doubles!\n");
			exit(-1);
		}
	}
	o = json_object_object_get(jobj, "center");
	if(o){
		if(!(arr = json_get_array(o, &Len)) || Len != 2){
			fprintf(stderr, "\"center\" must contain an array of two doubles!\n");
			exit(-1);
		}
		H->box.x0 = arr[0] - H->box.w/2.;
		H->box.y0 = arr[1] - H->box.h/2.;
		FREE(arr);
	}else{
		o = json_object_object_get(jobj, "bbox");
		if(o){
			if(!(arr = json_get_array(o, &Len)) || Len != 4){
				fprintf(stderr, "\"bbox\" must contain an array of four doubles!\n");
				exit(-1);
			}
			H->box.x0 = arr[0]; H->box.y0 = arr[1]; H->box.w = arr[2]; H->box.h = arr[3];
			FREE(arr);
		}
	}
}

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
			fprintf(stderr, "Invalid holes array format!\n");
			exit(-1);
		}
	}
	if(getLen) *getLen = arraylen;
	return H;
}

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

int main(int argc, char **argv){
	char *ptr;
	struct stat statbuf;
	enum json_type type;
	int fd, i, HolesNum;
	aHole *HolesArray;
	size_t Mlen;
	if(argc == 2){
		if ((fd = open (argv[1], O_RDONLY)) < 0) err (1, "Can't open %s for reading", argv[1]);
		if (fstat (fd, &statbuf) < 0) err (1, "Fstat error");
		Mlen = statbuf.st_size;
		if ((ptr = mmap (0, Mlen, PROT_READ, MAP_PRIVATE, fd, 0)) == MAP_FAILED)
			err(1, "Mmap error for input");
	}else{
		fprintf(stderr, "No file given!\n");
		exit(-1);
	}
	json_object * jobj = json_tokener_parse(ptr);
	get_obj_params(jobj, &globHole);
	json_object *o = json_object_object_get(jobj, "holes");
	if(!o){
		fprintf(stderr, "Corrupted file: no holes found!\n");
		exit(-1);
	}
	type = json_object_get_type(o);
	if(type == json_type_object){ // single hole
		HolesArray = calloc(1, sizeof(aHole));
		assert(HolesArray);
		HolesNum = 1;
		get_obj_params(o, HolesArray);
	}else{ // array of holes
		HolesArray = json_parse_holesarray(o, &HolesNum);
	}
	if(!HolesArray || HolesNum < 1){
		fprintf(stderr, "Didn't find any holes in json file!\n");
		exit(-1);
	}
	printf("Readed %d holes\n", HolesNum);
	for(i = 0; i < HolesNum; i++){
		BBox *B = &HolesArray[i].box;
		printf("Hole %d: type=%s, bbox={%g, %g, %g, %g}\n", i, gettype(&HolesArray[i]),
				B->x0, B->y0, B->w, B->h);
	}
	munmap(ptr, Mlen);
	return 0;
}
