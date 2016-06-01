/*
 * wrapper.c - CPU / GPU wrapper, try to run function on GPU, if fail - on CPU
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

#define WRAPPER_C
#include "wrapper.h"
#include "usefull_macros.h"
#include "cmdlnopts.h"
#ifdef EBUG
	#include "saveimg.h"
#endif

// Functions to manipulate of forced CPU execution ============================>
void noCUDA(){ // force run on CPU
	Only_CPU = 1;
	CUforce = 0;
}
void tryCUDA(){ // return to default state
#ifdef CUDA_FOUND
	Only_CPU = 0;
	CUforce = 0;
#else
	/// "Приложение скомпилировано без поддержки CUDA"
	WARN(_("Tool was compiled without CUDA support"));
#endif
}
int CUsuccess(){ // test if CU faield
#ifdef CUDA_FOUND
	if(Only_CPU) return 0;
	else return CUnoerr;
#else
	/// "Приложение скомпилировано без поддержки CUDA"
	WARN(_("Tool was compiled without CUDA support"));
	return 0;
#endif
}
void forceCUDA(){ // not run on CPU even if GPU failed
#ifdef CUDA_FOUND
	Only_CPU = 0;
	CUforce = 1;
#else
	/// "Приложение скомпилировано без поддержки CUDA"
	ERRX(_("Tool was compiled without CUDA support"));
#endif
}

#ifdef CUDA_FOUND
int testCUDA(){
	size_t mem = 100 * MB;
	/// "В вычислениях по возможности будет использоваться GPU\n"
	red(_("In computations will try to use GPU\n"));
	int status = 0;
	char *errors[] = {
		"",
		/// "Ошибка получения свойств видеоядра"
		_("Can't get properties"),
		/// "Ошибка определения доступной памяти"
		_("Can't determine free memory"),
		/// "Ошибка выделения памяти"
		_("Can't allocate memory")
		};
	size_t theFree, theTotal;
	do{
		#define _TEST(F, st) if(!F){status = st;break;}
		_TEST(CUgetprops(), 1);
		_TEST(CUgetMEM(0, &theFree, &theTotal), 2);
		// if there is enough memory
		if(theFree / 4 > mem) mem = theFree / 4;
		// try to allocate GPU memory
		_TEST(CUallocaTest(mem), 3);
		#undef _TEST
	}while(0);
	if(status){
		if(CUforce) ERRX(_("Can't run CUDA!"));
		Only_CPU = 1;
		/// "Ошибка в инициализации CUDA!"
		WARN(_("Error in CUDA initialisation!"));
		WARN(errors[status]);
	}else{
		/// "ПАМЯТЬ: свободная = "
		printf(_("MEMORY: free = "));
		green("%zdMB,",  theFree / MB);
		/// " суммарная = "
		printf(_(" total= "));
		green("%zdMB\n", theTotal / MB);
	}
	return status;
}
#endif


// Init function ==============================================================>
/*
 * Init CUDA context and/or test memory allocation
 * name: getprops
 */
void getprops(){
	FNAME();
	size_t mem = 100 * MB;
	int status = 0;
	/// "Тест на выделение как минимум 100МБ памяти\n"
	printf("Make a test for allocation at least 100MB memory\n");
	if(Only_CPU){
		/// "В вычислениях используется только CPU\n"
		green(_("Will use only CPU in computations\n"));
	}
#ifdef CUDA_FOUND
	else status = testCUDA();
#endif
	// at last step - try to allocate main memory
	char *ptr = (char *) malloc(mem);
	/// "Ошибка выделения памяти"
	if(!ptr || !memset(ptr, 0xaa, mem)) ERR("Error allocating memory");
	free(ptr);
	/// "Тест выделения 100МБ памяти пройден успешно\n"
	printf(_("Allocation test for 100MB of memory passed\n"));
	if(!status){
		/// "\n\nВсе тесты пройдены успешно"
		green(_("\n\nAll tests succeed"));
		printf("\n\n");
	}
}

// Functions for pseudo-random number generators initialisation ===============>
/*
 * Generate a quasy-random number to initialize PRNG
 * name: throw_random_seed
 * @return value for curandSetPseudoRandomGeneratorSeed or srand48
 */
long throw_random_seed(){
	//FNAME();
	long r_ini;
	int fail = 0;
	int fd = open("/dev/random", O_RDONLY);
	do{
		if(-1 == fd){
			/// "Не могу открыть"
			WARN("%s /dev/random!", _("Can't open"));
			fail = 1; break;
		}
		if(sizeof(long) != read(fd, &r_ini, sizeof(long))){
			/// "Не могу прочесть"
			WARN("%s /dev/random!", _("Can't read"));
			fail = 1;
		}
		close(fd);
	}while(0);
	if(fail){
		double tt = dtime() * 1e6;
		double mx = (double)LONG_MAX;
		r_ini = (long)(tt - mx * floor(tt/mx));
	}
	return (r_ini);
}

// Build mask ===================================================>
/**
 * Make an array which elements identify corresponding hole's box for given
 * box (pixel) on mirror
 * It try to make mask size of minSz, but if light from some "pixels" will fall
 * onto more than one hole, size would be increased 2 times. Iterations of
 * increasing mask size would continue till photons from every "pixel" would
 * fall only on one hole
 *
 * Builded mask attached to input diaphragm as d->mask, before attach a freeDmask()
 * is called, so BE CAREFULL! First run of makeDmask() should be with d->mask == NULL!
 *
 * Mask is an image with cartesian coordinates -> it looks like horizontally mirroring!
 *
 * @param d - diaphragm for mask making
 * @param minSz - minimal mask size
 * @param M - mirror parameters
 * @param D - mirror deviations
 * @return allocated mask (MUST BE FREE by freeDmask) or NULL in case of error
 */
mirMask *makeDmask(Diaphragm *d, size_t minSz, mirPar *M, mirDeviations *D){
	FNAME();
	if(minSz > 4096){
		WARNX("Size of matrix (%d^2) too large!", minSz);
		return NULL;
	}
	if(minSz < 4) minSz = 4;
	size_t S2;
	int x, y, N;
	uint16_t *mdata = NULL, *ddata = NULL; // arrays for diaphragm & mask
	mirPar mirror;
	memcpy(&mirror, M, sizeof(mirPar));
	mirror.foc = d->Z; // preparing for getPhotonXY
	DBG("minSz = %zd", minSz);
	S2 = minSz * minSz;
	// width & height of "pixel"
	double scalex = d->box.w/(double)minSz, scaley = d->box.h/(double)minSz;

	ddata = MALLOC(uint16_t, S2);
	// fill diafragm mask to identify which hole to check when "photon"
	// falls into that region; fill only BBoxes!
	for(N = d->Nholes-1; N > -1; N--){
		BBox *b = &(d->holes[N].box);
		// coordinates of hole x0, y0 relative to diaphragm x0,y0
		double X0 = (b->x0 - d->box.x0)/scalex, Y0 = (b->y0 - d->box.y0)/scaley;
		// coordinates of upper right corner, add 2. to substitute <= to < in next cycles & to avoid border loss
		int x1 = (int)(X0 + b->w/scalex + 2.), y1 = (int)(Y0 + b->h/scaley + 2.);
		int x0 = (int)X0, y0 = (int)Y0;
		//DBG("scalex: %f scaley: %f, x0=%d, y0=%d, x1=%d, y1=%d\n", scalex, scaley, x0,y0,x1,y1);
		uint16_t mark = N + 1;
		if(y1 - y0 < 1 || x1 - x0 < 1){ // don't allow scale less than 1pix per hole
			DBG("Scale: too little");
			FREE(ddata);
			return makeDmask(d, minSz*2, M, D);
		}
		if(y1 > minSz) y1 = minSz;
		if(y0 < 0) y0 = 0;
		if(x1 > minSz) x1 = minSz;
		if(x0 < 0) x0 = 0;
		for(y = y0; y < y1; y++){
			uint16_t *ptr = &ddata[y*minSz + x0];
			for(x = x0; x < x1; x++, ptr++){
				if(*ptr && *ptr != mark){ // zone already occupied, make grid smaller
					DBG("Ooops, occupied zone (marker=%d, found %d); try minSz = %zd",mark, *ptr, minSz*2);
					FREE(ddata);
					return makeDmask(d, minSz*2, M, D);
				}
				*ptr = mark;
			}
		}
	}
	// prepare photons
	// they "falls" to corners of inner grid of size (minSz-1)^2
	S2 = (minSz-1)*(minSz-1);
	ALLOC(float, Xp, S2);
	ALLOC(float, Yp, S2);
	// prepare coordinates of "photons"
	double mY = 0., dist = 1./(double)(minSz - 2); // distance between points on grid 0..1
	float *xptr = Xp, *yptr = Yp;
	for(y = 1; y < minSz; y++, mY += dist){
		double mX = 0.;
		for(x = 1; x < minSz; x++, mX += dist){
			*yptr++ = mY;
			*xptr++ = mX;
		}
	}
	double mdxy = mirror.D/(double)minSz; // scale on mirror
	// box for grid
	BBox mirB = {-mirror.D/2.+mdxy, -mirror.D/2.+mdxy, mirror.D-2*mdxy, mirror.D-2*mdxy};
	DBG("mirbox: LD/UR = %g, %g, %g, %g",mirB.x0, mirB.y0, mirB.w+mirB.x0, mirB.h+mirB.y0);
	if(!getPhotonXY(Xp, Yp, 0, D, &mirror, S2, &mirB)){
		FREE(Xp); FREE(Yp); FREE(ddata);
		return NULL;
	}
	#ifdef EBUG
	float minx=1000.,miny=1000.,maxx=-1000., maxy=-1000.;
	for(x=0;x<S2;x++){if(minx > Xp[x])minx=Xp[x]; else if(maxx < Xp[x] && Xp[x] < 1e8) maxx = Xp[x];
		if(miny > Yp[x])miny=Yp[x]; else if(maxy < Yp[x] && Yp[x] < 1e8) maxy = Yp[x];}
	DBG("minx: %g, maxx: %g, miny: %g, maxy: %g", minx,maxx,miny,maxy);
	#endif
	float xleft = d->box.x0, ybot = d->box.y0;
	// and now fill mirror mask
	int S = minSz - 1;
	mdata = MALLOC(uint16_t, minSz*minSz);
	ALLOC(uint8_t, histo, d->Nholes);
	DBG("xleft: %g, scalex: %g", xleft, scalex);
	xptr = Xp; yptr = Yp;
	for(y = 0; y < S; y++){
		for(x = 0; x < S; x++, xptr++, yptr++){
			if(*xptr > 1e9 || *yptr > 1e9) continue; // miss to mirror
			int curX = (int)((*xptr - xleft) / scalex);
			if(curX < 0 || curX >= S) continue; // miss
			int curY = (int)((*yptr - ybot) / scaley);
			if(curY < 0 ||curY >= S) continue; // miss
			uint16_t mark = ddata[curY*minSz + curX];
			if(!mark) continue; // no hole
			int pix = y*minSz + x;
			int err = 0;
			histo[mark-1] = 1; // mark hole as proceeded
			#define CHECK(X) if(mdata[X]&&mdata[X]!=mark){err=1;}else{mdata[X]=mark;}
			CHECK(pix); // pixel to the left and down
			CHECK(pix+1); // right and down
			CHECK(pix+minSz); // left & up
			CHECK(pix+minSz+1); // right & up
			#undef CHECK
			if(err){ // zone already occupied, make grid smaller
				DBG("Ooops, occupied zone; try minSz = %zd", minSz*2);
				FREE(Xp); FREE(Yp); FREE(ddata); FREE(mdata); FREE(histo);
				return makeDmask(d, minSz*2, M, D);
			}
		}
	}
	FREE(Xp); FREE(Yp); FREE(ddata);
	// Now chek whether all holes are present
	for(x = 0; x < d->Nholes; x++)
		if(!histo[x]){
			DBG("Oooops! Missed a hole!");
			FREE(mdata); FREE(histo);
			return makeDmask(d, minSz*2, M, D);
		}
#ifdef EBUG
int totalpts = 0; S2 = minSz*minSz; uint16_t *ptr = mdata;
for(x=0;x<S2;x++,ptr++)if(*ptr)totalpts++;
red("Total non-zero points on mask: %d (%.1f%%)", totalpts,100.*(double)totalpts/(double)S2);
printf("\n");
#endif
	FREE(histo);
	ALLOC(mirMask, Mask, 1);
	Mask->WH = minSz;
	Mask->data = mdata;
	freeDmask(d->mask);
	d->mask = Mask;
	return Mask;
}

void freeDmask(mirMask *m){
	if(!m) return;
	FREE(m->data);
	FREE(m);
}

// Build image ================================================================>

/**
 * fill matrix image with photons
 * @param phX, phY - photons coordinates
 * @param ph_sz - number of photons
 * @param image - resulting image (photons **adds** to it)
 * @param imW, imH - size of resulting image
 * @param imbox - bounding box of resulting image
 * @return 0 if fails
 */
int fillImage(float *phX, float *phY, size_t ph_sz,
				float *image, size_t imW, size_t imH, BBox *imbox){
	//FNAME();
	float x0 = imbox->x0, y0 = imbox->y0, x1 = imbox->x0 + imbox->w, y1 = imbox->y0 + imbox->h;
	float dX = imbox->w / (float)(imW - 1), dY = imbox->h / (float)(imH - 1), x=0,y=0;
	size_t N;
/*	#ifdef EBUG
	float sum = 0., miss = 0., Xc = 0., Yc = 0.;
	#endif */
	for(N = 0; N < ph_sz; N++){
		x = phX[N]; y = phY[N];
		size_t X,Y;
		if(x < x0 || x > x1 || y < y0 || y > y1){
/*			#ifdef EBUG
			miss += 1.;
			#endif */
		}else{
			X = (size_t)((x - x0) / dX + 0.5);
			//Y = (size_t)((y1 - y) / dY + 0.5);
			Y = (size_t)((y - y0) / dY + 0.5);
			image[Y*imW + X] += 1.f;
/*			#ifdef EBUG
			sum += 1.;
			Xc += x; Yc += y;
			#endif*/
		}
	}
//	DBG("Photons on image: %g, missed: %g; TOTAL: %g\ncenter: Xc=%gmm, Yc=%gmm\nPI=%g",
//		sum,miss, sum+miss, Xc/sum*1000., Yc/sum*1000., 4.*sum/(sum+miss));
	return 1;
}
