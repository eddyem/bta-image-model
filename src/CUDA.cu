/*
 * 		CUDA.cu - subroutines for GPU
 *
 *      Copyright 2013 Edward V. Emelianoff <eddy@sao.ru>
 *
 *      This program is free software; you can redistribute it and/or modify
 *      it under the terms of the GNU General Public License as published by
 *      the Free Software Foundation; either version 2 of the License, or
 *      (at your option) any later version.
 *
 *      This program is distributed in the hope that it will be useful,
 *      but WITHOUT ANY WARRANTY; without even the implied warranty of
 *      MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *      GNU General Public License for more details.
 *
 *      You should have received a copy of the GNU General Public License
 *      along with this program; if not, write to the Free Software
 *      Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
 *      MA 02110-1301, USA.
 */
#include <cuda.h>
#include <curand.h>
#include <cufft.h>
#include <cuda_runtime.h>
#include "cutil_math.h"


//#include <cuda_runtime_api.h>
//#include <crt/device_runtime.h>
//#include <device_functions.h>

#define EXTERN extern "C"
#define CUDA_CU
#include "mkHartmann.h"
#include "wrapper.h"

int SHMEMSZ = 16383; // default constants, changed runtime
int QBLKSZ  = 16;		// QBLKSZ = sqrt(LBLKSZ)
int LBLKSZ  = 512;

cudaError_t CUerr;
inline int CUERROR(char *str){
	if(CUerr != cudaSuccess){
		WARN("%s, %s", str, cudaGetErrorString(CUerr));
		return 1;
	}else return 0;
}

// Default error macro (return fail)
#define RETMACRO do{return 0;} while(0)
/*
 *  memory macros
 *
 * Each function using them must return an int type:
 *   1 - in case of success
 *   0 - in case of fail
 *
 * Function that use them should check this status and if fails,
 * call CPU-based function for same computations
 */
static int ret;
/*
 * this macro is for functions with alloc
 * All functions should in their beginning call "ret = 1;"
 * and in the end they should have a label "free_all:" after which
 * located memory free operators
 * in the end of functions should be "return ret;"
 */
#define FREERETMACRO do{ret = 0; goto free_all;} while(0)

#define CUALLOC(var, size)		do{				\
	CUerr = cudaMalloc((void**)&var, size);		\
	if(CUERROR("CUDA: can't allocate memory")){	\
		FREERETMACRO;							\
}}while(0)
#define CUALLOCPITCH(var, p, W, H) do{			\
	CUerr = cudaMallocPitch((void**)&var, p, W, H);\
	if(CUERROR("CUDA: can't allocate memory")){	\
		FREERETMACRO;							\
}}while(0)
#define CUMOV2DEV(dest, src, size) do{			\
	CUerr = cudaMemcpy(dest, src, size,			\
				cudaMemcpyHostToDevice);		\
	if(CUERROR("CUDA: can't copy data to device")){\
		FREERETMACRO;							\
}}while(0)
#define CUMOV2DEVPITCH(dst, dp, src, w, h) do{	\
	CUerr = cudaMemcpy2D(dst, dp, src, w, w, h,\
				cudaMemcpyHostToDevice);		\
	if(CUERROR("CUDA: can't copy data to device")){\
		FREERETMACRO;							\
}}while(0)
#define CUMOV2HOST(dest, src, size) do{			\
	CUerr = cudaMemcpy(dest, src, size,			\
				cudaMemcpyDeviceToHost);		\
	if(CUERROR("CUDA: can't copy data to host")){\
		FREERETMACRO;							\
}}while(0)
#define CUMOV2HOSTPITCH(dst,src,spitch,w,h) do{	\
	CUerr = cudaMemcpy2D(dst,w,src,spitch,w,h, \
				cudaMemcpyDeviceToHost);		\
	if(CUERROR("CUDA: can't copy data to device")){\
		FREERETMACRO;							\
}}while(0)
#define CUFREE(var) do{cudaFree(var); var = NULL; }while(0)
#define  CUFFTCALL(fn)		do{					\
	cufftResult fres = fn;						\
	if(CUFFT_SUCCESS != fres){					\
		WARN("CUDA fft error %d", fres);		\
		FREERETMACRO;							\
}}while(0)

texture<float, 2> cuTex;//1, cuTex2, cuTex3;
#define CUTEXTURE(t, data, W, H, pitch) do{		\
	CUerr = cudaBindTexture2D(NULL, t,			\
			data, W, H, pitch);					\
	if(CUERROR("CUDA: can't bind texture")){	\
		FREERETMACRO;}else{						\
	t.addressMode[0] = cudaAddressModeClamp;	\
	t.addressMode[1] = cudaAddressModeClamp;	\
	t.normalized = false;  						\
	t.filterMode = cudaFilterModePoint;			\
}}while(0)

//#define _TEXTURE_(N, ...) CUTEXTURE(cuTex ## N, __VA_ARGS__)
//#define TEXDATA(...) _TEXTURE_(__VA_ARGS__)
//#define TEXTURE(N) cuTex ## N
#define TEXDATA(...) CUTEXTURE(cuTex, __VA_ARGS__)
#define TEXTURE() cuTex

/**
 * getting the videocard parameters
 * @return 0 if check failed
 */
EXTERN int CUgetprops(){
	cudaDeviceProp dP;
	CUdevice dev;
	CUcontext ctx;
	if(cudaSuccess != cudaGetDeviceProperties(&dP, 0)) return 0;
	if(CUDA_SUCCESS != cuDeviceGet(&dev,0)) return 0;
	// create context for program run:
	if(CUDA_SUCCESS != cuCtxCreate(&ctx, 0, dev)) return 0;
	printf("\nDevice: %s, totalMem=%zd, memPerBlk=%zd,\n", dP.name, dP.totalGlobalMem, dP.sharedMemPerBlock);
	printf("warpSZ=%d, TPB=%d, TBDim=%dx%dx%d\n", dP.warpSize, dP.maxThreadsPerBlock,
			dP.maxThreadsDim[0],dP.maxThreadsDim[1],dP.maxThreadsDim[2]);
	printf("GridSz=%dx%dx%d, MemovrLap=%d, GPUs=%d\n", dP.maxGridSize[0],
			dP.maxGridSize[1],dP.maxGridSize[2],
			dP.deviceOverlap, dP.multiProcessorCount);
	printf("canMAPhostMEM=%d\n", dP.canMapHostMemory);
	printf("compute capability");
	green(" %d.%d.\n\n", dP.major, dP.minor);
	if(dP.major > 1){
		SHMEMSZ = 49151; QBLKSZ = 32; LBLKSZ = 1024;
	}
	// cuCtxDetach(ctx);
	return 1;
}

/**
 * check whether there is enough memory
 * @param memsz - memory needed
 * @param Free  - free memory (maybe NULL if not needed)
 * @param Total - total memory (maybe NULL if not needed)
 * @return 0 if check failed
 */
EXTERN int CUgetMEM(size_t memsz, size_t *Free, size_t *Total){
	size_t theFree = 0, theTotal = 0;
	if(CUDA_SUCCESS != cuMemGetInfo( &theFree, &theTotal )) return 0;
	if(Free) *Free = theFree;
	if(Total) *Total = theTotal;
	if(theFree < memsz) return 0;
	return 1;
}

/*
size_t theFree, theTotal;
CUgetMEM(0, &theFree, &theTotal);
printf(_("MEMORY: free = "));
green("%zdMB,",  theFree / MB);
printf(_(" total= "));
green("%zdMB\n", theTotal / MB);
*/

/**
 * Memory allocation & initialisation test
 * @param memsz - wanted memory to allocate
 * @return 0 if check failed
 */
EXTERN int CUallocaTest(size_t memsz){
	char *mem;
	ret = 1;
	printf("cudaMalloc(char, %zd)\n", memsz);
	CUALLOC(mem, memsz);
	if(cudaSuccess != cudaMemset(mem, 0xaa, memsz)) ret = 0;
free_all:
	CUFREE(mem);
	return ret;
}

/*
 *
 *
 * RANDOM NUMBER generation block  ============================================>
 *
 *
 */
/*
 * Possible functions:
 * curandGenerateUniform(curandGenerator_t generator, float *outputPtr, size_t num)
 * curandGenerateNormal(curandGenerator_t generator, float *outputPtr, size_t n, float mean, float stddev)
 * curandGenerateLogNormal(curandGenerator_t generator, float *outputPtr, size_t n, float mean, float stddev)
 * curandGeneratePoisson(curandGenerator_t generator, unsigned int *outputPtr, size_t n, double lambda)
 */
#define CURAND_CALL(x) do {						\
	curandStatus_t st = x;						\
	if(st!=CURAND_STATUS_SUCCESS) {				\
		WARN("CURAND error %d", st);			\
		FREERETMACRO;							\
}} while(0)


__global__ void multiply_rand(float *arr, size_t sz, float Amp){
	size_t IDX = threadIdx.x + blockDim.x * blockIdx.x;
	if(IDX >= sz) return;
	arr[IDX] *= Amp;
}
static int rand_initialized = 0;
static curandGenerator_t gen;
/**
 * Fills linear array of size sz by random float numbers
 * @param sz  - size of array
 * @param arr - array data (allocated outside)
 * @param Amp - amplitude of array
 * @return 0 if failed
 */
EXTERN int CUfillrandarr(size_t sz, float *arr, float Amp){
	FNAME();
	ret = 1;
	float *devmem;
	if(!rand_initialized){
		CURAND_CALL(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
		CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, throw_random_seed()));
		rand_initialized = 1;
	}
	CUALLOC(devmem, sz*sizeof(float));
	CURAND_CALL(curandGenerateUniform(gen, devmem, sz));
	cudaThreadSynchronize();
	if(fabs(Amp < 1.f) > FLT_EPSILON){
		size_t dimens = (sz + LBLKSZ - 1) / LBLKSZ;
		multiply_rand<<<dimens, LBLKSZ>>>(devmem, sz, Amp);
		cudaThreadSynchronize();
	}
	CUMOV2HOST(arr, devmem, sz*sizeof(float));
free_all:
	//CURAND_CALL(curandDestroyGenerator(gen));
	CUFREE(devmem);
	return ret;
}
/*
 *
 *
 * <=====================================  end of random number generation block
 *
 *
 */

/*
 *
 *
 * BICUBIC interpolation block  ===============================================>
 *
 *
 */
/**
 * matrix operation:
 *
 * p(t, a, b, c, d) =
 *                           /  0  2  0  0 \   / a \
 *   = 1/2 ( 1 t t^2 t^3 ) * | -1  0  1  0 | * | b |
 *                           |  2 -5  4 -1 |   | c |
 *                           \ -1  3 -3  1 /   \ d /
 */
inline __device__ float p_ta(float t, float t2, float t3,
						float a,float b,float c,float d){
	return (2*b + t*(-a+c) + t2*(2*a-5*b+4*c-d) + t3*(-a+3*b-3*c+d)) / 2.f;
}
// bicubic interpolation of texture in point with coordinates (x, y)
__device__ float interpolate_bicubic(float x, float y){
	#define TEX(X,Y) tex2D(TEXTURE(), X, y0+(Y))
	#define TX(Y) p_ta(pt.x, pt2.x, pt3.x, TEX(x0-1,Y), \
			TEX(x0,Y), TEX(x0+1,Y), TEX(x0+2,Y))
	float2 crd = make_float2(x,y);
	float2 zeropt = floor(crd);
	float2 pt = crd - zeropt;
	float2 pt2 = pt*pt;
	float2 pt3 = pt*pt2;
	float x0 = zeropt.x, y0 = zeropt.y;
	//return x;
	return p_ta(
		pt.y, pt2.y, pt3.y,
		TX(-1), TX(0), TX(1), TX(2)
	);
	#undef TEX
	#undef TX
}

/**
 * calculate base coordinates for new image in point (X,Y) relative to old image
 * @param out - new image data
 * @param fracX, fracY - scale of new image pixel relative to old image
 * @parami oW, oH - new image dimensions
 */
__global__ void bicubic_interp(float *out,
								float fracX, float fracY,
								unsigned int oW, unsigned int oH){
	int X, Y;  // pixel coordinates on output
	float x,y; // coordinates on output in value of input
	X = threadIdx.x + blockDim.x * blockIdx.x;
	Y = threadIdx.y + blockDim.y * blockIdx.y;
	if(X >= oW || Y >= oH) return;
	x = (float)X * fracX;
	y = (float)Y * fracY;
	out[Y*oW+X] = interpolate_bicubic(x, y);
}

/**
 * Interpolation of image by bicubic splines
 * @param out_dev - new image data in device memory
 * @param in - old image data
 * @param oH, oW  - new image size
 * @param iH, iW  - old image size
 * @return 0 if fails
 */
EXTERN int CUbicubic_interpDEV(float **out_dev, float *in,
								size_t oH, size_t oW, size_t iH, size_t iW){
	FNAME();
	size_t iWp = iW*sizeof(float), oSz = oH*oW*sizeof(float), pitch;
	float *in_dev = NULL;
	ret = 1;
	dim3 blkdim(QBLKSZ, QBLKSZ);
	dim3 griddim((oW+QBLKSZ-1)/QBLKSZ, (oH+QBLKSZ-1)/QBLKSZ);
	CUALLOCPITCH(in_dev, &pitch, iWp, iH);
	CUMOV2DEVPITCH(in_dev, pitch, in, iWp, iH);
	CUALLOC(*out_dev, oSz);
	TEXDATA(in_dev, iW, iH, pitch);
	bicubic_interp<<<griddim, blkdim>>>(*out_dev,
				(float)(iW-1)/(oW-1), (float)(iH-1)/(oH-1), oW, oH);
	cudaThreadSynchronize();
free_all:
	cudaUnbindTexture(TEXTURE());
	CUFREE(in_dev);
	return ret;
}
/**
 * Interpolation of image by bicubic splines
 * @param out, in - new and old images data
 * @param oH, oW  - new image size
 * @param iH, iW  - old image size
 * @return 0 if fails
 */
EXTERN int CUbicubic_interp(float *out, float *in,
								size_t oH, size_t oW, size_t iH, size_t iW){
	FNAME();
	float *out_dev = NULL;
	ret = 1;
	if(!CUbicubic_interpDEV(&out_dev,in,oH,oW,iH,iW)){FREERETMACRO;}
	CUMOV2HOST(out, out_dev, oH*oW*sizeof(float));
free_all:
	CUFREE(out_dev);
	return ret;
}
/*
 *
 *
 * <========================================= end of BICUBIC interpolation block
 *
 *
 */


/*
 *
 *
 * Begin of mirror normales block =============================================>
 *
 *
 */


/**
 * Compute X & Y components of "curved mirror" gradient
 * This components (interpolated) should be added to SURFACE gradient **before** normalisation
 * @param Sz - image size in both directions
 * @param mirPixSZ - pixel size in meters
 * @param dZdX, dZdY - addition gradient components
 *
 *
 * Normale of "curved mirror" computes so:
 * N1 = -i*x/2f -j*y/2f + k  ==> direction vector of ideal mirror
 * dN ==> bicubic interpolation of gradient matrix of mirror surface deviations
 * dN = i*dX + j*dY
 * N = N1 + dN ==> "real" direction vector
 * |N| = length(N) ==> its length
 * n = N / |N| ==> normale to "real" mirror surface
 * the normale n can be transformed due to mirror inclination and displation
 *
 * Let f be a direction vector of falling photon, then a=[-2*dot(f,n)*n]
 * will be a twice projection of f to n with "-" sign (dot(x,y) == scalar product)
 * reflected direction vector will be
 * r = a + f ==>
 *         r = f - 2*dot(f,n)*n
 * r doesn't need normalisation
 *
 * If (x0,y0,z0) is mirror surface coordinates, z0 = z(x,y) + dZ, wehere dZ interpolated;
 * then image coordinates on plane z=Zx would be:
 * X = x0 + k*rx,   Y = y0 + k*ry,   k = (Zx - z0) / rz.
 *
 * First check point on a mask plane; if photon falls to a hole in mask, increment
 * value on corresponding image pixel
 */
// 1./sqrt(2)
#define DIVSQ2  0.7071068f
// 2*(1+2/sqrt(2))
#define WEIGHT 482.842712f
__global__ void calcMir_dXdY(size_t Sz, float mirPixSZ,
					float *dZdX, float *dZdY){
	#define TEX(X,Y) tex2D(TEXTURE(), X0+(X), Y0+(Y))*100.f
	#define PAIR(X,Y)  (TEX((X),(Y))-TEX(-(X),-(Y)))
	int X0,Y0;
	X0 = threadIdx.x + blockDim.x * blockIdx.x;
	Y0 = threadIdx.y + blockDim.y * blockIdx.y;
	// check if we are inside image
	if(X0 >= Sz || Y0 >= Sz) return;
	// calculate gradient components
	int idx = Y0 * Sz + X0;
	mirPixSZ *= WEIGHT;
	dZdX[idx] = (PAIR(1,0) + DIVSQ2*(PAIR(1,-1)+PAIR(1,1)))/mirPixSZ;
	// REMEMBER!!! Axe Y looks up, so we must change the sign
	dZdY[idx] = -(PAIR(0,1) + DIVSQ2*(PAIR(1,1)+PAIR(-1,1)))/mirPixSZ;
	#undef PAIR
	#undef TEX
}

/**
 * Compute matrices of mirror surface deviation
 * @param map, mapWH - square matrix of surface deviations (in meters) and its size
 * @param mirDia - mirror diameter
 * @param mirWH - size of output square matrices (mirWH x mirWH)
 * @param mirZ  - matrix of mirror Z variations (add to mirror Z)
 * @param mirDX, mirDY - partial derivatives dZ/dx & dZ/dy (add to mirror der.)
 * @return 0 if fails
 */
EXTERN int CUmkmirDeviat(float *map, size_t mapWH, float mirDia,
						mirDeviations * mirDev){
	FNAME();
	ret = 1;
	size_t mirWH = mirDev->mirWH;
	float *mirDXd = NULL, *mirDYd = NULL, *mirZd = NULL;
	size_t mirSp = mirWH*sizeof(float);
	size_t mirSz = mirWH*mirSp, dimens = (mirWH+QBLKSZ-1)/QBLKSZ;
	size_t pitch;
	dim3 blkdim(QBLKSZ, QBLKSZ);
	dim3 griddim(dimens, dimens);
	// make Z -- simple approximation of
	if(!CUbicubic_interpDEV(&mirZd,map,mirWH,mirWH,mapWH,mapWH)){FREERETMACRO;}
	CUMOV2HOST(mirDev->mirZ, mirZd, mirSz);
	CUFREE(mirZd);
	// Z-data would be in pitched 2D array for texture operations
	//      (to simplify "stretching")
	CUALLOCPITCH(mirZd, &pitch, mirSp, mirWH);
	CUMOV2DEVPITCH(mirZd, pitch, mirDev->mirZ, mirSp, mirWH);
	TEXDATA(mirZd, mirWH, mirWH, pitch);

	CUALLOC(mirDXd, mirSz);
	CUALLOC(mirDYd, mirSz);
	calcMir_dXdY<<<griddim, blkdim>>>(mirWH, mirDia/((float)(mirWH-1)), mirDXd, mirDYd);
	cudaThreadSynchronize();
	CUMOV2HOST(mirDev->mirDX, mirDXd, mirSz);
	CUMOV2HOST(mirDev->mirDY, mirDYd, mirSz);
free_all:
	cudaUnbindTexture(TEXTURE());
	CUFREE(mirDXd);
	CUFREE(mirDYd);
	CUFREE(mirZd);
	return ret;
}
/*
 *
 *
 * <==============================================  end of mirror normales block
 *
 *
 */

/*
 *
 *
 * Photons trace block ========================================================>
 *
 *
 */
typedef struct{
	float3 l1;
	float3 l2;
	float3 l3;
}Matrix;

// rotation of vec on rotation matrix with lines rM1, rM2, rM3
__inline__ __device__ float3 rotZA(Matrix *M, float3 vec){
	return make_float3(dot(M->l1,vec), dot(M->l2,vec), dot(M->l3, vec));
}

// textures for linear interpolation of mirror deviations
texture<float, 2> TZ;
texture<float, 2> TdX;
texture<float, 2> TdY;
// struct for reducing parameters amount
typedef struct{
	Matrix M;
	mirPar P;
	BBox B;
	bool rot;
	Diaphragm D;
}MPB;
/**
 * Calculate reflected photon coordinates at plane Z0
 * @param photonX, photonY **INOUT** -
 * 				out: arrays with photon coordinates on image plane
 * 				 in: coordinates in diapazone [0,1)
 * @param photonSZ - size of prevoius arrays
 * @param Z0 - Z-coordinate of image plane
 * @param M - mirror rotation matrix or NULL if rotation is absent
 * @param mir_WH - mirror derivations matrix size
 * @param Parms - mirror parameters
 * @param f - light direction vector
 *
 * @param holes - array with
 */
__global__ void getXY(float *photonX, float *photonY, size_t photonSZ,
				float Z0, MPB *mpb, size_t mir_WH, float3 f){
	#define BADPHOTON() do{photonX[IDX] = 1e10f; photonY[IDX] = 1e10f; return;}while(0)
	size_t IDX;  // pixel number in array
	IDX = threadIdx.x + blockDim.x * blockIdx.x;
	if(IDX >= photonSZ) return;
	Matrix *M = &mpb->M;
	mirPar *Parms = &mpb->P;
	BBox *box = &mpb->B;
	float _2F = Parms->F * 2.f, R = Parms->D / 2.f;
	float x,y,z; // coordinates on mirror in meters
	x = box->x0 + photonX[IDX] * box->w; // coords of photons
	y = box->y0 + photonY[IDX] * box->h;
	float r2 = x*x + y*y;
	z = r2 / _2F / 2.f;
	// we don't mean large inclination so check border by non-inclinated mirror
	if(r2 > R*R) BADPHOTON();
	float pixSZ = Parms->D / float(mir_WH-1); // "pixel" size on mirror
	// coordinates on deviation matrix, don't forget about y-mirroring!
	float xOnMat = (x + R) / pixSZ, yOnMat = (R - y) / pixSZ;
	// now add z-deviations, linear interpolation of pre-computed matrix
	z += tex2D(TZ, xOnMat, yOnMat);
	// point on unrotated mirror
	float3 point = make_float3(x,y,z);
	// compute normals to unrotated mirror
	float3 normal = make_float3(	tex2D(TdX, xOnMat, yOnMat) - x/_2F,
									tex2D(TdY, xOnMat, yOnMat) - y/_2F,
									1.f
								);
	// rotate mirror
	if(mpb->rot){
		point  = rotZA(M, point);
		normal = rotZA(M, normal);
	}
	normal = normalize(normal);
	// calculate reflection direction vector
	float3 refl = 2.f*fabs(dot(f, normal))*normal + f;
	float K;
	if(mpb->D.Nholes){ // there is a diaphragm - test it
		Diaphragm *D = &(mpb->D);
		int S = D->mask->WH; // size of diaphragm matrix
		K = (D->Z - point.z) / refl.z; // scale to convert normal to vector
		float xleft = D->box.x0, ybot = D->box.y0; // left bottom angle of dia box
		float scalex = D->box.w/(float)S, scaley = D->box.h/(float)S;
		x = point.x + K*refl.x;
		y = point.y + K*refl.y;
		int curX = (int)((x - xleft) / scalex + 0.5f); // coords on dia matrix
		int curY = (int)((y - ybot) / scaley + 0.5f);
		if(curX < 0 || curX >= S || curY < 0 || curY >= S) BADPHOTON();
		uint16_t mark = D->mask->data[curY*S + curX];
		if(!mark) BADPHOTON();
		do{
			int t = D->holes[mark-1].type;
			BBox *b = &D->holes[mark-1].box;
			float rx = b->w/2.f, ry = b->h/2.f;
			float xc = b->x0 + rx, yc=b->y0 + ry;
			float sx = x - xc, sy = y - yc;
			switch(t){
				case H_SQUARE:
					if(fabs(sx) > rx || fabs(sy) > ry) mark = 0;
				break;
				case H_ELLIPSE:
					if(sx*sx/rx/rx+sy*sy/ry/ry > 1.f) mark = 0;
				break;
				default:
					mark = 0;
			}
		}while(0);
		if(!mark) BADPHOTON();
	}
	// OK, test is passed, calculate position on Z0:
	K = (Z0 - point.z) / refl.z;
	x = point.x + K*refl.x;
	y = point.y + K*refl.y;
	photonX[IDX] = x;
	photonY[IDX] = y;
	#undef BADPHOTON
}

/**
 * get coordinates of photons falling onto mirror at point X,Y, reflected
 * to plane with z=mirParms->foc
 *
 * @param xout, yout - arrays of size N_photons with coordinates (allocated outside)
 * @param R - use photon coordinates from xout, yout or generate random (in interval [0,1))
 * @param D - deviation & its derivative matrices
 * @param mirParms - parameters of mirror
 * @param N_photons - number of photons for output (not more than size of input!)
 * @param box - bbox where photons are falling
 * @return 0 if failed
 */
EXTERN int CUgetPhotonXY(float *xout, float *yout, int R, mirDeviations *D,
				mirPar *mirParms, size_t N_photons, BBox *box){
	//FNAME();
	ret = 1;
	MPB *mpbdev = NULL, mpb;
	aHole *hdev = NULL;
	mirMask *mmdev = NULL;
	uint16_t *mdatadev = NULL;
	float *X = NULL, *Y = NULL;
	float SZ = sin(mirParms->objZ);
	float z = mirParms->foc;
	float *Z_dev = NULL, *dX_dev = NULL, *dY_dev = NULL;
	float A = mirParms->Aincl, Z = mirParms->Zincl;
	size_t H = D->mirWH, Wp = H*sizeof(float), pitch;
	float cA = cos(A), sA = sin(A), cZ = cos(Z), sZ = sin(Z);
	size_t sz = N_photons * sizeof(float);
	size_t dimens = (N_photons+LBLKSZ-1)/LBLKSZ;
	// light direction vector
	float3 f = make_float3(-SZ*sin(mirParms->objA), -SZ*cos(mirParms->objA), -cos(mirParms->objZ));
	// rotation matrix by Z than by A:
	Matrix M, *Mptr = NULL;
	if(A != 0.f || Z != 0.f){
		M.l1 = make_float3(cA, sA*cZ, sA*sZ);
		M.l2 = make_float3(-sA, cA*cZ, cA*sZ);
		M.l3 = make_float3(0.f, -sZ, cZ);
		Mptr = &M;
	}
	// textures for linear interpolation
	CUALLOCPITCH(Z_dev, &pitch, Wp, H);
	CUALLOCPITCH(dX_dev, &pitch, Wp, H);
	CUALLOCPITCH(dY_dev, &pitch, Wp, H);
	CUMOV2DEVPITCH(Z_dev, pitch, D->mirZ, Wp, H);
	CUMOV2DEVPITCH(dX_dev, pitch, D->mirDX, Wp, H);
	CUMOV2DEVPITCH(dY_dev, pitch, D->mirDY, Wp, H);
	#define CUTEX(tex, var) CUTEXTURE(tex, var, H, H, pitch); tex.filterMode = cudaFilterModeLinear
	CUTEX(TZ, Z_dev);
	CUTEX(TdX, dX_dev);
	CUTEX(TdY, dY_dev);
	#undef CUTEX
	CUALLOC(X, sz);
	CUALLOC(Y, sz);
	if(R){
		// create __device__ random arrays X & Y
		if(!rand_initialized){
			CURAND_CALL(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
			CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, throw_random_seed()));
			rand_initialized = 1;
		}
		CURAND_CALL(curandGenerateUniform(gen, X, N_photons));
		cudaThreadSynchronize();
		CURAND_CALL(curandGenerateUniform(gen, Y, N_photons));
		cudaThreadSynchronize();
	}else{
		// move xout, yout to X,Y
		CUMOV2DEV(X, xout, sz);
		CUMOV2DEV(Y, yout, sz);
	}
	CUALLOC(mpbdev, sizeof(MPB));
	if(Mptr){
		mpb.rot = true;
		memcpy(&mpb.M, Mptr, sizeof(Matrix));
	}else{
		mpb.rot = false;
	}
	memcpy(&mpb.P, mirParms, sizeof(mirPar));
	memcpy(&mpb.B, box, sizeof(BBox));
	if(!mirParms->dia) mpb.D.Nholes = 0;
	// initialize diaphragm
	if(mirParms->dia){ // diaphragm is present - allocate memory for it
		Diaphragm tmpd;
		mirMask tmpM;
		memcpy(&tmpd, mirParms->dia, sizeof(Diaphragm));
		memcpy(&tmpM, mirParms->dia->mask, sizeof(mirMask));
		size_t S = sizeof(aHole) * mirParms->dia->Nholes;
		CUALLOC(hdev, S);
		CUMOV2DEV(hdev, mirParms->dia->holes, S);
		tmpd.holes = hdev;
		S = mirParms->dia->mask->WH;
		S = sizeof(uint16_t) * S * S;
		CUALLOC(mdatadev, S);
		CUMOV2DEV(mdatadev, mirParms->dia->mask->data, S);
		tmpM.data = mdatadev;
		CUALLOC(mmdev, sizeof(mirMask));
		CUMOV2DEV(mmdev, &tmpM, sizeof(mirMask));
		tmpd.mask = mmdev;
		memcpy(&mpb.D, &tmpd, sizeof(Diaphragm));
	}
	CUMOV2DEV(mpbdev, &mpb, sizeof(MPB));
	getXY<<<dimens, LBLKSZ>>>(X, Y, N_photons, z, mpbdev, H, f);
	cudaThreadSynchronize();
	CUMOV2HOST(xout, X, sz);
	CUMOV2HOST(yout, Y, sz);
free_all:
	CUFREE(hdev); CUFREE(mmdev); CUFREE(mdatadev);
	//CURAND_CALL(curandDestroyGenerator(gen));
	cudaUnbindTexture(TZ);
	cudaUnbindTexture(TdX);
	cudaUnbindTexture(TdY);
	CUFREE(mpbdev);
	CUFREE(Z_dev); CUFREE(dX_dev); CUFREE(dY_dev);
	CUFREE(X); CUFREE(Y);
	return ret;
}
/*
 *
 *
 * <================================================= end of Photons trace block
 *
 *
 */




/*
typedef struct{
	float x0; // (x0,y0) - left lower corner
	float y0;
	float x1; // (x1,y1) - right upper corner
	float y1;
	size_t pitch; // data pitch in array !!IN NUMBER OF ELEMENTS!!
	float dX; // pixel size, dX = (x1-x0) / [image width - 1 ]
	float dY; // pixel size, dY = (y1-y0) / [image height - 1]
}devBox;

__global__ void fillimage(float *image, float *xx, float *yy, size_t sz, devBox *b){
	size_t IDX = threadIdx.x + blockDim.x * blockIdx.x, X, Y;
	if(IDX >= sz) return;
	float x = xx[IDX], y = yy[IDX];
	if(x < b->x0 || x > b->x1) return;
	if(y < b->y0 || y > b->y1) return;
	X = (size_t)((x - b->x0) / b->dX);
	Y = (size_t)((b->y1 - y) / b->dY);
	//atomicAdd(&image[Y*b->pitch + X], 1.f)
	image[Y*b->pitch + X] += 1.f;
}*/
/**
 *
 * @param phX, phY - photons coordinates
 * @param ph_sz - number of photons
 * @param image - resulting image (photons **adds** to it)
 * @param imW, imH - size of resulting image
 * @param imbox - bounding box of resulting image
 * @return 0 if fails
 */
 /*
EXTERN int CUfillImage(float *phX, float *phY, size_t ph_sz,
				float *image, size_t imW, size_t imH, BBox *imbox){
	ret = 1;
	size_t linsz = ph_sz * sizeof(float);
	size_t pitch, iWp = imW * sizeof(float);
	float *xdev = NULL, *ydev = NULL, *imdev = NULL;
	devBox *bdev = NULL, devbox;
	size_t dimens = (ph_sz+LBLKSZ-1)/LBLKSZ;

	CUALLOC(xdev, linsz); CUALLOC(ydev, linsz);
	CUALLOC(bdev, sizeof(devBox));
	CUALLOCPITCH(imdev, &pitch, iWp, imH);
DBG("pitch: %zd", pitch);
	CUMOV2DEVPITCH(imdev, pitch, image, iWp, imH);
	CUMOV2DEV(xdev, phX, linsz);
	CUMOV2DEV(ydev, phY, linsz);
	devbox.x0 = imbox->x0; devbox.y0 = imbox->y0;
	devbox.x1 = imbox->x0 + imbox->w;
	devbox.y1 = imbox->y0 + imbox->h;
	devbox.pitch = pitch / sizeof(float);
	devbox.dX = imbox->w / (float)(imW - 1);
	devbox.dY = imbox->h / (float)(imH - 1);
	CUMOV2DEV(bdev, &devbox, sizeof(devBox));
	fillimage<<<dimens, LBLKSZ>>>(imdev, xdev, ydev, ph_sz, bdev);
	cudaThreadSynchronize();
	CUMOV2HOSTPITCH(image, imdev, pitch, iWp, imH);
free_all:
	CUFREE(xdev); CUFREE(ydev);
	CUFREE(imdev); CUFREE(bdev);
	return ret;
}
*/
