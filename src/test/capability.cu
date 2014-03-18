#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

int main(int argc, char **argv){
	cudaDeviceProp dP;
	//CUdevice dev;
	//CUcontext ctx;
	if(cudaSuccess != cudaGetDeviceProperties(&dP, 0)) return 0;
	/*if(CUDA_SUCCESS != cuDeviceGet(&dev,0)) return 0;

	// create context for program run:
	if(CUDA_SUCCESS != cuCtxCreate(&ctx, 0, dev)) return 0;
	printf("\nDevice: %s, totalMem=%zd, memPerBlk=%zd,\n", dP.name, dP.totalGlobalMem, dP.sharedMemPerBlock);
	printf("warpSZ=%d, TPB=%d, TBDim=%dx%dx%d\n", dP.warpSize, dP.maxThreadsPerBlock,
			dP.maxThreadsDim[0],dP.maxThreadsDim[1],dP.maxThreadsDim[2]);
	printf("GridSz=%dx%dx%d, MemovrLap=%d, GPUs=%d\n", dP.maxGridSize[0],
			dP.maxGridSize[1],dP.maxGridSize[2],
			dP.deviceOverlap, dP.multiProcessorCount);
	printf("canMAPhostMEM=%d\n", dP.canMapHostMemory);
	printf("compute capability");*/
	printf("-arch=sm_%d%d\n", dP.major, dP.minor);
	//cuCtxDetach(ctx);
	return 0;
}
