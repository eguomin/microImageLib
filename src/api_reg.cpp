#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>   
#include <iostream>
#include <fstream>
#include <string.h>
#include <math.h>
#include <ctime>
#include <time.h>
#ifdef _WIN32
#include <windows.h>
#include <tchar.h>
#include <strsafe.h>
//#else
//#include <sys/stat.h>
#endif

// Includes CUDA
//#include <cuda.h>
//#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <cufft.h>
//#include <cufftw.h> // ** cuFFT also comes with CPU-version FFTW, but seems not to work when image size is large.
#include "fftw3.h"
#include <memory>
#include "device_launch_parameters.h"

extern "C"{
#include "powell.h"
}
#define PROJECT_EXPORTS // To export API functions
#include "libapi.h"
#undef PROJECT_EXPORTS
#include "apifunc_internal.h"
#define blockSize 1024
#define blockSize2Dx 32
#define blockSize2Dy 32
#define blockSize3Dx 16
#define blockSize3Dy 8
#define blockSize3Dz 8
#define NDIM 12
#define SMALLVALUE 0.01

extern cudaError_t __err;
#define cudaCheckErrors(msg) \
    do { \
        __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                msg, cudaGetErrorString(__err), \
                __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            exit(1); \
								        } \
				    } while (0)

///// affine transformation
int atrans3dgpu(float *h_reg, float *iTmx, float *h_img2, unsigned int *imSize1, unsigned int *imSize2, int deviceNum){
	//image size
	long long int sx1 = imSize1[0], sy1 = imSize1[1], sz1 = imSize1[2];
	long long int sx2 = imSize2[0], sy2 = imSize2[1], sz2 = imSize2[2];
	// total pixel count for each images
	long long int totalSize1 = sx1*sy1*sz1;
	long long int totalSize2 = sx2*sy2*sz2;
	// GPU device
	cudaSetDevice(deviceNum);
	float *d_img3DTemp;
	cudaMalloc((void **)&d_img3DTemp, totalSize1 *sizeof(float));
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
	cudaArray *d_ArrayTemp;
	cudaMalloc3DArray(&d_ArrayTemp, &channelDesc, make_cudaExtent(sx2, sy2, sz2));
	cudaThreadSynchronize();
	cudaCheckErrors("****GPU array memory allocating fails !!!!*****\n");
	cudaMemset(d_img3DTemp, 0, totalSize1*sizeof(float));
	cudacopyhosttoarray(d_ArrayTemp, channelDesc, h_img2, sx2, sy2, sz2);
	BindTexture(d_ArrayTemp, channelDesc);
	CopyTranMatrix(iTmx, NDIM * sizeof(float));
	affineTransform(d_img3DTemp, sx1, sy1, sz1, sx2, sy2, sz2);
	UnbindTexture();
	cudaMemcpy(h_reg, d_img3DTemp, totalSize1 * sizeof(float), cudaMemcpyDeviceToHost);
	cudaFreeArray(d_ArrayTemp);
	cudaFree(d_img3DTemp);
	return 0;
}

int atrans3dgpu_16bit(unsigned short *h_reg, float *iTmx, unsigned short *h_img2, unsigned int *imSize1, unsigned int *imSize2, int deviceNum){
	//image size
	long long int sx1 = imSize1[0], sy1 = imSize1[1], sz1 = imSize1[2];
	long long int sx2 = imSize2[0], sy2 = imSize2[1], sz2 = imSize2[2];
	// total pixel count for each images
	long long int totalSize1 = sx1*sy1*sz1;
	long long int totalSize2 = sx2*sy2*sz2;
	// GPU device
	unsigned short *d_img3D16;
	cudaSetDevice(deviceNum);
	cudaMalloc((void **)&d_img3D16, totalSize1 *sizeof(unsigned short));
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<unsigned short>();
	cudaArray *d_Array;
	cudaMalloc3DArray(&d_Array, &channelDesc, make_cudaExtent(sx2, sy2, sz2));
	cudaThreadSynchronize();
	cudaCheckErrors("****GPU array memory allocating fails... GPU out of memory !!!!*****\n");
	cudaMemset(d_img3D16, 0, totalSize1*sizeof(unsigned short));
	cudacopyhosttoarray(d_Array, channelDesc, h_img2, sx2, sy2, sz2);
	BindTexture16(d_Array, channelDesc);
	CopyTranMatrix(iTmx, NDIM * sizeof(float));
	affineTransform(d_img3D16, sx1, sy1, sz1, sx2, sy2, sz2);
	UnbindTexture16();
	cudaMemcpy(h_reg, d_img3D16, totalSize1 * sizeof(unsigned short), cudaMemcpyDeviceToHost);
	cudaFreeArray(d_Array);
	cudaFree(d_img3D16);
	return 0;
}

///// 2D registration
int reg2d(float *h_reg, float *iTmx, float *h_img1, float *h_img2, unsigned int *imSize1, unsigned int *imSize2, int regChoice,
	bool flagTmx, float FTOL, int itLimit, int deviceNum, int gpuMemMode, bool verbose, float *records) {
	// **** 3D image registration: capable with phasor registraton and affine registration  ***
	/*
	*** registration choice: regChoice
	0: no phasor or affine registration; if flagTmx is true, transform h_img2 based on input matrix;
	1: phasor registraion (pixel-level translation only);
	2: affine registration (with or without input matrix);
	*
	*** flagTmx: only if regChoice == 0, 2
	true: use iTmx as input matrix;
	false: default;
	*
	*** gpuMemMode
	0: All on CPU. // need to add this option in the future
	1: sufficient GPU memory;
	*
	*** records: 11 element array
	[0]: actual gpu memory mode
	[1] -[3]: initial ZNCC (zero-normalized cross-correlation, negtive of the cost function), intermediate ZNCC, optimized ZNCC;
	[4] -[7]: single sub iteration time (in ms), total number of sub iterations, iteralation time (in s), whole registration time (in s);
	[8] -[10]: initial GPU memory, before registration, after processing ( all in MB), if use gpu
	*/

	// ************get basic input images information ******************	
	//image size
	long long int imx1, imy1, imx2, imy2;
	imx1 = imSize1[0]; imy1 = imSize1[1]; 
	imx2 = imSize2[0]; imy2 = imSize2[1];
	// total pixel count for each image
	long long int totalSize1 = imx1*imy1;
	long long int totalSize2 = imx2*imy2;
	long long int totalSizeMax = (totalSize1 > totalSize2) ? totalSize1 : totalSize2;

	// ****************** Processing Starts*****************//
	// variables for memory and time cost records
	clock_t start, time1, time2, time3, end;
	size_t totalMem = 0;
	size_t freeMem = 0;
	time1 = time2 = time3 = end = 0;
	start = clock();
	if (gpuMemMode == 1 ) {
		cudaSetDevice(deviceNum);
		cudaMemGetInfo(&freeMem, &totalMem);
		records[8] = (float)freeMem / 1048576.0f;
		if (verbose) {
			printf("...GPU free memory before registration is %.0f MB\n", (float)freeMem / 1048576.0f);
		}
	}
	records[0] = gpuMemMode;
	int affMethod = 1;
	long long int shiftXY[2];
	float *d_imgT = NULL, *d_imgS = NULL;
	switch (gpuMemMode) {
	case 0:
		switch (regChoice) {
		case 0:
			break;
		case 1:
			break;
		case 2:
			break;
		default:
			printf("\n ****Wrong registration choice is setup, no registraiton performed !!! **** \n");
			return 1;
		}
		printf("\n **** 2D CPU registration is currently not supported !!! **** \n");
		break;
	case 1:
		cudaMemGetInfo(&freeMem, &totalMem);
		records[9] = (float)freeMem / 1048576.0f;
		switch (regChoice) {
		case 0:
			if (flagTmx) {
				affMethod = 0;
				(void)reg2d_affine1(h_reg, iTmx, h_img1, h_img2, imx1, imy1, imx2, imy2, affMethod, flagTmx, FTOL, itLimit, records);
			}
			break;
		case 1:
			if ((imx1 != imx2) || (imy1 != imy2)) {
				printf("\n ****Image size of the 2D images is not matched, processing stop !!! **** \n");
				return 1;
			}
			cudaMalloc((void **)&d_imgT, totalSize1 * sizeof(float));
			cudaMalloc((void **)&d_imgS, totalSize1 * sizeof(float));
			cudaMemcpy(d_imgT, h_img1, totalSize1 * sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(d_imgS, h_img2, totalSize1 * sizeof(float), cudaMemcpyHostToDevice);
			(void)reg2d_phasor1(&shiftXY[0], d_imgT, d_imgS, imx1, imy1);
			imshiftgpu(d_imgT, d_imgS, imx1, imy1, 1, -shiftXY[0], -shiftXY[1], 0);
			cudaMemcpy(h_reg, d_imgT, totalSize1 * sizeof(float), cudaMemcpyDeviceToHost);
			iTmx[0] = 1; iTmx[1] = 0; iTmx[2] = shiftXY[0];
			iTmx[3] = 0; iTmx[4] = 1; iTmx[5] = shiftXY[1];
			break;
		case 2:
			(void)reg2d_affine1(h_reg, iTmx, h_img1, h_img2, imx1, imy1, imx2, imy2, affMethod, flagTmx, FTOL, itLimit, records);
			break;
		default:
			printf("\n ****Wrong registration choice is setup, no registraiton performed !!! **** \n");
			return 1;
		}
		break;
	default:
		printf("\n****Wrong gpuMemMode setup, no deconvolution performed !!! ****\n");
		return 1;
	}
	//
	if (gpuMemMode == 1) {
		cudaMemGetInfo(&freeMem, &totalMem);
		records[10] = (float)freeMem / 1048576.0f;
	}
	end = clock();
	records[7] = (float)(end - start) / CLOCKS_PER_SEC;
	if (verbose) {
		printf("Total time cost for whole processing is %2.3f s\n", records[7]);
	}
	return 0;
}

///// 3D registration
bool checkmatrix(float *m, long long int sx, long long int sy, long long int sz) {
	// check if affine matrix is reasonable
	bool mMatrix = true;
	float scaleLow = 0.5, scaleUp = 1.4, scaleSumLow = 2, scaleSumUp = 4, shiftRatio = 0.8;
	if (m[0]<scaleLow || m[0]>scaleUp || m[5]<scaleLow || m[5]>scaleUp || m[10]<scaleLow || m[10]>scaleUp) {
		mMatrix = false;
	}
	if ((m[0] + m[5] + m[10]) < scaleSumLow || (m[0] + m[5] + m[10]) > scaleSumUp) {
		mMatrix = false;
	}
	if (abs(m[3])>shiftRatio * sx || abs(m[7])>shiftRatio * sy || abs(m[11])>shiftRatio * sz) {
		mMatrix = false;
	}
	// ... more checking
	return mMatrix;
}

int reg3d(float *h_reg, float *iTmx, float *h_img1, float *h_img2, unsigned int *imSize1, unsigned int *imSize2, int regChoice, int affMethod,
	bool flagTmx, float FTOL, int itLimit, int deviceNum, int gpuMemMode, bool verbose, float *records) {
	// **** 3D image registration: capable with phasor registraton and affine registration  ***
	/*
	*** registration choice: regChoice
	0: no phasor or affine registration; if flagTmx is true, transform h_img2 based on input matrix;
	1: phasor registraion (pixel-level translation only);
	2: affine registration (with or without input matrix);
	3: phasor registration --> affine registration (input matrix disabled);
	4: 2D MIP registration --> affine registration (input matrix disabled);
	*
	*** affine registration method: affMethod, only if regChoice == 2, 3, 4
	0: no registration; if inputTmx is true, transform d_img2 based on input matrix;
	1: translation only;
	2: rigid body;
	3: 7 degrees of freedom (translation, rotation, scaling equally in 3 dimensions)
	4: 9 degrees of freedom(translation, rotation, scaling);
	5: 12 degrees of freedom;
	6: rigid body first, then do 12 degrees of freedom;
	7: 3 DOF --> 6 DOF --> 9 DOF --> 12 DOF
	*
	*** flagTmx: only if regChoice == 0, 2
	true: use iTmx as input matrix;
	false: default;
	*
	*** gpuMemMode  
	-1: Automatically set memory mode; 
	0: All on CPU. // need to add this option in the future
	1: sufficient GPU memory; 
	2: GPU memory optimized; 
	*
	*** records: 11 element array
	[0]: actual gpu memory mode
	[1] -[3]: initial ZNCC (zero-normalized cross-correlation, negtive of the cost function), intermediate ZNCC, optimized ZNCC;
	[4] -[7]: single sub iteration time (in ms), total number of sub iterations, iteralation time (in s), whole registration time (in s);
	[8] -[10]: initial GPU memory, before registration, after processing ( all in MB), if use gpu
	*/
	// ************get basic input images information ******************	
	//image size
	long long int imx1, imy1, imz1, imx2, imy2, imz2;
	imx1 = imSize1[0]; imy1 = imSize1[1]; imz1 = imSize1[2];
	imx2 = imSize2[0]; imy2 = imSize2[1]; imz2 = imSize2[2];
	// total pixel count for each image
	long long int totalSize1 = imx1*imy1*imz1;
	long long int totalSize2 = imx2*imy2*imz2;
	long long int totalSizeMax = (totalSize1 > totalSize2) ? totalSize1 : totalSize2;

	// ****************** Processing Starts*****************//
	// variables for memory and time cost records
	clock_t start, time1, time2, time3, end;
	size_t totalMem = 0;
	size_t freeMem = 0;
	time1 = time2 = time3 = end = 0;
	start = clock();
	if (gpuMemMode != 0) {
		cudaSetDevice(deviceNum);
		cudaMemGetInfo(&freeMem, &totalMem);
		records[8] = (float)freeMem / 1048576.0f;
		if (verbose) {
			printf("\t... GPU free memory before registration is %.0f MB\n", (float)freeMem / 1048576.0f);
		}	
	}
	// ***** Set GPU memory mode based on images size and available GPU memory ****
	// gpuMemMode --> Unified memory in next version???
	// -1: Automatically set memory mode based on calculations; 
	// 0: all in CPU; 1: sufficient GPU memory; 2: GPU memory optimized. 
	if (gpuMemMode == -1) { //Automatically set memory mode based on calculations.
		cudaMemGetInfo(&freeMem, &totalMem);
		if ((regChoice == 0)||(regChoice == 2)|| (regChoice == 4)) {
			if (freeMem > (4 * totalSizeMax + 4 * imx1*imy1) * sizeof(float)) {
				gpuMemMode = 1;
				if (verbose) {
					printf("\t... GPU memory is sufficient, processing in efficient mode !!!\n");
				}
			}
			else if (freeMem > (2 * totalSizeMax + 4 * imx1*imy1) * sizeof(float)) {
				gpuMemMode = 2;
				if (verbose) {
					printf("\t... GPU memory is optimized, processing in memory saved mode !!!\n");
				}
			}
			else { // all processing in CPU
				gpuMemMode = 0;
				if (verbose) {
					printf("\t... GPU memory is not enough, processing in CPU mode!!!\n");
				}
			}
		}
		else if ((regChoice == 1) || (regChoice == 3)) {
			if (freeMem > (5 * totalSizeMax + 4 * imx1*imy1) * sizeof(float)) {
				gpuMemMode = 1;
				if (verbose) {
					printf("\t... GPU memory is sufficient, processing in efficient mode !!!\n");
				}
			}
			else if (freeMem > (4 * totalSizeMax + 4 * imx1*imy1) * sizeof(float)) {
				gpuMemMode = 2;
				if (verbose) {
					printf("\t... GPU memory is optimized, processing in memory saved mode !!!\n");
				}
			}
			else { // all processing in CPU
				gpuMemMode = 0;
				if (verbose) {
					printf("\t... GPU memory is not enough, processing in CPU mode!!!\n");
				}
			}
		}
	}
	records[0] = gpuMemMode;

	float *h_imgT = NULL, *h_imgS = NULL;
	float *d_imgT = NULL, *d_imgS = NULL, *d_reg = NULL;
	long long int shiftXYZ[3], shiftXY[2], shiftZX[2];
	float *d_imgTemp1 = NULL, *d_imgTemp2 = NULL;
	float *d_imgTemp3 = NULL, *d_imgTemp4 = NULL;
	unsigned int im2DSize[2];
	long long int totalSize2DMax;
	switch (gpuMemMode) {
	case 0: // CPU calculation
		printf("\n ****CPU registraion function is under developing **** \n");
		return -1;
		break;
	case 1:// efficient GPU calculation
		time1 = clock();
		// allocate memory
		//cudaMalloc((void **)&d_reg, totalSize1 * sizeof(float));
		cudaMalloc((void **)&d_imgT, totalSizeMax * sizeof(float));
		cudaMalloc((void **)&d_imgS, totalSize1 * sizeof(float));
		cudaCheckErrors("****Memory allocating fails... GPU out of memory !!!!*****\n");
		if ((imx1 == imx2) && (imy1 == imy2) && (imz1 == imz2))
			cudaMemcpy(d_imgS, h_img2, totalSize2 * sizeof(float), cudaMemcpyHostToDevice);
		else {
			cudaMemcpy(d_imgT, h_img2, totalSize2 * sizeof(float), cudaMemcpyHostToDevice);
			alignsize3Dgpu(d_imgS, d_imgT, imz1, imy1, imx1, imz2, imy2, imx2);
		}
		cudaMemcpy(d_imgT, h_img1, totalSize1 * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemGetInfo(&freeMem, &totalMem);
		records[9] = (float)freeMem / 1048576.0f;
		switch (regChoice) {
		case 0:
			affMethod = 0;
			cudaMalloc((void **)&d_reg, totalSize1 * sizeof(float));
			(void)reg3d_affine1(d_reg, iTmx, d_imgT, d_imgS, imx1, imy1, imz1, affMethod, flagTmx, FTOL, itLimit, verbose, records);
			cudaMemcpy(h_reg, d_reg, totalSize1 * sizeof(float), cudaMemcpyDeviceToHost);
			cudaFree(d_reg);
			break;
		case 1:
			(void)reg3d_phasor1(&shiftXYZ[0], d_imgT, d_imgS, imx1, imy1, imz1);
			imshiftgpu(d_imgT, d_imgS, imx1, imy1, imz1, -shiftXYZ[0], -shiftXYZ[1], -shiftXYZ[2]);
			cudaMemcpy(h_reg, d_imgT, totalSize1 * sizeof(float), cudaMemcpyDeviceToHost);
			for (int j = 0; j < NDIM; j++) iTmx[j] = 0;
			iTmx[0] = iTmx[5] = iTmx[10] = 1;
			iTmx[3] = shiftXYZ[0]; 
			iTmx[7] = shiftXYZ[1]; 
			iTmx[11] = shiftXYZ[2];
			break;
		case 2:
			cudaMalloc((void **)&d_reg, totalSize1 * sizeof(float));
			(void)reg3d_affine1(d_reg, iTmx, d_imgT, d_imgS, imx1, imy1, imz1, affMethod, flagTmx, FTOL, itLimit, verbose, records);
			cudaMemcpy(h_reg, d_reg, totalSize1 * sizeof(float), cudaMemcpyDeviceToHost);
			cudaFree(d_reg);
			break;
		case 3:
			(void)reg3d_phasor1(&shiftXYZ[0], d_imgT, d_imgS, imx1, imy1, imz1);
			for (int j = 0; j < NDIM; j++) iTmx[j] = 0;
			iTmx[0] = iTmx[5] = iTmx[10] = 1;
			iTmx[3] = shiftXYZ[0];
			iTmx[7] = shiftXYZ[1];
			iTmx[11] = shiftXYZ[2];
			cudaMalloc((void **)&d_reg, totalSize1 * sizeof(float));
			flagTmx = true;
			(void)reg3d_affine1(d_reg, iTmx, d_imgT, d_imgS, imx1, imy1, imz1, affMethod, flagTmx, FTOL, itLimit, verbose, records);
			cudaMemcpy(h_reg, d_reg, totalSize1 * sizeof(float), cudaMemcpyDeviceToHost);
			cudaFree(d_reg);
			break;
		case 4:
			printf("\t... 2D MIP registration ... \n");
			totalSize2DMax = ((imx1 * imy1) > (imz1 * imx1)) ? (imx1 * imy1) : (imz1 * imx1);
			cudaMalloc((void **)&d_imgTemp3, totalSize2DMax * sizeof(float));
			cudaMalloc((void **)&d_imgTemp4, totalSize2DMax * sizeof(float));
			maxprojection(d_imgTemp3, d_imgT, imx1, imy1, imz1, 1);
			maxprojection(d_imgTemp4, d_imgS, imx1, imy1, imz1, 1);
			(void)reg2d_phasor1(&shiftXY[0], d_imgTemp3, d_imgTemp4, imx1, imy1);
			maxprojection(d_imgTemp3, d_imgT, imx1, imy1, imz1, 2);
			maxprojection(d_imgTemp4, d_imgS, imx1, imy1, imz1, 2);
			(void)reg2d_phasor1(&shiftZX[0], d_imgTemp3, d_imgTemp4, imz1, imx1);
			for (int j = 0; j < NDIM; j++) iTmx[j] = 0;
			iTmx[0] = iTmx[5] = iTmx[10] = 1;
			iTmx[3] = (shiftXY[0]+shiftZX[1])/2;
			iTmx[7] = shiftXY[1];
			iTmx[11] = shiftZX[0];
			cudaFree(d_imgTemp3); cudaFree(d_imgTemp4);
			printf("\t... 2D MIP registration completed. \n");
			printf("\t... 3D registration ... \n");
			cudaMalloc((void **)&d_reg, totalSize1 * sizeof(float));
			flagTmx = true;
			(void)reg3d_affine1(d_reg, iTmx, d_imgT, d_imgS, imx1, imy1, imz1, affMethod, flagTmx, FTOL, itLimit, verbose, records);
			cudaMemcpy(h_reg, d_reg, totalSize1 * sizeof(float), cudaMemcpyDeviceToHost);
			cudaFree(d_reg);
			break;
		default:
			printf("\n*** Wrong registration choice is setup, no registraiton performed !!! **** \n");
			return 1;
		}
		cudaFree(d_imgT);
		cudaFree(d_imgS);
		time2 = clock();
		break;
	case 2:
		time1 = clock();
		// allocate memory
		h_imgS = (float *)malloc(totalSize1 * sizeof(float));
		if ((imx1 == imx2) && (imy1 == imy2) && (imz1 == imz2))
			memcpy(h_imgS, h_img2, totalSize1 * sizeof(float));
		else {
			cudaMalloc((void **)&d_imgS, totalSize2 * sizeof(float));
			cudaMalloc((void **)&d_imgT, totalSize1 * sizeof(float));
			cudaCheckErrors("****Memory allocating fails... GPU out of memory !!!!*****\n");
			cudaMemcpy(d_imgS, h_img2, totalSize2 * sizeof(float), cudaMemcpyHostToDevice);
			alignsize3Dgpu(d_imgT, d_imgS, imz1, imy1, imx1, imz2, imy2, imx2);
			cudaMemcpy(h_imgS, d_imgT, totalSize1 * sizeof(float), cudaMemcpyDeviceToHost);
			cudaFree(d_imgS);
			cudaFree(d_imgT);
		}
		cudaMemGetInfo(&freeMem, &totalMem);
		records[9] = (float)freeMem / 1048576.0f;
		switch (regChoice) {
		case 0:
			affMethod = 0;
			cudaMalloc((void **)&d_reg, totalSize1 * sizeof(float));
			(void)reg3d_affine2(d_reg, iTmx, h_img1, h_imgS, imx1, imy1, imz1, affMethod, flagTmx, FTOL, itLimit, verbose, records);
			cudaMemcpy(h_reg, d_reg, totalSize1 * sizeof(float), cudaMemcpyDeviceToHost);
			cudaFree(d_reg);
			break;
		case 1:
			(void)reg3d_phasor2(&shiftXYZ[0], h_img1, h_imgS, imx1, imy1, imz1);
			cudaMalloc((void **)&d_imgTemp1, totalSize1 * sizeof(float));
			cudaMalloc((void **)&d_imgTemp2, totalSize1 * sizeof(float));
			cudaMemcpy(d_imgTemp1, h_imgS, totalSize2 * sizeof(float), cudaMemcpyHostToDevice);
			imshiftgpu(d_imgTemp2, d_imgTemp1, imx1, imy1, imz1, -shiftXYZ[0], -shiftXYZ[1], -shiftXYZ[2]);
			cudaMemcpy(h_reg, d_imgTemp2, totalSize1 * sizeof(float), cudaMemcpyDeviceToHost);
			for (int j = 0; j < NDIM; j++) iTmx[j] = 0;
			iTmx[0] = iTmx[5] = iTmx[10] = 1;
			iTmx[3] = shiftXYZ[0];
			iTmx[7] = shiftXYZ[1];
			iTmx[11] = shiftXYZ[2];
			cudaFree(d_imgTemp1);
			cudaFree(d_imgTemp2);
			break;
		case 2:
			cudaMalloc((void **)&d_reg, totalSize1 * sizeof(float));
			(void)reg3d_affine2(d_reg, iTmx, h_img1, h_imgS, imx1, imy1, imz1, affMethod, flagTmx, FTOL, itLimit, verbose, records);
			cudaMemcpy(h_reg, d_reg, totalSize1 * sizeof(float), cudaMemcpyDeviceToHost);
			cudaFree(d_reg);
			break;
		case 3:
			(void)reg3d_phasor2(&shiftXYZ[0], h_img1, h_imgS, imx1, imy1, imz1);
			for (int j = 0; j < NDIM; j++) iTmx[j] = 0;
			iTmx[0] = iTmx[5] = iTmx[10] = 1;
			iTmx[3] = shiftXYZ[0];
			iTmx[7] = shiftXYZ[1];
			iTmx[11] = shiftXYZ[2];
			cudaMalloc((void **)&d_reg, totalSize1 * sizeof(float));
			flagTmx = true;
			(void)reg3d_affine2(d_reg, iTmx, h_img1, h_imgS, imx1, imy1, imz1, affMethod, flagTmx, FTOL, itLimit, verbose, records);
			cudaMemcpy(h_reg, d_reg, totalSize1 * sizeof(float), cudaMemcpyDeviceToHost);
			cudaFree(d_reg);
			break;
		case 4:
			printf("\n ****2D MIP registration --> affine registraion function is under developing **** \n");
			return -1;
			break;
		default:
			printf("\n ****Wrong registration choice is setup, no registraiton performed !!! **** \n");
			return 1;
		}
		free(h_imgS);
		time2 = clock();
		break;
	default:
		printf("\n****Wrong gpuMemMode setup, no deconvolution performed !!! ****\n");
		return 1;
	}
	//
	if (gpuMemMode > 0) {
		cudaMemGetInfo(&freeMem, &totalMem);
		records[10] = (float)freeMem / 1048576.0f;
	}
	end = clock();
	records[7] = (float)(end - start) / CLOCKS_PER_SEC;
	if (verbose) {
		printf("\t... registration done !!! \n");
		//printf("\t...Time cost is %2.3f s\n", records[7]);
	}
	return 0;
}

int reg_3dgpu(float *h_reg, float *iTmx, float *h_img1, float *h_img2, unsigned int *imSize1, unsigned int *imSize2, int affMethod,
	int inputTmx, float FTOL, int itLimit, int subBgTrigger, int deviceNum, float *regRecords) {
	// **** 3D GPU affine registration: based on function reg3d  ***
	/*
	*** affine registration method
	0: no registration; if inputTmx is true, transform d_img2 based on input matrix;
	1: translation only;
	2: rigid body;
	3: 7 degrees of freedom (translation, rotation, scaling equally in 3 dimensions)
	4: 9 degrees of freedom(translation, rotation, scaling);
	5: 12 degrees of freedom;
	6: rigid body first, then do 12 degrees of freedom;
	7: 3 DOF --> 6 DOF --> 9 DOF --> 12 DOF
	*
	*** flagTmx
	true: use iTmx as input matrix;
	false: default;

	*** regRecords: 11 element array
	[0]: actual gpu memory mode
	[1] -[3]: initial ZNCC (zero-normalized cross-correlation, negtive of the cost function), intermediate ZNCC, optimized ZNCC;
	[4] -[7]: single sub iteration time (in ms), total number of sub iterations, iteralation time (in s), whole registration time (in s);
	[8] -[10]: initial GPU memory, before registration, after processing ( all in MB);
	*/

	// subBgTrigger: no longer used 
	int regChoice = 4; // Do 2D regitration first
	bool flagTmx = false;
	if (inputTmx == 1) {
		flagTmx = true;
		regChoice = 2; // if use input matrix, do not perform 2D registration
	}
	int gpuMemMode = 1;
	bool verbose = false;
	int regStatus = reg3d(h_reg, iTmx, h_img1, h_img2, imSize1, imSize2, regChoice, affMethod,
		flagTmx, FTOL, itLimit, deviceNum, gpuMemMode, verbose, regRecords);
	bool mStatus = checkmatrix(iTmx, imSize1[0], imSize1[1], imSize1[2]);
	if (!mStatus) {
		regChoice = 2;
		regStatus = reg3d(h_reg, iTmx, h_img1, h_img2, imSize1, imSize2, regChoice, affMethod,
			flagTmx, FTOL, itLimit, deviceNum, gpuMemMode, verbose, regRecords);
	}
	return regStatus;
}

#undef blockSize
#undef blockSize2Dx
#undef blockSize2Dy
#undef blockSize3Dx
#undef blockSize3Dy
#undef blockSize3Dz
#undef SMALLVALUE
#undef NDIM
