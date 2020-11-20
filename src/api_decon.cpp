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
#include <cuda_runtime.h>
#include <cufft.h>
//#include <cufftw.h> // ** cuFFT also comes with CPU-version FFTW, but seems not to work when image size is large.
#include "fftw3.h"
#include <memory>
#include "device_launch_parameters.h"

#define PROJECT_EXPORTS // To export API functions
#include "libapi.h"
#include "apifunc_internal.h"
#define blockSize 1024
#define blockSize2Dx 32
#define blockSize2Dy 32
#define blockSize3Dx 16
#define blockSize3Dy 8
#define blockSize3Dz 8
#define NDIM 12
#define SMALLVALUE 0.01
#undef PROJECT_EXPORTS

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


int decon_singleview(float *h_decon, float *h_img, unsigned int *imSize, float *h_psf, unsigned int *psfSize, bool flagConstInitial,
	int itNumForDecon, int deviceNum, int gpuMemMode, bool verbose, float *deconRecords, bool flagUnmatch, float *h_psf_bp) {
	// gpuMemMode --> -1: Automatically set memory mode; 0: All on CPU; 1: sufficient GPU memory; 2: GPU memory optimized.
	//deconRecords: 10 elements
	//[0]:  the actual GPU memory mode used;
	//[1] -[5]: initial GPU memory, after variables partially allocated, during processing, after processing, after variables released ( all in MB);
	//[6] -[9]: initializing time, prepocessing time, decon time, total time;	
	// image size
	long long int
		imx, imy, imz;
	imx = imSize[0], imy = imSize[1], imz = imSize[2];
	// PSF size
	long long int
		PSFx, PSFy, PSFz;
	PSFx = psfSize[0], PSFy = psfSize[1], PSFz = psfSize[2];
	// FFT size
	long long int
		FFTx, FFTy, FFTz;

	FFTx = snapTransformSize(imx);// snapTransformSize(imx + PSFx - 1);
	FFTy = snapTransformSize(imy);// snapTransformSize(imy + PSFy - 1);
	FFTz = snapTransformSize(imz);// snapTransformSize(imz + PSFz - 1);

	printf("Image information:\n");
	printf("...Image size %d x %d x %d\n  ", imx, imy, imz);
	printf("...PSF size %d x %d x %d\n  ", PSFx, PSFy, PSFz);
	printf("...FFT size %d x %d x %d\n  ", FFTx, FFTy, FFTz);
	printf("...Output Image size %d x %d x %d \n   ", imSize[0], imSize[1], imSize[2]);

	// total pixel count for each images
	long long int totalSize = imx*imy*imz; // in floating format
	long long int totalSizePSF = PSFx*PSFy*PSFz; // in floating format
	long long int totalSizeFFT = FFTx*FFTy*FFTz; // in floating format
	long long int totalSizeSpectrum = FFTx * FFTy*(FFTz / 2 + 1); // in complex floating format
	long long int totalSizeMax = totalSizeSpectrum * 2; // in floating format

	// ****************** Processing Starts***************** //
	// variables for memory and time cost records
	clock_t start, time1, time2, time3, end;
	size_t totalMem = 0;
	size_t freeMem = 0;
	time1 = time2 = time3 = end = 0;
	start = clock();
	if (gpuMemMode != 0) {
		cudaSetDevice(deviceNum);
		cudaMemGetInfo(&freeMem, &totalMem);
		printf("...GPU free memory(at beginning) is %.0f MBites\n", (float)freeMem / 1048576.0f);
		deconRecords[1] = (float)freeMem / 1048576.0f;
	}
	// ***** Set GPU memory mode based on images size and available GPU memory ****
	// gpuMemMode --> Unified memory in next version???
	// -1: Automatically set memory mode based on calculations; 
	// 0: all in CPU; 1: sufficient GPU memory; 2: GPU memory optimized. 
	if (gpuMemMode == -1) { //Automatically set memory mode based on calculations.
		// Test to create FFT plans to estimate GPU memory
		cufftHandle
			fftPlanFwd,
			fftPlanInv;
		cufftPlan3d(&fftPlanFwd, FFTx, FFTy, FFTz, CUFFT_R2C);
		cufftPlan3d(&fftPlanInv, FFTx, FFTy, FFTz, CUFFT_C2R);
		cudaCheckErrors("**** GPU out of memory during memory emstimating!!!!*****");
		cudaMemGetInfo(&freeMem, &totalMem);
		if (freeMem > 6 * totalSizeMax * sizeof(float)) { // 6 more GPU variables
			gpuMemMode = 1;
			printf("\n GPU memory is sufficient, processing in efficient mode !!!\n");
		}
		else if (freeMem > 2 * totalSizeMax * sizeof(float)) {// 2 more GPU variables
			gpuMemMode = 2;
			printf("\n GPU memory is optimized, processing in memory saved mode !!!\n");
		}
		else { // all processing in CPU
			gpuMemMode = 0;
			printf("\n GPU memory is not enough, processing in CPU mode!!!\n");
		}
		// destroy plans
		cufftDestroy(fftPlanFwd);
		cufftDestroy(fftPlanInv);
	}
	deconRecords[0] = gpuMemMode;

	float
		*h_StackA,
		*h_StackE,
		*d_StackA,
		*d_StackE;
	fComplex
		*h_PSFSpectrum,
		*h_FlippedPSFSpectrum,
		*d_PSFSpectrum,
		*d_FlippedPSFSpectrum,
		*d_StackESpectrum;

	switch (gpuMemMode) {
	case 0:
		// CPU deconvolution
		time1 = clock();
		// allocate memory
		h_StackA = (float *)malloc(totalSizeMax * sizeof(float));
		h_StackE = (float *)malloc(totalSizeMax * sizeof(float));
		h_PSFSpectrum = (fComplex *)malloc(totalSizeSpectrum * sizeof(fftwf_complex));
		h_FlippedPSFSpectrum = (fComplex *)malloc(totalSizeSpectrum * sizeof(fftwf_complex));

		// *** PSF Preparation
		// OTF 
		changestorageordercpu(h_StackA, h_psf, PSFx, PSFy, PSFz, 1);
		genOTFcpu((fftwf_complex *)h_PSFSpectrum, h_StackA, FFTx, FFTy, FFTz, PSFx, PSFy, PSFz, true);
		// OTF_bp
		if (flagUnmatch) {
			changestorageordercpu(h_StackA, h_psf_bp, PSFx, PSFy, PSFz, 1);
			genOTFcpu((fftwf_complex *)h_FlippedPSFSpectrum, h_StackA, FFTx, FFTy, FFTz, PSFx, PSFy, PSFz, true);
		}
		else { // traditional backprojector matched PSF 
			flipcpu(h_StackE, h_StackA, PSFx, PSFy, PSFz); // flip PSF
			genOTFcpu((fftwf_complex *)h_FlippedPSFSpectrum, h_StackE, FFTx, FFTy, FFTz, PSFx, PSFy, PSFz, true);
		}

		// *** image  Preparation
		if ((imx < FFTx) || (imy < FFTy) || (imz < FFTz)) {
			changestorageordercpu(h_StackE, h_img, imx, imy, imz, 1);
			padstackcpu(h_StackA, h_StackE, FFTx, FFTy, FFTz, imx, imy, imz);
		}
		else {
			changestorageordercpu(h_StackA, h_img, imx, imy, imz, 1);
		}

		// *** deconvolution ****
		memset(h_StackE, 0, totalSizeFFT * sizeof(float));
		decon_singleview_OTF0(h_StackE, h_StackA, (fftwf_complex *)h_PSFSpectrum,
			(fftwf_complex *)h_FlippedPSFSpectrum, FFTx, FFTy, FFTz, itNumForDecon, flagConstInitial);
		if ((imx < FFTx) || (imy < FFTy) || (imz < FFTz)) {
			cropcpu(h_StackA, h_StackE, imx, imy, imz, FFTx, FFTy, FFTz);
			changestorageordercpu(h_decon, h_StackA, imx, imy, imz, -1);
		}
		else {
			changestorageordercpu(h_decon, h_StackE, imx, imy, imz, -1);
		}
		//printf("...Deconvolution completed ! ! !\n")

		time3 = clock();
		// release variables
		free(h_StackA); free(h_StackE);
		free(h_PSFSpectrum);
		free(h_FlippedPSFSpectrum);
		break;
	case 1:// efficient GPU calculation
		time1 = clock();
		// allocate memory
		cudaMalloc((void **)&d_StackA, totalSizeMax * sizeof(float));
		cudaMalloc((void **)&d_StackE, totalSizeMax * sizeof(float));
		cudaMalloc((void **)&d_PSFSpectrum, totalSizeSpectrum * sizeof(fComplex));
		cudaMalloc((void **)&d_FlippedPSFSpectrum, totalSizeSpectrum * sizeof(fComplex));
		cudaMalloc((void **)&d_StackESpectrum, totalSizeSpectrum * sizeof(fComplex));
		//check GPU status
		cudaCheckErrors("****Memory allocating fails... GPU out of memory !!!!*****");
		cudaMemGetInfo(&freeMem, &totalMem);
		printf("...GPU free memory(after mallocing) is %.0f MBites\n", (float)freeMem / 1048576.0f);
		deconRecords[2] = (float)freeMem / 1048576.0f;
		// *** PSF Preparation
		// OTF 
		cudaMemcpy(d_StackE, h_psf, totalSizePSF * sizeof(float), cudaMemcpyHostToDevice);
		changestorageordergpu(d_StackA, d_StackE, PSFx, PSFy, PSFz, 1); //1: change tiff storage order to C storage order
		genOTFgpu(d_PSFSpectrum, d_StackA, FFTx, FFTy, FFTz, PSFx, PSFy, PSFz, true);
		// OTF_bp
		if (flagUnmatch){
			cudaMemcpy(d_StackE, h_psf_bp, totalSizePSF * sizeof(float), cudaMemcpyHostToDevice);
			changestorageordergpu(d_StackA, d_StackE, PSFx, PSFy, PSFz, 1); //1: change tiff storage order to C storage order
			genOTFgpu(d_FlippedPSFSpectrum, d_StackA, FFTx, FFTy, FFTz, PSFx, PSFy, PSFz, true);
		}
		else { // traditional backprojector matched PSF 
			flipgpu(d_StackE, d_StackA, PSFx, PSFy, PSFz); // flip PSF
			genOTFgpu(d_FlippedPSFSpectrum, d_StackE, FFTx, FFTy, FFTz, PSFx, PSFy, PSFz, true); //PSF already normalized
		}
		// *** image Preparation
		if ((imx < FFTx) || (imy < FFTy) || (imz < FFTz)) {
			cudaMemcpy(d_StackA, h_img, totalSize * sizeof(float), cudaMemcpyHostToDevice);
			changestorageordergpu(d_StackE, d_StackA, imx, imy, imz, 1); //1: change tiff storage order to C storage order
			padstackgpu(d_StackA, d_StackE, FFTx, FFTy, FFTz, imx, imy, imz);//
		}
		else {
			cudaMemcpy(d_StackE, h_img, totalSize * sizeof(float), cudaMemcpyHostToDevice);
			changestorageordergpu(d_StackA, d_StackE, FFTx, FFTy, FFTz, 1); //1: change tiff storage order to C storage order
		}
		// *** deconvolution ****
		cudaMemset(d_StackE, 0, totalSizeFFT * sizeof(float));
		decon_singleview_OTF1(d_StackE, d_StackA, d_PSFSpectrum, d_FlippedPSFSpectrum, FFTx, FFTy, FFTz, itNumForDecon, flagConstInitial);

		// transfer data back to CPU RAM
		if ((imx < FFTx) || (imy < FFTy) || (imz < FFTz)) {
			cropgpu(d_StackA, d_StackE, imx, imy, imz, FFTx, FFTy, FFTz);//
			changestorageordergpu(d_StackE, d_StackA, imx, imy, imz, -1); //-1: change C storage order to tiff storage order
			cudaMemcpy(h_decon, d_StackE, totalSize * sizeof(float), cudaMemcpyDeviceToHost);
		}
		else {
			changestorageordergpu(d_StackA, d_StackE, imx, imy, imz, -1); //-1: change C storage order to tiff storage order
			cudaMemcpy(h_decon, d_StackA, totalSize * sizeof(float), cudaMemcpyDeviceToHost);
		}
		//printf("...Deconvolution completed ! ! !\n")

		time3 = clock();
		cudaMemGetInfo(&freeMem, &totalMem);
		printf("...GPU free memory (after processing) is %.0f MBites\n", (float)freeMem / 1048576.0f);
		deconRecords[4] = (float)freeMem / 1048576.0f;
		// release CUDA variables
		cudaFree(d_StackA); cudaFree(d_StackE); 
		cudaFree(d_PSFSpectrum); cudaFree(d_FlippedPSFSpectrum); cudaFree(d_StackESpectrum);
		break;
	case 2:
		time1 = clock();

		h_PSFSpectrum = (fComplex *)malloc(totalSizeSpectrum * sizeof(fComplex));
		h_FlippedPSFSpectrum = (fComplex *)malloc(totalSizeSpectrum * sizeof(fComplex));
		cudaMalloc((void **)&d_StackA, totalSizeMax * sizeof(float)); // also to store spectrum images
		cudaMalloc((void **)&d_StackE, totalSizeMax * sizeof(float)); // also to store spectrum images
		//check GPU status
		cudaCheckErrors("****Memory allocating fails... GPU out of memory !!!!*****");
		cudaMemGetInfo(&freeMem, &totalMem);
		printf("...GPU free memory(after mallocing) is %.0f MBites\n", (float)freeMem / 1048576.0f);
		deconRecords[2] = (float)freeMem / 1048576.0f;

		// *** PSF Preparation
		// OTF 
		cudaMemcpy(d_StackE, h_psf, totalSizePSF * sizeof(float), cudaMemcpyHostToDevice);
		changestorageordergpu(d_StackA, d_StackE, PSFx, PSFy, PSFz, 1); //1: change tiff storage order to C storage order
		d_PSFSpectrum = (fComplex *)d_StackE;
		genOTFgpu(d_PSFSpectrum, d_StackA, FFTx, FFTy, FFTz, PSFx, PSFy, PSFz, true);
		cudaMemcpy(h_PSFSpectrum, d_PSFSpectrum, totalSizeSpectrum * sizeof(fComplex), cudaMemcpyDeviceToHost);
		// OTF_bp
		if (flagUnmatch) {
			cudaMemcpy(d_StackE, h_psf_bp, totalSizePSF * sizeof(float), cudaMemcpyHostToDevice);
			changestorageordergpu(d_StackA, d_StackE, PSFx, PSFy, PSFz, 1); //1: change tiff storage order to C storage order
			d_FlippedPSFSpectrum = (fComplex *)d_StackE;
			genOTFgpu(d_FlippedPSFSpectrum, d_StackA, FFTx, FFTy, FFTz, PSFx, PSFy, PSFz, true);
		}
		else { // traditional backprojector matched PSF 
			flipgpu(d_StackE, d_StackA, PSFx, PSFy, PSFz); // flip PSF
			d_FlippedPSFSpectrum = (fComplex *)d_StackA;
			genOTFgpu(d_FlippedPSFSpectrum, d_StackE, FFTx, FFTy, FFTz, PSFx, PSFy, PSFz, true); 
		}
		cudaMemcpy(h_FlippedPSFSpectrum, d_FlippedPSFSpectrum, totalSizeSpectrum * sizeof(fComplex), cudaMemcpyDeviceToHost);

		// *** image Preparation
		if ((imx < FFTx) || (imy < FFTy) || (imz < FFTz)) {
			cudaMemcpy(d_StackA, h_img, totalSize * sizeof(float), cudaMemcpyHostToDevice);
			changestorageordergpu(d_StackE, d_StackA, imx, imy, imz, 1); //1: change tiff storage order to C storage order
			padstackgpu(d_StackA, d_StackE, FFTx, FFTy, FFTz, imx, imy, imz);//
		}
		else {
			cudaMemcpy(d_StackE, h_img, totalSize * sizeof(float), cudaMemcpyHostToDevice);
			changestorageordergpu(d_StackA, d_StackE, FFTx, FFTy, FFTz, 1); //1: change tiff storage order to C storage order
		}
		// *** deconvolution ****
		cudaMemset(d_StackE, 0, totalSizeFFT * sizeof(float));
		decon_singleview_OTF2(d_StackE, d_StackA, h_PSFSpectrum, h_FlippedPSFSpectrum, FFTx, FFTy, FFTz, itNumForDecon, flagConstInitial);

		// transfer data back to CPU RAM
		if ((imx < FFTx) || (imy < FFTy) || (imz < FFTz)) {
			cropgpu(d_StackA, d_StackE, imx, imy, imz, FFTx, FFTy, FFTz);//
			changestorageordergpu(d_StackE, d_StackA, imx, imy, imz, -1); //-1: change C storage order to tiff storage order
			cudaMemcpy(h_decon, d_StackE, totalSize * sizeof(float), cudaMemcpyDeviceToHost);
		}
		else {
			changestorageordergpu(d_StackA, d_StackE, imx, imy, imz, -1); //-1: change C storage order to tiff storage order
			cudaMemcpy(h_decon, d_StackA, totalSize * sizeof(float), cudaMemcpyDeviceToHost);
		}
		//printf("...Deconvolution completed ! ! !\n")

		time3 = clock();
		cudaMemGetInfo(&freeMem, &totalMem);
		printf("...GPU free memory (after processing) is %.0f MBites\n", (float)freeMem / 1048576.0f);
		deconRecords[4] = (float)freeMem / 1048576.0f;
		// release variables
		free(h_PSFSpectrum); free(h_FlippedPSFSpectrum);
		cudaFree(d_StackA); cudaFree(d_StackE);
		break;
	default:
		printf("\n****Wrong gpuMemMode setup, no deconvolution performed !!! ****\n");
		return 1;
	}
	end = clock();
	if (gpuMemMode > 0) {
		cudaMemGetInfo(&freeMem, &totalMem);
		printf("GPU free memory (after variable released): %.0f MBites\n", (float)freeMem / 1048576.0f);
	}
	deconRecords[5] = (float)freeMem / 1048576.0f;
	deconRecords[6] = (float)(time1 - start) / CLOCKS_PER_SEC;
	deconRecords[7] = (float)(time2 - time1) / CLOCKS_PER_SEC;
	deconRecords[8] = (float)(time3 - time2) / CLOCKS_PER_SEC;
	deconRecords[9] = (float)(end - start) / CLOCKS_PER_SEC;
	return 0;
}

int decon_dualview(float *h_decon, float *h_img1, float *h_img2, unsigned int *imSize, float *h_psf1, float *h_psf2, unsigned int *psfSize,
	bool flagConstInitial, int itNumForDecon, int deviceNum, int gpuMemMode, bool verbose, float *deconRecords, bool flagUnmatch, float *h_psf_bp1, float *h_psf_bp2) {
	// gpuMemMode --> -1: Automatically set memory mode; 0: All on CPU; 1: sufficient GPU memory; 2: GPU memory optimized.
	//deconRecords: 10 elements
	//[0]:  the actual GPU memory mode used;
	//[1] -[5]: initial GPU memory, after variables partially allocated, during processing, after processing, after variables released ( all in MB);
	//[6] -[9]: initializing time, prepocessing time, decon time, total time;	
	// image size
	long long int
		imx, imy, imz;
	imx = imSize[0], imy = imSize[1], imz = imSize[2];
	// PSF size
	long long int
		PSFx, PSFy, PSFz;
	PSFx = psfSize[0], PSFy = psfSize[1], PSFz = psfSize[2];
	// FFT size
	long long int
		FFTx, FFTy, FFTz;

	FFTx = snapTransformSize(imx);// snapTransformSize(imx + PSFx - 1);
	FFTy = snapTransformSize(imy);// snapTransformSize(imy + PSFy - 1);
	FFTz = snapTransformSize(imz);// snapTransformSize(imz + PSFz - 1);

	if (verbose) {
		printf("\tImage information:\n");
		printf("\t... Image size %d x %d x %d\n  ", imx, imy, imz);
		printf("\t... PSF size %d x %d x %d\n  ", PSFx, PSFy, PSFz);
		printf("\t... FFT size %d x %d x %d\n  ", FFTx, FFTy, FFTz);
		printf("\t... Output Image size %d x %d x %d \n   ", imSize[0], imSize[1], imSize[2]);
	}
	
	// total pixel count for each images
	long long int totalSize = imx*imy*imz; // in floating format
	long long int totalSizePSF = PSFx*PSFy*PSFz; // in floating format
	long long int totalSizeFFT = FFTx*FFTy*FFTz; // in floating format
	long long int totalSizeSpectrum = FFTx * FFTy*(FFTz / 2 + 1); // in complex floating format
	long long int totalSizeMax = totalSizeSpectrum * 2; // in floating format

	// ****************** Processing Starts***************** //
	// variables for memory and time cost records
	clock_t start, time1, time2, time3, end;
	size_t totalMem = 0;
	size_t freeMem = 0;
	time1 = time2 = time3 = end = 0;
	start = clock();
	if (gpuMemMode != 0) {
		cudaSetDevice(deviceNum);
		cudaMemGetInfo(&freeMem, &totalMem);
		printf("\t... GPU free memory before deconvolution is %.0f MBites\n", (float)freeMem / 1048576.0f);
		deconRecords[1] = (float)freeMem / 1048576.0f;
	}
	// ***** Set GPU memory mode based on images size and available GPU memory ****
	// gpuMemMode --> Unified memory in next version???
	// -1: Automatically set memory mode based on calculations; 
	// 0: all in CPU; 1: sufficient GPU memory; 2: GPU memory optimized. 
	if (gpuMemMode == -1) { //Automatically set memory mode based on calculations.
						   // Test to create FFT plans to estimate GPU memory
		cufftHandle
			fftPlanFwd,
			fftPlanInv;
		cufftPlan3d(&fftPlanFwd, FFTx, FFTy, FFTz, CUFFT_R2C);
		cufftPlan3d(&fftPlanInv, FFTx, FFTy, FFTz, CUFFT_C2R);
		cudaCheckErrors("**** GPU out of memory during memory emstimating!!!!*****");
		cudaMemGetInfo(&freeMem, &totalMem);
		if (freeMem > 9 * totalSizeMax * sizeof(float)) { // 6 more GPU variables
			gpuMemMode = 1;
			printf("\t... GPU memory is sufficient, processing in efficient mode !!!\n");
		}
		else if (freeMem > 2 * totalSizeMax * sizeof(float)) {// 2 more GPU variables
			gpuMemMode = 2;
			printf("\t... GPU memory is optimized, processing in memory saved mode !!!\n");
		}
		else { // all processing in CPU
			gpuMemMode = 0;
			printf("\t... GPU memory is not enough, processing in CPU mode!!!\n");
		}
		// destroy plans
		cufftDestroy(fftPlanFwd);
		cufftDestroy(fftPlanInv);
	}
	deconRecords[0] = gpuMemMode;
	
	float
		*h_StackA,
		*h_StackB,
		*h_StackE,
		*d_StackA,
		*d_StackB,
		*d_StackE;

	fComplex
		*h_PSFASpectrum,
		*h_PSFBSpectrum,
		*h_FlippedPSFASpectrum,
		*h_FlippedPSFBSpectrum,
		*d_PSFSpectrum,
		*d_FlippedPSFSpectrum,
		*d_PSFASpectrum,
		*d_PSFBSpectrum,
		*d_FlippedPSFASpectrum,
		*d_FlippedPSFBSpectrum;
	
	switch (gpuMemMode) {
	case 0:
		// CPU deconvolution
		time1 = clock();
		// allocate memory
		h_StackA = (float *)malloc(totalSizeMax * sizeof(float));
		h_StackB = (float *)malloc(totalSizeMax * sizeof(float));
		h_StackE = (float *)malloc(totalSizeMax * sizeof(float));
		h_PSFASpectrum = (fComplex *)malloc(totalSizeSpectrum * sizeof(fftwf_complex));
		h_PSFBSpectrum = (fComplex *)malloc(totalSizeSpectrum * sizeof(fftwf_complex));
		h_FlippedPSFASpectrum = (fComplex *)malloc(totalSizeSpectrum * sizeof(fftwf_complex));
		h_FlippedPSFBSpectrum = (fComplex *)malloc(totalSizeSpectrum * sizeof(fftwf_complex));

		// *** PSF A Preparation
		// OTF 
		changestorageordercpu(h_StackA, h_psf1, PSFx, PSFy, PSFz, 1);
		genOTFcpu((fftwf_complex *)h_PSFASpectrum, h_StackA, FFTx, FFTy, FFTz, PSFx, PSFy, PSFz, true);
		// OTF_bp
		if (flagUnmatch) {
			changestorageordercpu(h_StackA, h_psf_bp1, PSFx, PSFy, PSFz, 1);
			genOTFcpu((fftwf_complex *)h_FlippedPSFASpectrum, h_StackA, FFTx, FFTy, FFTz, PSFx, PSFy, PSFz, true);
		}
		else { // traditional backprojector matched PSF 
			flipcpu(h_StackE, h_StackA, PSFx, PSFy, PSFz); // flip PSF
			genOTFcpu((fftwf_complex *)h_FlippedPSFASpectrum, h_StackE, FFTx, FFTy, FFTz, PSFx, PSFy, PSFz, true);
		}

		// *** PSF B Preparation
		// OTF 
		changestorageordercpu(h_StackA, h_psf2, PSFx, PSFy, PSFz, 1);
		genOTFcpu((fftwf_complex *)h_PSFBSpectrum, h_StackA, FFTx, FFTy, FFTz, PSFx, PSFy, PSFz, true);
		// OTF_bp
		if (flagUnmatch) {
			changestorageordercpu(h_StackA, h_psf_bp2, PSFx, PSFy, PSFz, 1);
			genOTFcpu((fftwf_complex *)h_FlippedPSFBSpectrum, h_StackA, FFTx, FFTy, FFTz, PSFx, PSFy, PSFz, true);
		}
		else { // traditional backprojector matched PSF 
			flipcpu(h_StackE, h_StackA, PSFx, PSFy, PSFz); // flip PSF
			genOTFcpu((fftwf_complex *)h_FlippedPSFBSpectrum, h_StackE, FFTx, FFTy, FFTz, PSFx, PSFy, PSFz, true);
		}
		// *** image A Preparation
		if ((imx < FFTx) || (imy < FFTy) || (imz < FFTz)) {
			changestorageordercpu(h_StackE, h_img1, imx, imy, imz, 1);
			padstackcpu(h_StackA, h_StackE, FFTx, FFTy, FFTz, imx, imy, imz);
		}
		else {
			changestorageordercpu(h_StackA, h_img1, imx, imy, imz, 1);
		}
		// *** image B Preparation
		if ((imx < FFTx) || (imy < FFTy) || (imz < FFTz)) {
			changestorageordercpu(h_StackE, h_img2, imx, imy, imz, 1);
			padstackcpu(h_StackB, h_StackE, FFTx, FFTy, FFTz, imx, imy, imz);
		}
		else {
			changestorageordercpu(h_StackB, h_img2, imx, imy, imz, 1);
		}
		// *** deconvolution ****
		memset(h_StackE, 0, totalSizeFFT * sizeof(float));
		decon_dualview_OTF0(h_StackE, h_StackA, h_StackB, (fftwf_complex *)h_PSFASpectrum, (fftwf_complex *)h_PSFBSpectrum,
			(fftwf_complex *)h_FlippedPSFASpectrum, (fftwf_complex *)h_FlippedPSFBSpectrum, FFTx, FFTy, FFTz, itNumForDecon, flagConstInitial);
		if ((imx < FFTx) || (imy < FFTy) || (imz < FFTz)) {
			cropcpu(h_StackA, h_StackE, imx, imy, imz, FFTx, FFTy, FFTz);
			changestorageordercpu(h_decon, h_StackA, imx, imy, imz, -1);
		}
		else {
			changestorageordercpu(h_decon, h_StackE, imx, imy, imz, -1);
		}
		//printf("...Deconvolution completed ! ! !\n")

		time3 = clock();
		// release variables
		free(h_StackA); free(h_StackB);  free(h_StackE);
		free(h_PSFASpectrum); free(h_PSFBSpectrum);
		free(h_FlippedPSFASpectrum); free(h_FlippedPSFBSpectrum);
		break;
	case 1:// efficient GPU calculation
		time1 = clock();
		// allocate memory
		cudaMalloc((void **)&d_StackA, totalSizeMax * sizeof(float));
		cudaMalloc((void **)&d_StackE, totalSizeMax * sizeof(float));
		cudaMalloc((void **)&d_PSFASpectrum, totalSizeSpectrum * sizeof(fComplex));
		cudaMalloc((void **)&d_PSFBSpectrum, totalSizeSpectrum * sizeof(fComplex));
		cudaMalloc((void **)&d_FlippedPSFASpectrum, totalSizeSpectrum * sizeof(fComplex));
		cudaMalloc((void **)&d_FlippedPSFBSpectrum, totalSizeSpectrum * sizeof(fComplex));
		//check GPU status
		cudaCheckErrors("****Memory allocating fails... GPU out of memory !!!!*****");
		cudaMemGetInfo(&freeMem, &totalMem);
		printf("\t... GPU free memory(after mallocing) is %.0f MBites\n", (float)freeMem / 1048576.0f);
		deconRecords[2] = (float)freeMem / 1048576.0f;
		// *** PSF A Preparation
		// OTF 
		cudaMemcpy(d_StackE, h_psf1, totalSizePSF * sizeof(float), cudaMemcpyHostToDevice);
		changestorageordergpu(d_StackA, d_StackE, PSFx, PSFy, PSFz, 1); 
		genOTFgpu(d_PSFASpectrum, d_StackA, FFTx, FFTy, FFTz, PSFx, PSFy, PSFz, true);
		// OTF_bp
		if (flagUnmatch) {
			cudaMemcpy(d_StackE, h_psf_bp1, totalSizePSF * sizeof(float), cudaMemcpyHostToDevice);
			changestorageordergpu(d_StackA, d_StackE, PSFx, PSFy, PSFz, 1); 
			genOTFgpu(d_FlippedPSFASpectrum, d_StackA, FFTx, FFTy, FFTz, PSFx, PSFy, PSFz, true);
		}
		else { // traditional backprojector matched PSF 
			flipgpu(d_StackE, d_StackA, PSFx, PSFy, PSFz); // flip PSF
			genOTFgpu(d_FlippedPSFASpectrum, d_StackE, FFTx, FFTy, FFTz, PSFx, PSFy, PSFz, true);
		}
		// *** PSF B Preparation
		// OTF 
		cudaMemcpy(d_StackE, h_psf2, totalSizePSF * sizeof(float), cudaMemcpyHostToDevice);
		changestorageordergpu(d_StackA, d_StackE, PSFx, PSFy, PSFz, 1); 
		genOTFgpu(d_PSFBSpectrum, d_StackA, FFTx, FFTy, FFTz, PSFx, PSFy, PSFz, true);
		// OTF_bp
		if (flagUnmatch) {
			cudaMemcpy(d_StackE, h_psf_bp2, totalSizePSF * sizeof(float), cudaMemcpyHostToDevice);
			changestorageordergpu(d_StackA, d_StackE, PSFx, PSFy, PSFz, 1); 
			genOTFgpu(d_FlippedPSFBSpectrum, d_StackA, FFTx, FFTy, FFTz, PSFx, PSFy, PSFz, true);
		}
		else { // traditional backprojector matched PSF 
			flipgpu(d_StackE, d_StackA, PSFx, PSFy, PSFz); // flip PSF
			genOTFgpu(d_FlippedPSFBSpectrum, d_StackE, FFTx, FFTy, FFTz, PSFx, PSFy, PSFz, true); 
		}
		cudaCheckErrors("****PSF and OTF preparation failed !!!!*****");
		cudaMalloc((void **)&d_StackB, totalSizeMax * sizeof(float));
		// *** image A Preparation
		if ((imx < FFTx) || (imy < FFTy) || (imz < FFTz)) {
			cudaMemcpy(d_StackA, h_img1, totalSize * sizeof(float), cudaMemcpyHostToDevice);
			changestorageordergpu(d_StackE, d_StackA, imx, imy, imz, 1); 
			padstackgpu(d_StackA, d_StackE, FFTx, FFTy, FFTz, imx, imy, imz);
		}
		else {
			cudaMemcpy(d_StackE, h_img1, totalSize * sizeof(float), cudaMemcpyHostToDevice);
			changestorageordergpu(d_StackA, d_StackE, FFTx, FFTy, FFTz, 1);
		}
		// *** image B Preparation
		if ((imx < FFTx) || (imy < FFTy) || (imz < FFTz)) {
			cudaMemcpy(d_StackB, h_img2, totalSize * sizeof(float), cudaMemcpyHostToDevice);
			changestorageordergpu(d_StackE, d_StackB, imx, imy, imz, 1); 
			padstackgpu(d_StackB, d_StackE, FFTx, FFTy, FFTz, imx, imy, imz);
		}
		else {
			cudaMemcpy(d_StackE, h_img2, totalSize * sizeof(float), cudaMemcpyHostToDevice);
			changestorageordergpu(d_StackB, d_StackE, FFTx, FFTy, FFTz, 1);
		}
		cudaCheckErrors("****Image preparation failed !!!!*****");
		// *** deconvolution ****
		cudaMemset(d_StackE, 0, totalSizeFFT * sizeof(float));
		decon_dualview_OTF1(d_StackE, d_StackA, d_StackB, d_PSFASpectrum, d_PSFBSpectrum, 
			d_FlippedPSFASpectrum, d_FlippedPSFBSpectrum,FFTx, FFTy, FFTz, itNumForDecon, flagConstInitial);
		// transfer data back to CPU RAM
		if ((imx < FFTx) || (imy < FFTy) || (imz < FFTz)) {
			cropgpu(d_StackA, d_StackE, imx, imy, imz, FFTx, FFTy, FFTz);
			changestorageordergpu(d_StackE, d_StackA, imx, imy, imz, -1); 
			cudaMemcpy(h_decon, d_StackE, totalSize * sizeof(float), cudaMemcpyDeviceToHost);
		}
		else {
			changestorageordergpu(d_StackA, d_StackE, imx, imy, imz, -1); 
			cudaMemcpy(h_decon, d_StackA, totalSize * sizeof(float), cudaMemcpyDeviceToHost);
		}
		//printf("...Deconvolution completed ! ! !\n")

		time3 = clock();
		cudaMemGetInfo(&freeMem, &totalMem);
		printf("\t... GPU free memory (after processing) is %.0f MBites\n", (float)freeMem / 1048576.0f);
		deconRecords[4] = (float)freeMem / 1048576.0f;
		// release CUDA variables
		cudaFree(d_StackA); cudaFree(d_StackB);  cudaFree(d_StackE);
		cudaFree(d_PSFASpectrum); cudaFree(d_PSFBSpectrum); 
		cudaFree(d_FlippedPSFASpectrum); cudaFree(d_FlippedPSFBSpectrum); 
		break;
	case 2:
		time1 = clock();
		h_StackB = (float *)malloc(totalSizeMax * sizeof(float));
		h_PSFASpectrum = (fComplex *)malloc(totalSizeSpectrum * sizeof(fComplex));
		h_PSFBSpectrum = (fComplex *)malloc(totalSizeSpectrum * sizeof(fComplex));
		h_FlippedPSFASpectrum = (fComplex *)malloc(totalSizeSpectrum * sizeof(fComplex));
		h_FlippedPSFBSpectrum = (fComplex *)malloc(totalSizeSpectrum * sizeof(fComplex));
		cudaMalloc((void **)&d_StackA, totalSizeMax * sizeof(float)); // also to store spectrum images
		cudaMalloc((void **)&d_StackE, totalSizeMax * sizeof(float)); // also to store spectrum images
																	  //check GPU status
		cudaCheckErrors("****Memory allocating fails... GPU out of memory !!!!*****");
		cudaMemGetInfo(&freeMem, &totalMem);
		printf("\t... GPU free memory(after mallocing) is %.0f MBites\n", (float)freeMem / 1048576.0f);
		deconRecords[2] = (float)freeMem / 1048576.0f;

		// *** PSF A Preparation
		// OTF 
		cudaMemcpy(d_StackE, h_psf1, totalSizePSF * sizeof(float), cudaMemcpyHostToDevice);
		changestorageordergpu(d_StackA, d_StackE, PSFx, PSFy, PSFz, 1); //1: change tiff storage order to C storage order
		d_PSFSpectrum = (fComplex *)d_StackE;
		genOTFgpu(d_PSFSpectrum, d_StackA, FFTx, FFTy, FFTz, PSFx, PSFy, PSFz, true);
		cudaMemcpy(h_PSFASpectrum, d_PSFSpectrum, totalSizeSpectrum * sizeof(fComplex), cudaMemcpyDeviceToHost);
		// OTF_bp
		if (flagUnmatch) {
			cudaMemcpy(d_StackE, h_psf_bp1, totalSizePSF * sizeof(float), cudaMemcpyHostToDevice);
			changestorageordergpu(d_StackA, d_StackE, PSFx, PSFy, PSFz, 1); 
			genOTFgpu(d_PSFSpectrum, d_StackA, FFTx, FFTy, FFTz, PSFx, PSFy, PSFz, true);
		}
		else { // traditional backprojector matched PSF 
			flipgpu(d_StackE, d_StackA, PSFx, PSFy, PSFz); // flip PSF
			d_PSFSpectrum = (fComplex *)d_StackA;
			genOTFgpu(d_PSFSpectrum, d_StackE, FFTx, FFTy, FFTz, PSFx, PSFy, PSFz, true); 
		}
		cudaMemcpy(h_FlippedPSFASpectrum, d_PSFSpectrum, totalSizeSpectrum * sizeof(fComplex), cudaMemcpyDeviceToHost);

		// *** PSF B Preparation
		// OTF 
		cudaMemcpy(d_StackE, h_psf2, totalSizePSF * sizeof(float), cudaMemcpyHostToDevice);
		changestorageordergpu(d_StackA, d_StackE, PSFx, PSFy, PSFz, 1); 
		d_PSFSpectrum = (fComplex *)d_StackE;
		genOTFgpu(d_PSFSpectrum, d_StackA, FFTx, FFTy, FFTz, PSFx, PSFy, PSFz, true);
		cudaMemcpy(h_PSFBSpectrum, d_PSFSpectrum, totalSizeSpectrum * sizeof(fComplex), cudaMemcpyDeviceToHost);
		// OTF_bp
		if (flagUnmatch) {
			cudaMemcpy(d_StackE, h_psf_bp2, totalSizePSF * sizeof(float), cudaMemcpyHostToDevice);
			changestorageordergpu(d_StackA, d_StackE, PSFx, PSFy, PSFz, 1); 
			genOTFgpu(d_PSFSpectrum, d_StackA, FFTx, FFTy, FFTz, PSFx, PSFy, PSFz, true);
		}
		else { // traditional backprojector matched PSF 
			flipgpu(d_StackE, d_StackA, PSFx, PSFy, PSFz); // flip PSF
			d_PSFSpectrum = (fComplex *)d_StackA;
			genOTFgpu(d_PSFSpectrum, d_StackE, FFTx, FFTy, FFTz, PSFx, PSFy, PSFz, true); //PSF already normalized
		}
		cudaMemcpy(h_FlippedPSFBSpectrum, d_PSFSpectrum, totalSizeSpectrum * sizeof(fComplex), cudaMemcpyDeviceToHost);
		cudaCheckErrors("****PSF and OTF preparation failed !!!!*****");

		// *** image B Preparation
		if ((imx < FFTx) || (imy < FFTy) || (imz < FFTz)) {
			cudaMemcpy(d_StackA, h_img2, totalSize * sizeof(float), cudaMemcpyHostToDevice);
			changestorageordergpu(d_StackE, d_StackA, imx, imy, imz, 1); 
			padstackgpu(d_StackA, d_StackE, FFTx, FFTy, FFTz, imx, imy, imz);//
		}
		else {
			cudaMemcpy(d_StackE, h_img2, totalSize * sizeof(float), cudaMemcpyHostToDevice);
			changestorageordergpu(d_StackA, d_StackE, imx, imy, imz, 1); 
		}
		cudaMemcpy(h_StackB, d_StackA, totalSizeFFT * sizeof(float), cudaMemcpyDeviceToHost);
		// *** image A Preparation
		if ((imx < FFTx) || (imy < FFTy) || (imz < FFTz)) {
			cudaMemcpy(d_StackA, h_img1, totalSize * sizeof(float), cudaMemcpyHostToDevice);
			changestorageordergpu(d_StackE, d_StackA, imx, imy, imz, 1); 
			padstackgpu(d_StackA, d_StackE, FFTx, FFTy, FFTz, imx, imy, imz);//
		}
		else {
			cudaMemcpy(d_StackE, h_img1, totalSize * sizeof(float), cudaMemcpyHostToDevice);
			changestorageordergpu(d_StackA, d_StackE, imx, imy, imz, 1); 
		}
		cudaCheckErrors("****Image preparation failed !!!!*****");
		
		// *** deconvolution ****
		cudaMemset(d_StackE, 0, totalSizeFFT * sizeof(float));
		decon_dualview_OTF2(d_StackE, d_StackA, h_StackB, h_PSFASpectrum, h_PSFBSpectrum,
			h_FlippedPSFASpectrum, h_FlippedPSFBSpectrum, FFTx, FFTy, FFTz, itNumForDecon, flagConstInitial);

		// transfer data back to CPU RAM
		if ((imx < FFTx) || (imy < FFTy) || (imz < FFTz)) {
			cropgpu(d_StackA, d_StackE, imx, imy, imz, FFTx, FFTy, FFTz);//
			changestorageordergpu(d_StackE, d_StackA, imx, imy, imz, -1); //-1: change C storage order to tiff storage order
			cudaMemcpy(h_decon, d_StackE, totalSize * sizeof(float), cudaMemcpyDeviceToHost);
		}
		else {
			changestorageordergpu(d_StackA, d_StackE, imx, imy, imz, -1); //-1: change C storage order to tiff storage order
			cudaMemcpy(h_decon, d_StackA, totalSize * sizeof(float), cudaMemcpyDeviceToHost);
		}
		//printf("...Deconvolution completed ! ! !\n")

		time3 = clock();
		cudaMemGetInfo(&freeMem, &totalMem);
		printf("\t... GPU free memory (after processing) is %.0f MBites\n", (float)freeMem / 1048576.0f);
		deconRecords[4] = (float)freeMem / 1048576.0f;
		// release variables
		free(h_PSFASpectrum); free(h_PSFBSpectrum);
		free(h_FlippedPSFASpectrum); free(h_FlippedPSFBSpectrum);
		cudaFree(d_StackA); cudaFree(d_StackE);
		break;
	default:
		printf("\n****Wrong gpuMemMode setup, no deconvolution performed !!! ****\n");
		return -1;
	}
	end = clock();
	if (gpuMemMode > 0) {
		cudaMemGetInfo(&freeMem, &totalMem);
		printf("\t... GPU free memory (after variable released): %.0f MBites\n", (float)freeMem / 1048576.0f);
	}
	deconRecords[5] = (float)freeMem / 1048576.0f;
	deconRecords[6] = (float)(time1 - start) / CLOCKS_PER_SEC;
	deconRecords[7] = (float)(time2 - time1) / CLOCKS_PER_SEC;
	deconRecords[8] = (float)(time3 - time2) / CLOCKS_PER_SEC;
	deconRecords[9] = (float)(end - start) / CLOCKS_PER_SEC;
	if (verbose) {
		printf("\t... deconvolution done !!! \n");
		//printf("\t...Time cost is %2.3f s\n", records[9]);
	}
	return 0;
}

// batch
int decon_dualview_batch(float *h_decon, float *h_img1, float *h_img2, unsigned int *imSize, fComplex *OTF1, fComplex *OTF2, fComplex *OTF1_bp, fComplex *OTF2_bp, unsigned int *otfSize,
	bool flagConstInitial, int itNumForDecon, int deviceNum, int gpuMemMode, bool verbose, float *deconRecords) {
	// gpuMemMode --> -1: Automatically set memory mode; 0: All on CPU; 1: sufficient GPU memory; 2: GPU memory optimized.
	//deconRecords: 10 elements
	//[0]:  the actual GPU memory mode used;
	//[1] -[5]: initial GPU memory, after variables partially allocated, during processing, after processing, after variables released ( all in MB);
	//[6] -[9]: initializing time, prepocessing time, decon time, total time;	
	// image size
	long long int
		imx, imy, imz;
	imx = imSize[0], imy = imSize[1], imz = imSize[2];
	// FFT size
	long long int
		FFTx, FFTy, FFTz;
	FFTx = otfSize[0];
	FFTy = otfSize[1];
	FFTz = otfSize[2];

	// total pixel count for each images
	long long int totalSize = imx*imy*imz; // in floating format
	long long int totalSizeFFT = FFTx*FFTy*FFTz; // in floating format
	long long int totalSizeSpectrum = FFTx * FFTy*(FFTz / 2 + 1); // in complex floating format
	long long int totalSizeMax = totalSizeSpectrum * 2; // in floating format

														// ****************** Processing Starts***************** //
														// variables for memory and time cost records
	clock_t start, time1, time2, time3, end;
	size_t totalMem = 0;
	size_t freeMem = 0;
	time1 = time2 = time3 = end = 0;
	start = clock();
	if (gpuMemMode != 0) {
		cudaSetDevice(deviceNum);
		cudaMemGetInfo(&freeMem, &totalMem);
		printf("...GPU free memory(at beginning) is %.0f MBites\n", (float)freeMem / 1048576.0f);
		deconRecords[1] = (float)freeMem / 1048576.0f;
	}
	// ***** Set GPU memory mode based on images size and available GPU memory ****
	// gpuMemMode --> Unified memory in next version???
	// -1: Automatically set memory mode based on calculations; 
	// 0: all in CPU; 1: sufficient GPU memory; 2: GPU memory optimized. 
	if (gpuMemMode == -1) { //Automatically set memory mode based on calculations.
							// Test to create FFT plans to estimate GPU memory
		cufftHandle
			fftPlanFwd,
			fftPlanInv;
		cufftPlan3d(&fftPlanFwd, FFTx, FFTy, FFTz, CUFFT_R2C);
		cufftPlan3d(&fftPlanInv, FFTx, FFTy, FFTz, CUFFT_C2R);
		cudaCheckErrors("**** GPU out of memory during memory emstimating!!!!*****");
		cudaMemGetInfo(&freeMem, &totalMem);
		if (freeMem > 9 * totalSizeMax * sizeof(float)) { // 6 more GPU variables
			gpuMemMode = 1;
			printf("\n GPU memory is sufficient, processing in efficient mode !!!\n");
		}
		else if (freeMem > 2 * totalSizeMax * sizeof(float)) {// 2 more GPU variables
			gpuMemMode = 2;
			printf("\n GPU memory is optimized, processing in memory saved mode !!!\n");
		}
		else { // all processing in CPU
			gpuMemMode = 0;
			printf("\n GPU memory is not enough, processing in CPU mode!!!\n");
		}
		// destroy plans
		cufftDestroy(fftPlanFwd);
		cufftDestroy(fftPlanInv);
	}
	deconRecords[0] = gpuMemMode;

	float
		*h_StackA,
		*h_StackB,
		*h_StackE,
		*d_StackA,
		*d_StackB,
		*d_StackE;

	fComplex
		*h_PSFASpectrum,
		*h_PSFBSpectrum,
		*h_FlippedPSFASpectrum,
		*h_FlippedPSFBSpectrum,
		*d_PSFSpectrum,
		*d_FlippedPSFSpectrum,
		*d_PSFASpectrum,
		*d_PSFBSpectrum,
		*d_FlippedPSFASpectrum,
		*d_FlippedPSFBSpectrum;

	switch (gpuMemMode) {
	case 0:
		// CPU deconvolution
		time1 = clock();
		// allocate memory
		h_StackA = (float *)malloc(totalSizeMax * sizeof(float));
		h_StackB = (float *)malloc(totalSizeMax * sizeof(float));
		h_StackE = (float *)malloc(totalSizeMax * sizeof(float));
		// *** PSF A Preparation
		// OTF 
		h_PSFASpectrum = OTF1;
		h_FlippedPSFASpectrum = OTF1_bp;
		// *** PSF B Preparation
		// OTF 
		h_PSFBSpectrum = OTF2;
		h_FlippedPSFBSpectrum = OTF2_bp;
		// *** image A Preparation
		if ((imx < FFTx) || (imy < FFTy) || (imz < FFTz)) {
			changestorageordercpu(h_StackE, h_img1, imx, imy, imz, 1);
			padstackcpu(h_StackA, h_StackE, FFTx, FFTy, FFTz, imx, imy, imz);
		}
		else {
			changestorageordercpu(h_StackA, h_img1, imx, imy, imz, 1);
		}
		// *** image B Preparation
		if ((imx < FFTx) || (imy < FFTy) || (imz < FFTz)) {
			changestorageordercpu(h_StackE, h_img2, imx, imy, imz, 1);
			padstackcpu(h_StackB, h_StackE, FFTx, FFTy, FFTz, imx, imy, imz);
		}
		else {
			changestorageordercpu(h_StackB, h_img2, imx, imy, imz, 1);
		}
		// *** deconvolution ****
		memset(h_StackE, 0, totalSizeFFT * sizeof(float));
		decon_dualview_OTF0(h_StackE, h_StackA, h_StackB, (fftwf_complex *)h_PSFASpectrum, (fftwf_complex *)h_PSFBSpectrum,
			(fftwf_complex *)h_FlippedPSFASpectrum, (fftwf_complex *)h_FlippedPSFBSpectrum, FFTx, FFTy, FFTz, itNumForDecon, flagConstInitial);
		if ((imx < FFTx) || (imy < FFTy) || (imz < FFTz)) {
			cropcpu(h_StackA, h_StackE, imx, imy, imz, FFTx, FFTy, FFTz);
			changestorageordercpu(h_decon, h_StackA, imx, imy, imz, -1);
		}
		else {
			changestorageordercpu(h_decon, h_StackE, imx, imy, imz, -1);
		}
		//printf("...Deconvolution completed ! ! !\n")

		time3 = clock();
		// release variables
		free(h_StackA); free(h_StackB);  free(h_StackE);
		break;
	case 1:// efficient GPU calculation
		time1 = clock();
		// allocate memory
		cudaMalloc((void **)&d_StackA, totalSizeMax * sizeof(float));
		cudaMalloc((void **)&d_StackE, totalSizeMax * sizeof(float));
		//check GPU status
		cudaCheckErrors("****Memory allocating fails... GPU out of memory !!!!*****");
		cudaMemGetInfo(&freeMem, &totalMem);
		deconRecords[2] = (float)freeMem / 1048576.0f;
		// *** PSF A Preparation
		// OTF 
		h_PSFASpectrum = OTF1;
		h_FlippedPSFASpectrum = OTF1_bp;
		// *** PSF B Preparation
		// OTF 
		h_PSFBSpectrum = OTF2;
		h_FlippedPSFBSpectrum = OTF2_bp;
		// *** image A Preparation
		if ((imx < FFTx) || (imy < FFTy) || (imz < FFTz)) {
			cudaMemcpy(d_StackA, h_img1, totalSize * sizeof(float), cudaMemcpyHostToDevice);
			changestorageordergpu(d_StackE, d_StackA, imx, imy, imz, 1);
			padstackgpu(d_StackA, d_StackE, FFTx, FFTy, FFTz, imx, imy, imz);
		}
		else {
			cudaMemcpy(d_StackE, h_img1, totalSize * sizeof(float), cudaMemcpyHostToDevice);
			changestorageordergpu(d_StackA, d_StackE, FFTx, FFTy, FFTz, 1);
		}
		// *** image B Preparation
		if ((imx < FFTx) || (imy < FFTy) || (imz < FFTz)) {
			cudaMemcpy(d_StackB, h_img2, totalSize * sizeof(float), cudaMemcpyHostToDevice);
			changestorageordergpu(d_StackE, d_StackB, imx, imy, imz, 1);
			padstackgpu(d_StackB, d_StackE, FFTx, FFTy, FFTz, imx, imy, imz);
		}
		else {
			cudaMemcpy(d_StackE, h_img2, totalSize * sizeof(float), cudaMemcpyHostToDevice);
			changestorageordergpu(d_StackB, d_StackE, FFTx, FFTy, FFTz, 1);
		}
		cudaCheckErrors("****Image preparation failed !!!!*****");
		// *** deconvolution ****
		cudaMemset(d_StackE, 0, totalSizeFFT * sizeof(float));
		decon_dualview_OTF1(d_StackE, d_StackA, d_StackB, d_PSFASpectrum, d_PSFBSpectrum,
			d_FlippedPSFASpectrum, d_FlippedPSFBSpectrum, FFTx, FFTy, FFTz, itNumForDecon, flagConstInitial);
		// transfer data back to CPU RAM
		if ((imx < FFTx) || (imy < FFTy) || (imz < FFTz)) {
			cropgpu(d_StackA, d_StackE, imx, imy, imz, FFTx, FFTy, FFTz);
			changestorageordergpu(d_StackE, d_StackA, imx, imy, imz, -1);
			cudaMemcpy(h_decon, d_StackE, totalSize * sizeof(float), cudaMemcpyDeviceToHost);
		}
		else {
			changestorageordergpu(d_StackA, d_StackE, imx, imy, imz, -1);
			cudaMemcpy(h_decon, d_StackA, totalSize * sizeof(float), cudaMemcpyDeviceToHost);
		}
		//printf("...Deconvolution completed ! ! !\n")

		time3 = clock();
		cudaMemGetInfo(&freeMem, &totalMem);
		deconRecords[4] = (float)freeMem / 1048576.0f;
		// release CUDA variables
		cudaFree(d_StackA); cudaFree(d_StackB);  cudaFree(d_StackE);
		break;
	case 2:
		time1 = clock();
		h_StackB = (float *)malloc(totalSizeMax * sizeof(float));
		cudaMalloc((void **)&d_StackA, totalSizeMax * sizeof(float)); // also to store spectrum images
		cudaMalloc((void **)&d_StackE, totalSizeMax * sizeof(float)); // also to store spectrum images
																	  //check GPU status
		cudaCheckErrors("****Memory allocating fails... GPU out of memory !!!!*****");

		deconRecords[2] = (float)freeMem / 1048576.0f;

		// *** PSF A Preparation
		// OTF 
		h_PSFASpectrum = OTF1;
		h_FlippedPSFASpectrum = OTF1_bp;
		// *** PSF B Preparation
		// OTF 
		h_PSFBSpectrum = OTF2;
		h_FlippedPSFBSpectrum = OTF2_bp;

		// *** image B Preparation
		if ((imx < FFTx) || (imy < FFTy) || (imz < FFTz)) {
			cudaMemcpy(d_StackA, h_img2, totalSize * sizeof(float), cudaMemcpyHostToDevice);
			changestorageordergpu(d_StackE, d_StackA, imx, imy, imz, 1);
			padstackgpu(d_StackA, d_StackE, FFTx, FFTy, FFTz, imx, imy, imz);//
		}
		else {
			cudaMemcpy(d_StackE, h_img2, totalSize * sizeof(float), cudaMemcpyHostToDevice);
			changestorageordergpu(d_StackA, d_StackE, imx, imy, imz, 1);
		}
		cudaMemcpy(h_StackB, d_StackA, totalSizeFFT * sizeof(float), cudaMemcpyDeviceToHost);
		// *** image A Preparation
		if ((imx < FFTx) || (imy < FFTy) || (imz < FFTz)) {
			cudaMemcpy(d_StackA, h_img1, totalSize * sizeof(float), cudaMemcpyHostToDevice);
			changestorageordergpu(d_StackE, d_StackA, imx, imy, imz, 1);
			padstackgpu(d_StackA, d_StackE, FFTx, FFTy, FFTz, imx, imy, imz);//
		}
		else {
			cudaMemcpy(d_StackE, h_img1, totalSize * sizeof(float), cudaMemcpyHostToDevice);
			changestorageordergpu(d_StackA, d_StackE, imx, imy, imz, 1);
		}
		cudaCheckErrors("****Image preparation failed !!!!*****");

		// *** deconvolution ****
		cudaMemset(d_StackE, 0, totalSizeFFT * sizeof(float));
		decon_dualview_OTF2(d_StackE, d_StackA, h_StackB, h_PSFASpectrum, h_PSFBSpectrum,
			h_FlippedPSFASpectrum, h_FlippedPSFBSpectrum, FFTx, FFTy, FFTz, itNumForDecon, flagConstInitial);

		// transfer data back to CPU RAM
		if ((imx < FFTx) || (imy < FFTy) || (imz < FFTz)) {
			cropgpu(d_StackA, d_StackE, imx, imy, imz, FFTx, FFTy, FFTz);//
			changestorageordergpu(d_StackE, d_StackA, imx, imy, imz, -1); //-1: change C storage order to tiff storage order
			cudaMemcpy(h_decon, d_StackE, totalSize * sizeof(float), cudaMemcpyDeviceToHost);
		}
		else {
			changestorageordergpu(d_StackA, d_StackE, imx, imy, imz, -1); //-1: change C storage order to tiff storage order
			cudaMemcpy(h_decon, d_StackA, totalSize * sizeof(float), cudaMemcpyDeviceToHost);
		}
		//printf("...Deconvolution completed ! ! !\n")

		time3 = clock();
		cudaMemGetInfo(&freeMem, &totalMem);
		printf("...GPU free memory (after processing) is %.0f MBites\n", (float)freeMem / 1048576.0f);
		deconRecords[4] = (float)freeMem / 1048576.0f;
		// release variables
		cudaFree(d_StackA); cudaFree(d_StackE);
		break;
	default:
		printf("\n****Wrong gpuMemMode setup, no deconvolution performed !!! ****\n");
		return -1;
	}
	end = clock();
	if (gpuMemMode > 0) {
		cudaMemGetInfo(&freeMem, &totalMem);
		printf("GPU free memory (after variable released): %.0f MBites\n", (float)freeMem / 1048576.0f);
	}
	deconRecords[5] = (float)freeMem / 1048576.0f;
	deconRecords[6] = (float)(time1 - start) / CLOCKS_PER_SEC;
	deconRecords[7] = (float)(time2 - time1) / CLOCKS_PER_SEC;
	deconRecords[8] = (float)(time3 - time2) / CLOCKS_PER_SEC;
	deconRecords[9] = (float)(end - start) / CLOCKS_PER_SEC;
	return 0;
}

//// 3D fusion: registration and deconvolution
int fusion_dualview(float *h_decon, float *h_reg, float *h_prereg1, float *h_prereg2, float *iTmx, float *h_img1, float *h_img2, unsigned int *imSizeIn1, unsigned int *imSizeIn2,
	float *pixelSize1, float *pixelSize2, int imRotation, bool flagTmx, int regChoice, float FTOL, int itLimit, float *h_psf1, float *h_psf2,
	unsigned int *psfSizeIn, int itNumForDecon, int deviceNum, int gpuMemMode, bool verbose, float *fusionRecords, bool flagUnmatch, float *h_psf_bp1, float *h_psf_bp2){
	// **** registration and joint deconvolution for two images:  ***
	/*
	*** imBRotation: image B rotation
	0: no rotation; 
	1: 90deg rotation by y axis ; 
	-1: -90deg rotation by y axis;
	*
	*** registration choice: regChoice
	0: no phasor or affine registration; if flagTmx is true, transform d_img2 based on input matrix;
	1: phasor registraion (pixel-level translation only);
	2: affine registration (with or without input matrix); affine: 12 degrees of freedom;
	3: phasor registration --> affine registration (input matrix disabled); affine: 3 DOF --> 6 DOF --> 9 DOF --> 12 DOF
	4: 2D MIP registration --> affine registration (input matrix disabled); affine: 3 DOF --> 6 DOF --> 9 DOF --> 12 DOF
	*
	*** flagTmx: only if regChoice == 0, 2
	true: use iTmx as input matrix;
	false: default;
	*
	*** gpuMemMode
	-1: Automatically set memory mode;
	0: All on CPU. // currently does not work
	1: sufficient GPU memory;
	2: GPU memory optimized;
	*
	*** fusionRecords: 22 element array
	--> 0-10: regRecords; 11-20: deconRecords; 21: total time;
	[0]: actual gpu memory mode for registration
	[1] -[3]: initial ZNCC (zero-normalized cross-correlation, negtive of the cost function), intermediate ZNCC, optimized ZNCC;
	[4] -[7]: single sub iteration time (in ms), total number of sub iterations, iteralation time (in s), whole registration time (in s);
	[8] -[10]: initial GPU memory, before registration, after processing ( all in MB), if use gpu
	[11]:  the actual GPU memory mode used for deconvolution;
	[12] -[16]: initial GPU memory, after variables partially allocated, during processing, after processing, after variables released ( all in MB);
	[17] -[20]: initializing time, prepocessing time, decon time, total time;
	[21]: total time
	*
	*** flagUnmatach 
	false: use traditional backprojector (flipped PSF);
	true: use unmatch back projector;
	*/

	// ************get basic input images information ******************	
	// variables for memory and time cost records
	clock_t start, end;
	end = 0;
	start = clock();
	// ****************** calculate images' size ************************* //
	long long int imx, imy, imz;
	long long int imx1, imy1, imz1, imx2, imy2, imz2;
	unsigned int imSize[3], imSize1[3], imSize2[3], imSizeTemp[3]; // modify to long long 
	float pixelSize[3], pixelSizeTemp[3];
	bool flagInterp1 = true, flagInterp2 = true;
	if ((pixelSize1[0] == pixelSize1[1]) && (pixelSize1[0] == pixelSize1[2]))
		flagInterp1 = false;
	if ((pixelSize2[0] == pixelSize1[0]) && (pixelSize2[1] == pixelSize1[0]) && (pixelSize2[2] == pixelSize1[0]))
		flagInterp2 = false;
	// image A: base image
	pixelSize[0] = pixelSize[1] = pixelSize[2] = pixelSize1[0];
	imx1 = imSizeIn1[0];
	imy1 = round((float)imSizeIn1[1] * pixelSize1[1] / pixelSize[1]);
	imz1 = round((float)imSizeIn1[2] * pixelSize1[2] / pixelSize[2]);
	imSize[0] = imSize1[0] = imx1; imSize[1] = imSize1[1] = imy1; imSize[2] = imSize1[2] = imz1;
	imx = imx1; imy = imy1; imz = imz1; // also as output size

	// image B: target image
	imSizeTemp[0] = imSizeIn2[0]; imSizeTemp[1] = imSizeIn2[1]; imSizeTemp[2] = imSizeIn2[2];
	pixelSizeTemp[0] = pixelSize2[0]; pixelSizeTemp[1] = pixelSize2[1]; pixelSizeTemp[2] = pixelSize2[2];
	if ((imRotation == 1) || (imRotation == -1)){ //if there is rotation for B, change image dimemsion size
		imSizeIn2[0] = imSizeTemp[2];
		imSizeIn2[2] = imSizeTemp[0];
		pixelSize2[0] = pixelSizeTemp[2];
		pixelSize2[2] = pixelSizeTemp[0];
	}
	float pixelSizeRatioBx = pixelSize2[0] / pixelSize[0];
	float pixelSizeRatioBy = pixelSize2[1] / pixelSize[1];
	float pixelSizeRatioBz = pixelSize2[2] / pixelSize[2];
	imx2 = round((float)imSizeIn2[0] * pixelSizeRatioBx);
	imy2 = round((float)imSizeIn2[1] * pixelSizeRatioBy);
	imz2 = round((float)imSizeIn2[2] * pixelSizeRatioBz);
	imSize2[0] = imx2; imSize2[1] = imy2; imSize2[2] = imz2;

	// PSF size
	long long int
		PSFx, PSFy, PSFz;
	PSFx = psfSizeIn[0], PSFy = psfSizeIn[1], PSFz = psfSizeIn[2];

	//FFT size
	long long int
		FFTx, FFTy, FFTz;

	FFTx = snapTransformSize(imx);// snapTransformSize(imx + PSFx - 1);
	FFTy = snapTransformSize(imy);// snapTransformSize(imy + PSFy - 1);
	FFTz = snapTransformSize(imz);// snapTransformSize(imz + PSFz - 1);

	// total pixel count for each images
	long long int totalSizeIn1 = imSizeIn1[0] * imSizeIn1[1] * imSizeIn1[2]; // in floating format
	long long int totalSizeIn2 = imSizeIn2[0] * imSizeIn2[1] * imSizeIn2[2]; // in floating format
	long long int totalSize1 = imx1*imy1*imz1; // in floating format
	long long int totalSize2 = imx2*imy2*imz2; // in floating format
	long long int totalSize = totalSize1; // in floating format
	long long int totalSizeFFT = FFTx*FFTy*(FFTz / 2 + 1); // in complex floating format
	long long int totalSize12 = (totalSize1 > totalSize2) ? totalSize1 : totalSize2;
	long long int totalSizeMax = (totalSize1 > totalSizeFFT * 2) ? totalSize1 : totalSizeFFT * 2; // in floating format
	
	// ****************** Processing Starts*****************
	size_t totalMem = 0;
	size_t freeMem = 0;
	if (gpuMemMode != 0) {
		cudaSetDevice(deviceNum);
		cudaMemGetInfo(&freeMem, &totalMem);
		printf("...GPU free memory(at beginning) is %.0f MBites\n", (float)freeMem / 1048576.0f);
	}

	// ***** Set GPU memory mode based on images size and available GPU memory ****
	// gpuMemMode --> Unified memory in next version???
	// -1: Automatically set memory mode based on calculations; 
	// 0: all in CPU; 1: sufficient GPU memory; 2: GPU memory optimized. 
	if (gpuMemMode == -1) { //Automatically set memory mode based on calculations.
							// Test to create FFT plans to estimate GPU memory
		cufftHandle
			fftPlanFwd,
			fftPlanInv;
		cufftPlan3d(&fftPlanFwd, FFTx, FFTy, FFTz, CUFFT_R2C);
		cufftPlan3d(&fftPlanInv, FFTx, FFTy, FFTz, CUFFT_C2R);
		cudaCheckErrors("**** GPU out of memory during memory emstimating!!!!*****");
		cudaMemGetInfo(&freeMem, &totalMem);
		if (freeMem > 9 * totalSizeMax * sizeof(float)) { // 6 more GPU variables
			gpuMemMode = 1;
			printf("\n GPU memory is sufficient, processing in efficient mode !!!\n");
		}
		else if (freeMem > 2 * totalSizeMax * sizeof(float)) {// 2 more GPU variables
			gpuMemMode = 2;
			printf("\n GPU memory is optimized, processing in memory saved mode !!!\n");
		}
		else { // all processing in CPU
			gpuMemMode = 0;
			printf("\n GPU memory is not enough, processing in CPU mode!!!\n");
		}
		// destroy plans
		cufftDestroy(fftPlanFwd);
		cufftDestroy(fftPlanInv);
	}

	if ((gpuMemMode != 1) || (gpuMemMode != 2)) {
		printf("\n****Wrong gpuMemMode setup (All in CPU is currently not supported), processing stopped !!! ****\n");
		return 1;
	}

	// ************** Registration *************
	// ***interpolation and rotation
	float
		*h_StackA,
		*h_StackB;
	float
		*d_imgE;
	float *d_img3D = NULL, *d_img2DMax = NULL;
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
	cudaArray *d_Array1, *d_Array2;
	float *h_aff12 = (float *)malloc((NDIM)* sizeof(float));
	h_StackA = (float *)malloc(totalSize12 * sizeof(float));
	h_StackB = (float *)malloc(totalSize12 * sizeof(float));
	cudaMalloc((void **)&d_img3D, totalSize12 *sizeof(float));
		
	//// image 1
	if (flagInterp1){
		cudaMalloc3DArray(&d_Array1, &channelDesc, make_cudaExtent(imSizeIn1[0], imSizeIn1[1], imSizeIn1[2]));
		cudacopyhosttoarray(d_Array1, channelDesc, h_img1, imSizeIn1[0], imSizeIn1[1], imSizeIn1[2]);
		BindTexture(d_Array1, channelDesc);
		cudaCheckErrors("Texture create fail");
		// transform matrix for Stack A interpolation
		h_aff12[0] = 1, h_aff12[1] = 0, h_aff12[2] = 0, h_aff12[3] = 0;
		h_aff12[4] = 0, h_aff12[5] = pixelSize[1] / pixelSize1[1], h_aff12[6] = 0, h_aff12[7] = 0;
		h_aff12[8] = 0, h_aff12[9] = 0, h_aff12[10] = pixelSize[2] / pixelSize1[2], h_aff12[11] = 0;
		CopyTranMatrix(h_aff12, NDIM * sizeof(float));
		affineTransform(d_img3D, imx1, imy1, imz1, imSizeIn1[0], imSizeIn1[1], imSizeIn1[2]);
		UnbindTexture();
		cudaFreeArray(d_Array1);
		cudaMemcpy(h_StackA, d_img3D, totalSize * sizeof(float), cudaMemcpyDeviceToHost);
	}// after interpolation, Stack A size: imx x imy x imz;
	else
		memcpy(h_StackA, h_img1, totalSize * sizeof(float));
	cudaThreadSynchronize();

	//// image 2
	// rotation
	if ((imRotation == 1) || (imRotation == -1)){
		cudaMalloc((void **)&d_imgE, totalSizeIn2 * sizeof(float));
		cudaMemcpy(d_imgE, h_img2, totalSizeIn2 * sizeof(float), cudaMemcpyHostToDevice);
		rotbyyaxis(d_img3D, d_imgE, imSizeIn2[0], imSizeIn2[1], imSizeIn2[2], imRotation);
		cudaMemcpy(h_StackB, d_img3D, imSizeIn2[0] * imSizeIn2[1] * imSizeIn2[2] * sizeof(float), cudaMemcpyDeviceToHost);
		cudaFree(d_imgE);
	}
	if (flagInterp2){
		cudaMalloc3DArray(&d_Array2, &channelDesc, make_cudaExtent(imSizeIn2[0], imSizeIn2[1], imSizeIn2[2]));
		if ((imRotation == 1) || (imRotation == -1))
			cudacopyhosttoarray(d_Array2, channelDesc, h_StackB, imSizeIn2[0], imSizeIn2[1], imSizeIn2[2]);
		else
			cudacopyhosttoarray(d_Array2, channelDesc, h_img2, imSizeIn2[0], imSizeIn2[1], imSizeIn2[2]);
		BindTexture(d_Array2, channelDesc);
		cudaCheckErrors("Texture create fail");
		// transform matrix for Stack A interpolation
		h_aff12[0] = pixelSize[0] / pixelSize2[0], h_aff12[1] = 0, h_aff12[2] = 0, h_aff12[3] = 0;
		h_aff12[4] = 0, h_aff12[5] = pixelSize[1] / pixelSize2[1], h_aff12[6] = 0, h_aff12[7] = 0;
		h_aff12[8] = 0, h_aff12[9] = 0, h_aff12[10] = pixelSize[2] / pixelSize2[2], h_aff12[11] = 0;
		CopyTranMatrix(h_aff12, NDIM * sizeof(float));
		affineTransform(d_img3D, imx2, imy2, imz2, imSizeIn2[0], imSizeIn2[1], imSizeIn2[2]);
		UnbindTexture();
		cudaFreeArray(d_Array2);
		cudaMemcpy(h_StackB, d_img3D, totalSize2 * sizeof(float), cudaMemcpyDeviceToHost);
	}// after interpolation, Stack A size: imx x imy x imz;
	else
		memcpy(h_StackB, h_img2, totalSize2 * sizeof(float));
	cudaThreadSynchronize();
	cudaFree(d_img3D);
	int runStatus = 0;
	memcpy(h_prereg1, h_StackA, totalSize * sizeof(float));
	runStatus = alignsize3d(h_prereg2, h_StackB, imz, imy, imx, imz2, imy2, imx2,gpuMemMode);
	// ***** perform registration
	printf("Running registration ...\n");
	int affMethod = 7;
	switch (regChoice) {
	case 0:
		break;
	case 1:
		break;
	case 2:
		if (flagTmx)
			affMethod = 5;
		else
			affMethod = 7;
		break;
	case 3:
		flagTmx = false;
		affMethod = 7;
		break;
	case 4:
		flagTmx = false;
		affMethod = 7;
		break;
	default:
		printf("Wrong registration choice, processing stopped !!!\n");
		return 1;
	}
	float *regRecords = (float *)malloc(11 * sizeof(float));
	int regStatus = reg3d(h_reg, iTmx, h_prereg1, h_prereg2, &imSize[0], &imSize[0], regChoice, affMethod,
		flagTmx, FTOL, itLimit, deviceNum, gpuMemMode, verbose, regRecords);
	memcpy(fusionRecords, regRecords, 11 * sizeof(float));
	free(h_StackB);
	free(regRecords);
	if (regStatus != 0) {
		printf("Registration error, processing stopped !!!\n");
		return 1;
	}
	bool mStatus = checkmatrix(iTmx, imx, imy, imz);
	if (!mStatus) {
		regChoice = 2;
		regStatus = reg3d(h_reg, iTmx, h_img1, h_img2, imSize1, imSize2, regChoice, affMethod,
			flagTmx, FTOL, itLimit, deviceNum, gpuMemMode, verbose, regRecords);
	}
	
	
	cudaMemGetInfo(&freeMem, &totalMem);
	printf("...GPU free memory before deconvolution is %.0f MBites\n", (float)freeMem / 1048576.0f);
	// ***** Joint deconvolution
	float *deconRecords = (float *)malloc(10 * sizeof(float));
	int deconStatus =  decon_dualview(h_decon, h_prereg1, h_reg, &imSize[0], h_psf1, h_psf2,
		psfSizeIn, true, itNumForDecon, deviceNum, gpuMemMode, verbose, deconRecords, flagUnmatch, h_psf_bp1, h_psf_bp2);
	memcpy(&fusionRecords[11], deconRecords, 10 * sizeof(float));
	free(deconRecords);
	free(h_StackA);
	free(h_aff12);
	cudaMemGetInfo(&freeMem, &totalMem);
	printf("...GPU free memory after whole processing is %.0f MBites\n", (float)freeMem / 1048576.0f);
	end = clock();
	deconRecords[21] = (float)(end - start) / CLOCKS_PER_SEC;
	return 0;	
}

#undef blockSize
#undef blockSize2Dx
#undef blockSize2Dy
#undef blockSize3Dx
#undef blockSize3Dy
#undef blockSize3Dz
#undef SMALLVALUE
#undef NDIM
