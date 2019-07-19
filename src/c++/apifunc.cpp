#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>       // va_*
#include <iostream>
#include <fstream>
#include <string.h>
#include <math.h>
#include <ctime>
#include <time.h>
#include <sys/stat.h>
#ifdef _WIN32
#include <windows.h>
#include <tchar.h>
#include <strsafe.h>
//#else
//#include <sys/stat.h>
#endif

// Includes CUDA
//#include <cuda.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <memory>
#include "device_launch_parameters.h"

#include "tiff.h"
#include "tiffio.h"

#define NRANSI
extern "C"{
#include "nr.h"
#include "nrutil.h"
}

#include "libapi.h"
#define blockSize 1024
#define blockSize2Dx 32
#define blockSize2Dy 32
#define blockSize3Dx 16
#define blockSize3Dy 8
#define blockSize3Dz 8
#define NDIM 12
#define SMALLVALUE 0.01


cudaError_t __err = cudaGetLastError();
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
//concatenate any number of strings

char* concat(int count, ...)
{

	va_list ap;
	int i;

	// Find required length to store merged string
	int len = 1; // room for NULL
	va_start(ap, count);
	for (i = 0; i<count; i++)
		len += strlen(va_arg(ap, char*));
	va_end(ap);

	// Allocate memory to concat strings
	char *merged = (char*)calloc(sizeof(char), len);
	int null_pos = 0;

	// Actually concatenate strings
	va_start(ap, count);
	for (i = 0; i<count; i++)
	{
		char *s = va_arg(ap, char*);
		strcpy(merged + null_pos, s);
		null_pos += strlen(s);
	}
	va_end(ap);


	return merged;
}

//check file exists or not
bool fexists(const char * filename){
	if (FILE * file = fopen(filename, "r")) {
		fclose(file);
		return true;
	}
	return false;
}

#ifdef _WIN32
int findSubFolders(char *subFolderNames, char *pathIn)
{
	TCHAR szDir[MAX_PATH];
	StringCchCopy(szDir, MAX_PATH, pathIn);
	StringCchCat(szDir, MAX_PATH, TEXT("\\*"));
	WIN32_FIND_DATA fd;

	HANDLE hFile = FindFirstFile((char*)szDir, &fd);
	int i = 0;
	if (hFile != INVALID_HANDLE_VALUE){
		do
		{
			if ((*(char*)fd.cFileName == '.') || (*(char*)fd.cFileName == '..'))
				continue;
			strcpy(&subFolderNames[i*MAX_PATH], fd.cFileName);
			i++;
		} while (FindNextFile(hFile, &fd));
	}
		
	return i;
}
#endif

unsigned short gettifinfo(char tifdir[], unsigned int *tifSize){
	TIFF *tif = TIFFOpen(tifdir, "r");
	uint16 bitPerSample;
	TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &tifSize[0]);           // uint32 width;
	TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &tifSize[1]);        // uint32 height;
	TIFFGetField(tif, TIFFTAG_BITSPERSAMPLE, &bitPerSample);
	/// Get slice number
	int nSlice = 0;
	if (tif){
		do{
			nSlice++;
		} while (TIFFReadDirectory(tif));
	}
	tifSize[2] = nSlice;
	(void)TIFFClose(tif);
	return bitPerSample;
}

// Read tiff image
void readtifstack(float *h_Image, char tifdir[], unsigned int *imsize){
	TIFF *tif = TIFFOpen(tifdir, "r");
	uint16 bitPerSample;
	/// Get slice number
	int nSlice = 0;
	if (tif){
		do{
			nSlice++;
		} while (TIFFReadDirectory(tif));
	}
	imsize[2] = nSlice;
	(void)TIFFClose(tif);

	tif = TIFFOpen(tifdir, "r");
	TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &imsize[0]);           // uint32 width;
	TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &imsize[1]);        // uint32 height;
	TIFFGetField(tif, TIFFTAG_BITSPERSAMPLE, &bitPerSample);        // uint16 bps;

	uint32 nByte = bitPerSample / 8;
	if (bitPerSample == 16){
		uint16 *buf = (uint16 *)_TIFFmalloc(imsize[0] * imsize[1] * imsize[2] * nByte);
		if (tif){
			uint32 n = 0; // slice number
			do{
				for (uint32 row = 0; row < imsize[1]; row++){
					TIFFReadScanline(tif, &buf[row*imsize[0] + n*imsize[0] * imsize[1]], row, 0);
				}
				n++;
			} while (TIFFReadDirectory(tif));
		}
		(void)TIFFClose(tif);

		for (uint32 i = 0; i < imsize[0] * imsize[1] * imsize[2]; i++){
			h_Image[i] = (float)buf[i];
		}
		_TIFFfree(buf);
	}
	else if (bitPerSample == 32){
		if (tif){
			uint32 n = 0; // slice number
			do{
				for (uint32 row = 0; row < imsize[1]; row++){// possible to read in floating 32bit tiff images
					TIFFReadScanline(tif, &h_Image[row*imsize[0] + n*imsize[0] * imsize[1]], row, 0);
				}
				n++;
			} while (TIFFReadDirectory(tif));
		}
		(void)TIFFClose(tif);
	}
}

void readtifstack_16to16(unsigned short *h_Image, char tifdir[], unsigned int *imsize){
	TIFF *tif = TIFFOpen(tifdir, "r");
	uint16 bitPerSample;
	/// Get slice number
	int nSlice = 0;
	if (tif){
		do{
			nSlice++;
		} while (TIFFReadDirectory(tif));
	}
	imsize[2] = nSlice;
	(void)TIFFClose(tif);

	tif = TIFFOpen(tifdir, "r");
	TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &imsize[0]);           // uint32 width;
	TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &imsize[1]);        // uint32 height;
	TIFFGetField(tif, TIFFTAG_BITSPERSAMPLE, &bitPerSample);        // uint16 bps;

	if (bitPerSample == 16){
		if (tif){
			uint32 n = 0; // slice number
			do{
				for (uint32 row = 0; row < imsize[1]; row++){
					TIFFReadScanline(tif, &h_Image[row*imsize[0] + n*imsize[0] * imsize[1]], row, 0);
				}
				n++;
			} while (TIFFReadDirectory(tif));
		}
		(void)TIFFClose(tif);

		
	}
	else
		printf("Image bit per sample is not supported, please set input image as 16 bit!!!\n\n");
}

// Write tiff image
void writetifstack(char tifdir[], float *h_Image, unsigned int *imsize, unsigned short bitPerSample){
	int imTotalSize = imsize[0] * imsize[1] * imsize[2];
	uint32 imxy = imsize[0] * imsize[1];
	uint32 nByte = (uint32)(bitPerSample / 8);

	if (bitPerSample == 16){
		uint16 *buf = (uint16 *)_TIFFmalloc(imTotalSize * sizeof(uint16));
		for (int i = 0; i < imTotalSize; i++){
			buf[i] = (uint16)h_Image[i];
		}

		TIFF *tif = TIFFOpen(tifdir, "w");
		for (uint32 n = 0; n < imsize[2]; n++){
			TIFFSetField(tif, TIFFTAG_IMAGEWIDTH, imsize[0]);  // set the width of the image
			TIFFSetField(tif, TIFFTAG_IMAGELENGTH, imsize[1]);    // set the height of the image
			TIFFSetField(tif, TIFFTAG_SAMPLESPERPIXEL, 1);   // set number of channels per pixel
			TIFFSetField(tif, TIFFTAG_BITSPERSAMPLE, bitPerSample);    // set the size of the channels
			TIFFSetField(tif, TIFFTAG_ORIENTATION, (int)ORIENTATION_TOPLEFT);    // set the origin of the image.
			TIFFSetField(tif, TIFFTAG_PLANARCONFIG, PLANARCONFIG_SEPARATE);
			TIFFSetField(tif, TIFFTAG_COMPRESSION, COMPRESSION_NONE);
			//   Some other essential fields to set that you do not have to understand for now.
			TIFFSetField(tif, TIFFTAG_ROWSPERSTRIP, imsize[1]);
			TIFFSetField(tif, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_MINISBLACK);
			TIFFWriteEncodedStrip(tif, 0, &buf[imxy * n], imxy * nByte);
			TIFFWriteDirectory(tif);
		}
		(void)TIFFClose(tif);
		_TIFFfree(buf);
	}
	else if (bitPerSample == 32){
		TIFF *tif = TIFFOpen(tifdir, "w");
		for (uint32 n = 0; n < imsize[2]; n++){

			TIFFSetField(tif, TIFFTAG_IMAGEWIDTH, imsize[0]);  // set the width of the image
			TIFFSetField(tif, TIFFTAG_IMAGELENGTH, imsize[1]);    // set the height of the image
			TIFFSetField(tif, TIFFTAG_SAMPLESPERPIXEL, 1);   // set number of channels per pixel
			TIFFSetField(tif, TIFFTAG_BITSPERSAMPLE, bitPerSample);    // set the size of the channels
			TIFFSetField(tif, TIFFTAG_SAMPLEFORMAT, SAMPLEFORMAT_IEEEFP);    // ***set each pixel as floating point data ****
			TIFFSetField(tif, TIFFTAG_ORIENTATION, (int)ORIENTATION_TOPLEFT);    // set the origin of the image.
			TIFFSetField(tif, TIFFTAG_PLANARCONFIG, PLANARCONFIG_SEPARATE);
			TIFFSetField(tif, TIFFTAG_COMPRESSION, COMPRESSION_NONE);
			//   Some other essential fields to set that you do not have to understand for now.
			TIFFSetField(tif, TIFFTAG_ROWSPERSTRIP, imsize[1]);
			TIFFSetField(tif, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_MINISBLACK);

			TIFFWriteEncodedStrip(tif, 0, &h_Image[imxy * n], imxy * nByte);
			TIFFWriteDirectory(tif);
		}
		(void)TIFFClose(tif);
	}
	else
		printf("Image bit per sample is not supported, please set bitPerPample to 16 or 32 !!!\n\n");
}


void writetifstack_16to16(char tifdir[], unsigned short *h_Image, unsigned int *imsize){
	int imTotalSize = imsize[0] * imsize[1] * imsize[2];
	uint32 imxy = imsize[0] * imsize[1];
	uint32 nByte = (uint32)(16 / 8);
	TIFF *tif = TIFFOpen(tifdir, "w");
	for (uint32 n = 0; n < imsize[2]; n++){
		TIFFSetField(tif, TIFFTAG_IMAGEWIDTH, imsize[0]);  // set the width of the image
		TIFFSetField(tif, TIFFTAG_IMAGELENGTH, imsize[1]);    // set the height of the image
		TIFFSetField(tif, TIFFTAG_SAMPLESPERPIXEL, 1);   // set number of channels per pixel
		TIFFSetField(tif, TIFFTAG_BITSPERSAMPLE, 16);    // set the size of the channels
		TIFFSetField(tif, TIFFTAG_ORIENTATION, (int)ORIENTATION_TOPLEFT);    // set the origin of the image.
		TIFFSetField(tif, TIFFTAG_PLANARCONFIG, PLANARCONFIG_SEPARATE);
		TIFFSetField(tif, TIFFTAG_COMPRESSION, COMPRESSION_NONE);
		//   Some other essential fields to set that you do not have to understand for now.
		TIFFSetField(tif, TIFFTAG_ROWSPERSTRIP, imsize[1]);
		TIFFSetField(tif, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_MINISBLACK);
		TIFFWriteEncodedStrip(tif, 0, &h_Image[imxy * n], imxy * nByte);
		TIFFWriteDirectory(tif);
	}
	(void)TIFFClose(tif);
}

void queryDevice(){
	printf(" \n ===========================================\n");
	int deviceCount = 0;
	cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

	if (error_id != cudaSuccess)
	{
		printf("cudaGetDeviceCount returned %d\n-> %s\n", (int)error_id, cudaGetErrorString(error_id));
		printf("Result = FAIL\n");
		exit(EXIT_FAILURE);
	}

	// This function call returns 0 if there are no CUDA capable devices.
	if (deviceCount == 0)
	{
		printf("There are no available device(s) that support CUDA\n");
	}
	else
	{
		printf("Detected %d CUDA Capable device(s)\n", deviceCount);
	}

	int dev, driverVersion = 0, runtimeVersion = 0;

	for (dev = 0; dev < deviceCount; ++dev)
	{
		cudaSetDevice(dev);
		cudaDeviceProp deviceProp;
		cudaGetDeviceProperties(&deviceProp, dev);
		printf("\nDevice %d: \"%s\"\n", dev, deviceProp.name);
		cudaDriverGetVersion(&driverVersion);
		cudaRuntimeGetVersion(&runtimeVersion);
		printf("  CUDA Driver Version / Runtime Version          %d.%d / %d.%d\n", driverVersion / 1000, (driverVersion % 100) / 10, runtimeVersion / 1000, (runtimeVersion % 100) / 10);
		printf("  CUDA Capability Major/Minor version number:    %d.%d\n", deviceProp.major, deviceProp.minor);

		printf("  Total amount of global memory:                 %.0f MBytes (%llu bytes)\n",
			(float)deviceProp.totalGlobalMem / 1048576.0f, (unsigned long long) deviceProp.totalGlobalMem);

		printf("  Total amount of constant memory:               %lu bytes\n", deviceProp.totalConstMem);
		printf("  Total amount of shared memory per block:       %lu bytes\n", deviceProp.sharedMemPerBlock);
		printf("  Total number of registers available per block: %d\n", deviceProp.regsPerBlock);
		printf("  Warp size:                                     %d\n", deviceProp.warpSize);
		printf("  Maximum number of threads per multiprocessor:  %d\n", deviceProp.maxThreadsPerMultiProcessor);
		printf("  Maximum number of threads per block:           %d\n", deviceProp.maxThreadsPerBlock);
		printf("  Max dimension size of a thread block (x,y,z): (%d, %d, %d)\n",
			deviceProp.maxThreadsDim[0],
			deviceProp.maxThreadsDim[1],
			deviceProp.maxThreadsDim[2]);
		printf("  Max dimension size of a grid size    (x,y,z): (%d, %d, %d)\n",
			deviceProp.maxGridSize[0],
			deviceProp.maxGridSize[1],
			deviceProp.maxGridSize[2]);
		printf("  Maximum memory pitch:                          %lu bytes\n", deviceProp.memPitch);
		printf("  Texture alignment:                             %lu bytes\n", deviceProp.textureAlignment);
		printf("  Concurrent copy and kernel execution:          %s with %d copy engine(s)\n", (deviceProp.deviceOverlap ? "Yes" : "No"), deviceProp.asyncEngineCount);
		printf("  Run time limit on kernels:                     %s\n", deviceProp.kernelExecTimeoutEnabled ? "Yes" : "No");
		printf("  Integrated GPU sharing Host Memory:            %s\n", deviceProp.integrated ? "Yes" : "No");
		printf("  Support host page-locked memory mapping:       %s\n", deviceProp.canMapHostMemory ? "Yes" : "No");
		printf("  Alignment requirement for Surfaces:            %s\n", deviceProp.surfaceAlignment ? "Yes" : "No");
		printf("  Device has ECC support:                        %s\n", deviceProp.ECCEnabled ? "Enabled" : "Disabled");
	}

	printf(" ===========================================\n\n");

}

////// ***************** 3D registration ****
static float *h_s3D = NULL, *h_t3D = NULL, *d_img3D = NULL; // , *d_img2 = NULL, *d_imgE = NULL;
static double *d_2D = NULL, *h_2D = NULL, *d_2D2 = NULL, *d_2D3 = NULL;
static double corrDenominator;
static int imx1, imy1, imz1, imx2, imy2, imz2;
static float *h_aff;
static int totalItNum, dofNum;
static bool dof9Flag;

float costfunc(float *x){
	if (dof9Flag){
		dof9tomatrix(h_aff, x, dofNum);
	}
	else{
		p2matrix(h_aff, x);
	}
	/*
	double costValue = corrfunc(d_img3D, // target image
	d_img2, // source image
	d_imgE,
	h_aff,
	d_2D,
	h_2D,
	imx1,
	imy1,
	imz1,
	imx2,
	imy2,
	imz2
	);
	*/

	// optimized correlation function
	/*
	double costValue = corrfunc2(d_img3D, // target image
	h_aff,
	d_2D,
	d_2D2,
	h_2D,
	imx1,
	imy1,
	imz1,
	imx2,
	imy2,
	imz2
	);
	*/

	double costValue = corrfunc3(d_img3D, // target image
		h_aff,
		d_2D,
		d_2D2,
		d_2D3,
		h_2D,
		imx1,
		imy1,
		imz1,
		imx2,
		imy2,
		imz2
		);

	totalItNum += 1;
	return (float)(-costValue / corrDenominator);
}

float costfunccpu(float *x){
	if (dof9Flag){
		dof9tomatrix(h_aff, x, dofNum);
	}
	else{
		p2matrix(h_aff, x);
	}

	double costValue = corrfunccpu2(h_s3D, // source image
		h_t3D,
		h_aff,
		imx1,
		imy1,
		imz1,
		imx2,
		imy2,
		imz2
		);

	totalItNum += 1;
	return (float)(-costValue / corrDenominator);
}

//extern float TOL = 0.01; // global variable for linmin.c

////// ***************** 2D registration ****
static float *h_aff2D;
static double corrDenominator2D;
static float *d_img2D, *d_img2DE, *d_img2DT, *h_2Dreg;
static int imx2D1, imy2D1, imx2D2, imy2D2;
static int totalItNum2D;

float costfunc2D(float *x){
	h_aff2D[0] = x[1], h_aff2D[1] = x[2], h_aff2D[2] = x[3];
	h_aff2D[3] = x[4], h_aff2D[4] = x[5], h_aff2D[5] = x[6];
	double costValue = corrfunc2D(d_img2D, // source stack
		d_img2DE,
		d_img2DT,
		h_aff2D,
		h_2Dreg,
		imx2D1,
		imy2D1,
		imx2D2,
		imy2D2);
	totalItNum2D += 1;
	return (float)(-costValue / corrDenominator2D);
}

int reg_2dgpu(float *h_reg, float *iTmx, float *h_img1, float *h_img2, int imSizex1, int imSizey1, int imSizex2, int imSizey2,
	int inputTmx, float FTOL, int itLimit, int deviceNum, float *regRecords){
	//regRecords: 9-element array 
	//[0] -[3]: initial GPU memory, after variables allocated, after processing, after variables released ( all in MB)
	//[4] -[5]: initial cost function value, minimized cost function value
	//[6] -[8]: registration time (in s), whole time (in s), total sub iterations

	imx2D1 = imSizex1; imy2D1 = imSizey1;
	imx2D2 = imSizex2; imy2D2 = imSizey2;

	// total pixel count for each images
	int totalSize1 = imx2D1*imy2D1;
	int totalSize2 = imx2D2*imy2D2;
	int totalSizeMax = (totalSize1 > totalSize2) ? totalSize1 : totalSize2;
	// print GPU devices information
	cudaSetDevice(deviceNum);

	//****************** Processing Starts*****************//
	// variables for memory and time cost records
	clock_t start, startWhole, end, endWhole;
	size_t totalMem = 0;
	size_t freeMem = 0;
	cudaMemGetInfo(&freeMem, &totalMem);
	regRecords[0] = (float)freeMem / 1048576.0f;

	startWhole = clock();
	int iter;
	float fret;
	int DIM2D = 6;
	h_aff2D = (float *)malloc(DIM2D * sizeof(float));
	static float *p2D = (float *)malloc((DIM2D+1) * sizeof(float));
	float **xi2D;
	xi2D = matrix(1, DIM2D, 1, DIM2D);

	h_2Dreg = (float *)malloc(totalSizeMax * sizeof(float));
	cudaMalloc((void **)&d_img2D, totalSize1*sizeof(float));
	cudaMalloc((void **)&d_img2DE, totalSize1*sizeof(float));
	cudaMalloc((void **)&d_img2DT, totalSize1*sizeof(float));
	cudaCheckErrors("****Memory allocating fails... GPU out of memory !!!!*****\n");

	cudaChannelFormatDesc channelDesc2D =
		cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
	cudaArray *d_Array2D;
	cudaMallocArray(&d_Array2D, &channelDesc2D, imx2D2, imy2D2);
	cudaCheckErrors("****Memory array allocating fails... GPU out of memory !!!!*****\n");
	cudaMemGetInfo(&freeMem, &totalMem);
	regRecords[1] = (float)freeMem / 1048576.0f;


	start = clock();
	if (inputTmx){
		memcpy(h_aff2D, iTmx, DIM2D * sizeof(float));
	}
	else{
		h_aff2D[0] = 1, h_aff2D[1] = 0, h_aff2D[2] = (imx2D2 - imx2D1) / 2;
		h_aff2D[3] = 0, h_aff2D[4] = 1, h_aff2D[5] = (imy2D2 - imy2D1) / 2;
	}
	p2D[0] = 0;
	p2D[1] = h_aff2D[0], p2D[2] = h_aff2D[1], p2D[3] = h_aff2D[2];
	p2D[4] = h_aff2D[3], p2D[5] = h_aff2D[4], p2D[6] = h_aff2D[5];
	for (int i = 1; i <= DIM2D; i++)
		for (int j = 1; j <= DIM2D; j++)
			xi2D[i][j] = (i == j ? 1.0 : 0.0);

	multicpu(h_2Dreg, h_img1, h_img1, totalSize1);
	double sumSqrA = sumcpu(h_2Dreg, totalSize1);
	corrDenominator2D = sqrt(sumSqrA);

	cudaMemcpy(d_img2D, h_img1, totalSize1*sizeof(float), cudaMemcpyHostToDevice);
	cudacopyhosttoarray2D(d_Array2D, channelDesc2D, h_img2, totalSize2);
	BindTexture2D(d_Array2D, channelDesc2D);
	cudaCheckErrors("****Fail to bind 2D texture!!!!*****\n");
	totalItNum2D = 0;
	regRecords[4] = costfunc2D(p2D);
	powell(p2D, xi2D, DIM2D, FTOL, &iter, &fret, costfunc2D, &totalItNum, itLimit);
	affineTransform2D(d_img2D, imx2D1, imy2D1, imx2D2, imy2D2);
	memcpy(iTmx, h_aff2D, DIM2D * sizeof(float));
	cudaMemcpy(h_reg, d_img2D, totalSize1*sizeof(float), cudaMemcpyDeviceToHost);

	end = clock();
	cudaMemGetInfo(&freeMem, &totalMem);
	regRecords[2] = (float)freeMem / 1048576.0f;
	regRecords[5] = fret;
	regRecords[6] = (float)(end - start) / CLOCKS_PER_SEC;
	regRecords[8] = totalItNum;

	free(h_2Dreg);
	//free(p2D);
	free(h_aff2D);
	free_matrix(xi2D, 1, DIM2D, 1, DIM2D);
	cudaFree(d_img2D);
	cudaFree(d_img2DE);
	cudaFree(d_img2DT);
	cudaFreeArray(d_Array2D);


	cudaMemGetInfo(&freeMem, &totalMem);
	regRecords[3] = (float)freeMem / 1048576.0f;
	endWhole = clock();
	regRecords[7] = (float)(endWhole - startWhole) / CLOCKS_PER_SEC;
	return 0;
}

int reg_2dshiftaligngpu(float *h_reg, float *iTmx, float *h_img1, float *h_img2, int imSizex1, int imSizey1, int imSizex2, int imSizey2,
	int inputTmx, float shiftRegion, int totalStep, int deviceNum, float *regRecords){
	//regRecords: 9-element array 
	//[0] -[3]: initial GPU memory, after variables allocated, after processing, after variables released ( all in MB)
	//[4] -[5]: initial cost function value, minimized cost function value
	//[6] -[8]: registration time (in s), whole time (in s), total sub iterations
	imx2D1 = imSizex1; imy2D1 = imSizey1;
	imx2D2 = imSizex2; imy2D2 = imSizey2;

	// total pixel count for each images
	int totalSize1 = imx2D1*imy2D1;
	int totalSize2 = imx2D2*imy2D2;
	int totalSizeMax = (totalSize1 > totalSize2) ? totalSize1 : totalSize2;
	// print GPU devices information
	cudaSetDevice(deviceNum);

	//****************** Processing Starts*****************//
	// variables for memory and time cost records
	clock_t start, startWhole, end, endWhole;
	size_t totalMem = 0;
	size_t freeMem = 0;
	cudaMemGetInfo(&freeMem, &totalMem);
	regRecords[0] = (float)freeMem / 1048576.0f;

	startWhole = clock();
	float fret;
	int DIM2D = 6;
	h_aff2D = (float *)malloc(DIM2D * sizeof(float));
	static float *p2D = (float *)malloc((DIM2D + 1) * sizeof(float));
	float **xi2D;
	xi2D = matrix(1, DIM2D, 1, DIM2D);

	float shiftX, shiftY, offSetX, offSetY;
	float costValue2D, costValue2DXYmin;
	//float totalStep = 40;
	//float shiftRegion = 0.6; //0< shiftRegion <1
	float stepSizex, stepSizey;



	h_2Dreg = (float *)malloc(totalSizeMax * sizeof(float));
	cudaMalloc((void **)&d_img2D, totalSize1*sizeof(float));
	cudaMalloc((void **)&d_img2DE, totalSize1*sizeof(float));
	cudaMalloc((void **)&d_img2DT, totalSize1*sizeof(float));
	cudaCheckErrors("****Memory allocating fails... GPU out of memory !!!!*****\n");

	cudaChannelFormatDesc channelDesc2D =
		cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
	cudaArray *d_Array2D;
	cudaMallocArray(&d_Array2D, &channelDesc2D, imx2D2, imy2D2);
	cudaCheckErrors("****Memory array allocating fails... GPU out of memory !!!!*****\n");
	cudaMemGetInfo(&freeMem, &totalMem);
	regRecords[1] = (float)freeMem / 1048576.0f;


	start = clock();
	if (inputTmx){
		memcpy(h_aff2D, iTmx, DIM2D * sizeof(float));
	}
	else{
		h_aff2D[0] = 1, h_aff2D[1] = 0, h_aff2D[2] = (imx2D2 - imx2D1) / 2;
		h_aff2D[3] = 0, h_aff2D[4] = 1, h_aff2D[5] = (imy2D2 - imy2D1) / 2;
	}
	p2D[0] = 0;
	p2D[1] = h_aff2D[0], p2D[2] = h_aff2D[1], p2D[3] = h_aff2D[2];
	p2D[4] = h_aff2D[3], p2D[5] = h_aff2D[4], p2D[6] = h_aff2D[5];
	for (int i = 1; i <= DIM2D; i++)
		for (int j = 1; j <= DIM2D; j++)
			xi2D[i][j] = (i == j ? 1.0 : 0.0);

	multicpu(h_2Dreg, h_img1, h_img1, totalSize1);
	double sumSqrA = sumcpu(h_2Dreg, totalSize1);
	corrDenominator2D = sqrt(sumSqrA);

	cudaMemcpy(d_img2D, h_img1, totalSize1*sizeof(float), cudaMemcpyHostToDevice);
	cudacopyhosttoarray2D(d_Array2D, channelDesc2D, h_img2, totalSize2);
	BindTexture2D(d_Array2D, channelDesc2D);
	cudaCheckErrors("****Fail to bind 2D texture!!!!*****\n");
	totalItNum2D = 0;
	regRecords[4] = costfunc2D(p2D);
	
	shiftX = 0; shiftY = 0;
	offSetX = h_aff2D[2];  offSetY = h_aff2D[5];
	//*** translate step by step
	costValue2DXYmin = 0;
	stepSizex = imx2D2 * shiftRegion / totalStep;
	stepSizey = imy2D2 * shiftRegion / totalStep;
	for (int i = -totalStep; i < totalStep; i++){
		p2D[3] = offSetX + stepSizex * i;
		for (int j = -totalStep; j < totalStep; j++){
			p2D[6] = offSetY + stepSizey * j;
			costValue2D = costfunc2D(p2D);
			if (costValue2D < costValue2DXYmin){
				costValue2DXYmin = costValue2D;
				shiftX = h_aff2D[2];
				shiftY = h_aff2D[5];
			}
		}
	}
	p2D[3] = shiftX;
	p2D[6] = shiftY;
	fret = costfunc2D(p2D);
	affineTransform2D(d_img2D, imx2D1, imy2D1, imx2D2, imy2D2);
	memcpy(iTmx, h_aff2D, DIM2D * sizeof(float));
	cudaMemcpy(h_reg, d_img2D, totalSize1*sizeof(float), cudaMemcpyDeviceToHost);

	end = clock();
	cudaMemGetInfo(&freeMem, &totalMem);
	regRecords[2] = (float)freeMem / 1048576.0f;
	regRecords[5] = fret;
	regRecords[6] = (float)(end - start) / CLOCKS_PER_SEC;
	regRecords[8] = (2*totalStep+1)^2;

	free(h_2Dreg);
	//free(p2D);
	free(h_aff2D);
	free_matrix(xi2D, 1, DIM2D, 1, DIM2D);
	cudaFree(d_img2D);
	cudaFree(d_img2DE);
	cudaFree(d_img2DT);
	cudaFreeArray(d_Array2D);


	cudaMemGetInfo(&freeMem, &totalMem);
	regRecords[3] = (float)freeMem / 1048576.0f;
	endWhole = clock();
	regRecords[7] = (float)(endWhole - startWhole) / CLOCKS_PER_SEC;
	return 0;
}

int reg_2dshiftalignXgpu(float *h_reg, float *iTmx, float *h_img1, float *h_img2, int imSizex1, int imSizey1, int imSizex2, int imSizey2,
	int inputTmx, float shiftRegion, int totalStep, int deviceNum, float *regRecords){
	//regRecords: 9-element array 
	//[0] -[3]: initial GPU memory, after variables allocated, after processing, after variables released ( all in MB)
	//[4] -[5]: initial cost function value, minimized cost function value
	//[6] -[8]: registration time (in s), whole time (in s), total sub iterations
	imx2D1 = imSizex1; imy2D1 = imSizey1;
	imx2D2 = imSizex2; imy2D2 = imSizey2;

	// total pixel count for each images
	int totalSize1 = imx2D1*imy2D1;
	int totalSize2 = imx2D2*imy2D2;
	int totalSizeMax = (totalSize1 > totalSize2) ? totalSize1 : totalSize2;
	// print GPU devices information
	cudaSetDevice(deviceNum);

	//****************** Processing Starts*****************//
	// variables for memory and time cost records
	clock_t start, startWhole, end, endWhole;
	size_t totalMem = 0;
	size_t freeMem = 0;
	cudaMemGetInfo(&freeMem, &totalMem);
	regRecords[0] = (float)freeMem / 1048576.0f;

	startWhole = clock();
	float fret;
	int DIM2D = 6;
	h_aff2D = (float *)malloc(DIM2D * sizeof(float));
	static float *p2D = (float *)malloc((DIM2D + 1) * sizeof(float));
	float **xi2D;
	xi2D = matrix(1, DIM2D, 1, DIM2D);

	float shiftX, offSetX;
	float costValue2D, costValue2DXYmin;
	//float totalStep = 40;
	//float shiftRegion = 0.6; //0< shiftRegion <1
	float stepSizex;



	h_2Dreg = (float *)malloc(totalSizeMax * sizeof(float));
	cudaMalloc((void **)&d_img2D, totalSize1*sizeof(float));
	cudaMalloc((void **)&d_img2DE, totalSize1*sizeof(float));
	cudaMalloc((void **)&d_img2DT, totalSize1*sizeof(float));
	cudaCheckErrors("****Memory allocating fails... GPU out of memory !!!!*****\n");

	cudaChannelFormatDesc channelDesc2D =
		cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
	cudaArray *d_Array2D;
	cudaMallocArray(&d_Array2D, &channelDesc2D, imx2D2, imy2D2);
	cudaCheckErrors("****Memory array allocating fails... GPU out of memory !!!!*****\n");
	cudaMemGetInfo(&freeMem, &totalMem);
	regRecords[1] = (float)freeMem / 1048576.0f;


	start = clock();
	if (inputTmx){
		memcpy(h_aff2D, iTmx, DIM2D * sizeof(float));
	}
	else{
		h_aff2D[0] = 1, h_aff2D[1] = 0, h_aff2D[2] = (imx2D2 - imx2D1) / 2;
		h_aff2D[3] = 0, h_aff2D[4] = 1, h_aff2D[5] = (imy2D2 - imy2D1) / 2;
	}
	p2D[0] = 0;
	p2D[1] = h_aff2D[0], p2D[2] = h_aff2D[1], p2D[3] = h_aff2D[2];
	p2D[4] = h_aff2D[3], p2D[5] = h_aff2D[4], p2D[6] = h_aff2D[5];
	for (int i = 1; i <= DIM2D; i++)
		for (int j = 1; j <= DIM2D; j++)
			xi2D[i][j] = (i == j ? 1.0 : 0.0);

	multicpu(h_2Dreg, h_img1, h_img1, totalSize1);
	double sumSqrA = sumcpu(h_2Dreg, totalSize1);
	corrDenominator2D = sqrt(sumSqrA);

	cudaMemcpy(d_img2D, h_img1, totalSize1*sizeof(float), cudaMemcpyHostToDevice);
	cudacopyhosttoarray2D(d_Array2D, channelDesc2D, h_img2, totalSize2);
	BindTexture2D(d_Array2D, channelDesc2D);
	cudaCheckErrors("****Fail to bind 2D texture!!!!*****\n");
	totalItNum2D = 0;
	regRecords[4] = costfunc2D(p2D);

	shiftX = 0; 
	offSetX = h_aff2D[2]; 
	//*** translate step by step
	costValue2DXYmin = 0;
	stepSizex = imx2D2 * shiftRegion / totalStep;
	for (int i = -totalStep; i < totalStep; i++){
		for (int j = -totalStep; j < totalStep; j++){
			p2D[3] = offSetX + stepSizex * i;
			costValue2D = costfunc2D(p2D);
			if (costValue2D < costValue2DXYmin){
				costValue2DXYmin = costValue2D;
				shiftX = h_aff2D[2];
			}
		}
	}
	p2D[3] = shiftX;
	fret = costfunc2D(p2D);
	affineTransform2D(d_img2D, imx2D1, imy2D1, imx2D2, imy2D2);
	memcpy(iTmx, h_aff2D, DIM2D * sizeof(float));
	cudaMemcpy(h_reg, d_img2D, totalSize1*sizeof(float), cudaMemcpyDeviceToHost);

	end = clock();
	cudaMemGetInfo(&freeMem, &totalMem);
	regRecords[2] = (float)freeMem / 1048576.0f;
	regRecords[5] = fret;
	regRecords[6] = (float)(end - start) / CLOCKS_PER_SEC;
	regRecords[8] = 2 * totalStep + 1;

	free(h_2Dreg);
	//free(p2D);
	free(h_aff2D);
	free_matrix(xi2D, 1, DIM2D, 1, DIM2D);
	cudaFree(d_img2D);
	cudaFree(d_img2DE);
	cudaFree(d_img2DT);
	cudaFreeArray(d_Array2D);


	cudaMemGetInfo(&freeMem, &totalMem);
	regRecords[3] = (float)freeMem / 1048576.0f;
	endWhole = clock();
	regRecords[7] = (float)(endWhole - startWhole) / CLOCKS_PER_SEC;
	return 0;
}


int affinetrans_3dgpu(float *h_reg, float *iTmx, float *h_img2, unsigned int *imSize1, unsigned int *imSize2, int deviceNum){
	//image size
	int sx1 = imSize1[0], sy1 = imSize1[1], sz1 = imSize1[2];
	int sx2 = imSize2[0], sy2 = imSize2[1], sz2 = imSize2[2];
	// total pixel count for each images
	int totalSize1 = sx1*sy1*sz1;
	int totalSize2 = sx2*sy2*sz2;
	// GPU device
	cudaSetDevice(deviceNum);
	float *d_img3DTemp;
	cudaMalloc((void **)&d_img3DTemp, totalSize1 *sizeof(float));
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
	cudaArray *d_ArrayTemp;
	cudaMalloc3DArray(&d_ArrayTemp, &channelDesc, make_cudaExtent(sx2, sy2, sz2));
	cudaDeviceSynchronize();
	cudaCheckErrors("****GPU array memory allocating fails... GPU out of memory !!!!*****\n");
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

int affinetrans_3dgpu_16to16(unsigned short *h_reg, float *iTmx, unsigned short *h_img2, unsigned int *imSize1, unsigned int *imSize2, int deviceNum){
	//image size
	imx1 = imSize1[0]; imy1 = imSize1[1]; imz1 = imSize1[2];
	imx2 = imSize2[0]; imy2 = imSize2[1]; imz2 = imSize2[2];
	// total pixel count for each images
	int totalSize1 = imx1*imy1*imz1;
	int totalSize2 = imx2*imy2*imz2;
	// GPU device
	unsigned short *d_img3D16;
	cudaSetDevice(deviceNum);
	cudaMalloc((void **)&d_img3D16, totalSize1 *sizeof(unsigned short));
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<unsigned short>();
	cudaArray *d_Array;
	cudaMalloc3DArray(&d_Array, &channelDesc, make_cudaExtent(imx2, imy2, imz2));
	cudaDeviceSynchronize();
	cudaCheckErrors("****GPU array memory allocating fails... GPU out of memory !!!!*****\n");
	cudaMemset(d_img3D16, 0, totalSize1*sizeof(unsigned short));
	cudacopyhosttoarray(d_Array, channelDesc, h_img2, imx2, imy2, imz2);
	BindTexture16(d_Array, channelDesc);
	CopyTranMatrix(iTmx, NDIM * sizeof(float));
	affineTransform(d_img3D16, imx1, imy1, imz1, imx2, imy2, imz2);
	UnbindTexture16();
	cudaMemcpy(h_reg, d_img3D16, totalSize1 * sizeof(unsigned short), cudaMemcpyDeviceToHost);
	cudaFreeArray(d_Array);
	cudaFree(d_img3D16);
	return 0;
}

int reg_3dphasetransgpu(int *shiftXYZ, float *h_img1, float *h_img2, unsigned int *imSize1, unsigned int *imSize2, int downSample, int deviceNum, float *regRecords){
	//image size
	int sx1 = imSize1[0], sy1 = imSize1[1], sz1 = imSize1[2];
	int sx2 = imSize2[0], sy2 = imSize2[1], sz2 = imSize2[2];
	// print GPU devices information
	cudaSetDevice(deviceNum);
	//****************** Processing Starts*****************//
	// variables for memory and time cost records
	clock_t start, end;
	size_t totalMem = 0;
	size_t freeMem = 0;
	float gpuTimeCost = 0;
	start = clock();
	cudaMemGetInfo(&freeMem, &totalMem);
	regRecords[0] = (float)freeMem / 1048576.0f;

	// variables and FFT plans
	int sx3, sy3, sz3;
	sx3 = snapTransformSize(sx1 / downSample);
	sy3 = snapTransformSize(sy1 / downSample);
	sz3 = snapTransformSize(sz1 / downSample);
	//unsigned int imSizeTemp[3];
	//imSizeTemp[0] = sx3; imSizeTemp[1] = sy3; imSizeTemp[2] = sz3;
	int totalSize = sx3 * sy3 * sz3;
	int totalSizeSpectrum = sx3 * sy3*(sz3 / 2 + 1); // in complex floating format
	unsigned int imSize[3];
	imSize[0] = sx3; imSize[1] = sy3; imSize[2] = sz3;
	float *iTmx = (float *)malloc((NDIM + 4)* sizeof(float));
	for (int i = 0; i < 15; i++) iTmx[i] = 0; iTmx[15] = 1;
	float *h_img1_downSample = (float *)malloc(totalSize* sizeof(float));
	float *h_img2_downSample = (float *)malloc(totalSize* sizeof(float));
	float *d_downSample = NULL;
	fComplex *d_Spectrum1 = NULL, *d_Spectrum2 = NULL;
	cudaMalloc((void **)&d_downSample, totalSize *sizeof(float));
	cudaMalloc((void **)&d_Spectrum1, totalSizeSpectrum*sizeof(fComplex));
	cudaMalloc((void **)&d_Spectrum2, totalSizeSpectrum*sizeof(fComplex));
	// Create FFT plans
	cufftHandle
		fftPlanFwd,
		fftPlanInv;
	cufftPlan3d(&fftPlanFwd, sx3, sy3, sz3, CUFFT_R2C);
	cufftPlan3d(&fftPlanInv, sx3, sy3, sz3, CUFFT_C2R);
	cudaCheckErrors("****Memory allocating fails... GPU out of memory !!!!*****");
	cudaMemGetInfo(&freeMem, &totalMem);
	regRecords[1] = (float)freeMem / 1048576.0f;
	//**** downsample and adjust image size;
	iTmx[0] = iTmx[5] = iTmx[10] = (float)downSample;
	affinetrans_3dgpu(h_img1_downSample, iTmx, h_img1, &imSize[0], imSize1, deviceNum);
	//writetifstack("tranform1.tif", h_img1_downSample, &imSizeTemp[0], 16);
	affinetrans_3dgpu(h_img2_downSample, iTmx, h_img2, &imSize[0], imSize2, deviceNum);
	//writetifstack("tranform2.tif", h_img2_downSample, &imSizeTemp[0], 16);
	//*** FFT transform
	// image 2
	cudaMemcpy(d_downSample, h_img2_downSample, totalSize* sizeof(float), cudaMemcpyHostToDevice);
	cufftExecR2C(fftPlanFwd, (cufftReal *)d_downSample, (cufftComplex *)d_Spectrum1);
	conj3Dgpu(d_Spectrum2, d_Spectrum1, sx3, sy3, (sz3 / 2 + 1));
	// image 1
	cudaMemcpy(d_downSample, h_img1_downSample, totalSize* sizeof(float), cudaMemcpyHostToDevice);
	cufftExecR2C(fftPlanFwd, (cufftReal *)d_downSample, (cufftComplex *)d_Spectrum1);
	// multiplication
	multicomplexnorm3Dgpu(d_Spectrum2, d_Spectrum1, d_Spectrum2, sx3, sy3, (sz3 / 2 + 1));
	// iFFT
	cufftExecC2R(fftPlanInv, (cufftComplex *)d_Spectrum2, (cufftReal *)d_downSample);
	int corXYZ[3];
	float peakValue = max3Dgpu(&corXYZ[0], d_downSample, sx3, sy3, sz3);
	//printf("...peak value: %f;\n", peakValue);
	//printf("...phase translation, X: %d; Y: %d; Z: %d\n", corXYZ[0], corXYZ[1], corXYZ[2]);
	//cudaMemcpy(h_img1_downSample, d_downSample, totalSize* sizeof(float), cudaMemcpyDeviceToHost);
	//float peakValue = max3Dcpu(shiftXYZ, h_img1_downSample, FFTx, FFTy, FFTz);
	if (corXYZ[0] < sx3 / 4) shiftXYZ[0] = corXYZ[0] * downSample; 
	else if (corXYZ[0] < sx3 * 3 / 4) shiftXYZ[0] = (corXYZ[0] - sx3 / 2) * downSample;
	else shiftXYZ[0] = (corXYZ[0] - sx3) * downSample;
	if (corXYZ[1] < sy3 / 4) shiftXYZ[1] = corXYZ[1] * downSample;
	else if (corXYZ[1] < sy3 * 3 / 4) shiftXYZ[1] = (corXYZ[1] - sy3 / 2) * downSample;
	else shiftXYZ[1] = (corXYZ[1] - sy3) * downSample;
	if (corXYZ[2] < sz3 / 4) shiftXYZ[2] = corXYZ[2] * downSample;
	else if (corXYZ[2] < sz3 * 3 / 4) shiftXYZ[2] = (corXYZ[2] - sz3 / 2) * downSample;
	else shiftXYZ[2] = (corXYZ[2] - sz3) * downSample;

	//printf("...peak value: %f;\n", peakValue);
	//printf("...phase translation, X: %d; Y: %d; Z: %d\n", shiftXYZ[0], shiftXYZ[1], shiftXYZ[2]);
	cudaMemcpy(h_img1_downSample, d_downSample, totalSize* sizeof(float), cudaMemcpyDeviceToHost);
	//writetifstack("max.tif", h_img1_downSample, &imSizeTemp[0], 32);
	free(iTmx); free(h_img1_downSample); free(h_img2_downSample);
	cudaFree(d_downSample); cudaFree(d_Spectrum1); cudaFree(d_Spectrum2);
	cufftDestroy(fftPlanFwd);
	cufftDestroy(fftPlanInv);
	end = clock();
	cudaMemGetInfo(&freeMem, &totalMem);
	regRecords[2] = (float)freeMem / 1048576.0f;
	regRecords[3] = (float)(end - start) / CLOCKS_PER_SEC;
	return 0;
}

int reg_3dgpu(float *h_reg, float *iTmx, float *h_img1, float *h_img2, unsigned int *imSize1, unsigned int *imSize2, int regMethod,
	int inputTmx, float FTOL, int itLimit, int subBgTrigger, int deviceNum, float *regRecords){
	//registration method--> 0: no registration, transform imageB based on input matrix;1: translation only; 2: rigid body; 
	//  3: 7 degrees of freedom (translation, rotation, scaling equally in 3 dimensions)  4: 9 degrees of freedom(translation, rotation, scaling); 
	//  5: 12 degrees of freedom; 6:rigid body first, then do 12 degrees of freedom; 7:3 DOF --> 6 DOF --> 9 DOF --> 12 DOF
	// inputTmx: 0, default; 1: use input matrix; 2: do 3D translation registation based on phase registration. 3: 2D translation based (there is a bug with this option).
	//regRecords: 11 element array
	//[0] -[3]: initial GPU memory, after variables allocated, after processing, after variables released ( all in MB)
	//[4] -[6]: initial cost function value, minimized cost function value, intermediate cost function value
	//[7] -[10]: registration time (in s), whole time (in s), single sub iteration time (in ms), total sub iterations

	//************get basic input images information ******************	
	//image size
	imx1 = imSize1[0]; imy1 = imSize1[1]; imz1 = imSize1[2];
	imx2 = imSize2[0]; imy2 = imSize2[1]; imz2 = imSize2[2];
	// total pixel count for each images
	int totalSize1 = imx1*imy1*imz1;
	int totalSize2 = imx2*imy2*imz2;
	int totalSizeMax = (totalSize1 > totalSize2) ? totalSize1 : totalSize2;
	// print GPU devices information
	cudaSetDevice(deviceNum);
	//****************** Processing Starts*****************//
	// variables for memory and time cost records
	clock_t start, startTemp, startWhole, end, endTemp, endWhole;
	size_t totalMem = 0;
	size_t freeMem = 0;
	float gpuTimeCost = 0;
	cudaMemGetInfo(&freeMem, &totalMem);
	regRecords[0] = (float)freeMem / 1048576.0f;

	startWhole = clock();
	/*
	// write records
	freopen("log.txt", "w", stdout);
	printf("3D registration: calling CUDA DLL....\n");
	printf("GPU device: #%d \n", deviceNum);
	printf("...GPU free memory(at beginning) is %.0f MBites\n", (float)freeMem / 1048576.0f);
	printf("\nImage information:\n");
	printf("...Target size %d x %d x %d\n", imx1, imy1, imz1);
	printf("...Source size %d x %d x %d\n", imx2, imy2, imz2);
	printf("...Registered image size %d x %d x %d\n", imx1, imy1, imz1);
	printf("\nRegistration setup:\n");
	if (regTrigger)
	printf("...Perform registration\n");
	else
	printf("...No registration\n");
	switch (regMethod){
	case 1:
	printf("...Registration method: translation only\n"); break;
	case 2:
	printf("...Registration method: rigid body\n"); break;
	case 3:
	printf("...Registration method: 7 degrees of freedom(translation, rotation, scaling equally in 3 dimemsions)\n"); break;
	case 4:
	printf("...Registration method: 9 degrees of freedom(translation, rotation, scaling)\n"); break;
	case 5:
	printf("...Registration method: 12 degrees of freedom\n"); break;
	case 6:
	printf("...Registration method: rigid body first, then do 12 degrees of freedom\n"); break;
	default:
	printf("...no registration method matched!!! please try other method\n"); return 0;
	}
	if (subBgTrigger)
	printf("...Subtract background for registration: Yes\n");
	else
	printf("...Subtract background for registration: No\n");
	if (inputTmx)
	printf("...Initial transformation matrix: Input Matrix\n");
	else
	printf("...Initial transformation matrix: Default\n");
	printf("...Registration convergence threshold:%f\n", FTOL);
	printf("...Maximum iteration number :~ %d\n", itLimit);
	fclose(stdout);
	*/
	//define variables and malloc memory
	//for powell searching
	float costValueInitial = 0;
	h_aff = (float *)malloc((NDIM)* sizeof(float));
	float *h_aff_temp = (float *)malloc((NDIM)* sizeof(float));
	float *h_affInitial = (float *)malloc((NDIM)* sizeof(float));
	static float *p = (float *)malloc((NDIM + 1) * sizeof(float));
	static float *p_dof9 = (float *)malloc((10) * sizeof(float));

	int iter;
	float fret, **xi, **xi_dof9;
	xi = matrix(1, NDIM, 1, NDIM);
	xi_dof9 = matrix(1, 9, 1, 9);

	//**** allocate memory for the images: 
	// CPU memory
	float
		*h_imgE = NULL;
	//cudaError_t cudaStatus;
	h_imgE = (float *)malloc(totalSizeMax * sizeof(float));
	int im2Dsize = (imx1*imy1 > imx2*imy2) ? imx1*imy1 : imx2*imy2;
	h_2D = (double *)malloc(im2Dsize * sizeof(double));

	// GPU memory
	cudaMalloc((void **)&d_img3D, totalSizeMax *sizeof(float));
	cudaMalloc((void **)&d_2D, im2Dsize*sizeof(double));
	cudaMalloc((void **)&d_2D2, im2Dsize*sizeof(double));
	cudaMalloc((void **)&d_2D3, im2Dsize*sizeof(double));
	cudaCheckErrors("****Memory allocating fails... GPU out of memory !!!!*****\n");

	cudaMemset(d_img3D, 0, totalSizeMax*sizeof(float));
	cudaMemset(d_2D, 0, im2Dsize*sizeof(double));
	cudaMemset(d_2D2, 0, im2Dsize*sizeof(double));
	cudaMemset(d_2D3, 0, im2Dsize*sizeof(double));

	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
	cudaArray *d_Array;
	cudaMalloc3DArray(&d_Array, &channelDesc, make_cudaExtent(imx2, imy2, imz2));
	cudaDeviceSynchronize();
	cudaCheckErrors("****GPU array memory allocating fails... GPU out of memory !!!!*****\n");
	cudaMemGetInfo(&freeMem, &totalMem);
	regRecords[1] = (float)freeMem / 1048576.0f;

	double
		sumImg1 = 0,
		sumImg2 = 0,
		sumSqr1 = 0;

	//*****************************************************
	//************** Start processing ******************
	start = clock();

	// ****************Preprocess images for registration ****************************//////
	// subtract mean value
	if (subBgTrigger){
		cudaMemcpy(d_img3D, h_img2, totalSize2 * sizeof(float), cudaMemcpyHostToDevice);
		cudaCheckErrors("Background subtraction: image2 memcpy fails");
		sumImg2 = sum3Dgpu(d_img3D, d_2D, h_2D, imx2, imy2, imz2);
		addvaluegpu(d_img3D, d_img3D, -float(sumImg2) / float(totalSize2), imx2, imy2, imz2);
		maxvalue3Dgpu(d_img3D, d_img3D, float(SMALLVALUE), imx2, imy2, imz2);
		cudacopydevicetoarray(d_Array, channelDesc, d_img3D, imx2, imy2, imz2);
		cudaCheckErrors("Background subtraction: image2 copy2array fails");


		cudaMemcpy(d_img3D, h_img1, totalSize1 * sizeof(float), cudaMemcpyHostToDevice);
		cudaCheckErrors("Background subtraction: image1 memcpy fail");
		sumImg1 = sum3Dgpu(d_img3D, d_2D, h_2D, imx1, imy1, imz1);
		addvaluegpu(d_img3D, d_img3D, -float(sumImg1) / float(totalSize1), imx1, imy1, imz1);
		maxvalue3Dgpu(d_img3D, d_img3D, float(SMALLVALUE), imx1, imy1, imz1);
		cudaMemcpy(h_imgE, d_img3D, totalSize1 * sizeof(float), cudaMemcpyDeviceToHost);

	}
	else{
		cudacopyhosttoarray(d_Array, channelDesc, h_img2, imx2, imy2, imz2);
		cudaCheckErrors("image2 copy2array fail");
	}
	//*****initialize transformation matrix*****//
	// use input matrix as initialization if inputTmx is true
	if (inputTmx == 1){
		memcpy(h_affInitial, iTmx, NDIM*sizeof(float));
	}
	else if (inputTmx == 2){
		// initial transform matrix
		//			1	0	0	a
		// h_aff =  0	1	0	b
		//			0	0	1	c
		for (int j = 0; j < NDIM; j++) h_affInitial[j] = 0;
		h_affInitial[0] = 1;
		h_affInitial[5] = 1;
		h_affInitial[10] = 1;
		int shiftXYZ[3];
		shiftXYZ[0] = shiftXYZ[1] = shiftXYZ[2] = 0;
		float *regRecordsTemp = (float *)malloc(11 * sizeof(float));
		int regStatus = reg_3dphasetransgpu(&shiftXYZ[0], h_img1, h_img2, &imSize1[0], &imSize2[0], 2, deviceNum, regRecordsTemp);
		h_affInitial[3] = -(float)shiftXYZ[0];
		h_affInitial[7] = -(float)shiftXYZ[1];
		h_affInitial[11] = -(float)shiftXYZ[2];
		printf("...phase translation, X: %f; Y: %f; Z: %f\n", h_affInitial[3], h_affInitial[7], h_affInitial[11]);
		free(regRecordsTemp);
		//memcpy(iTmx, h_affInitial, NDIM*sizeof(float));
	}
	else if (inputTmx == 3){
		float *tmx1 = (float *)malloc(6 * sizeof(float));
		float *tmx2 = (float *)malloc(6 * sizeof(float));
		int img2DSizeMax1 = ((imx1 * imy1) > (imz1 * imx1)) ? (imx1 * imy1) : (imz1 * imx1);
		int img2DSizeMax2 = ((imx2 * imy2) > (imz2 * imx2)) ? (imx2 * imy2) : (imz2 * imx2);
		int img2DSizeMax = (img2DSizeMax1 > img2DSizeMax2) ? img2DSizeMax1 : img2DSizeMax2;
		float *h_img2D1 = (float *)malloc(img2DSizeMax1 * sizeof(float));
		float *h_img2D2 = (float *)malloc(img2DSizeMax2 * sizeof(float));
		float *h_img2Dreg = (float *)malloc(img2DSizeMax1 * sizeof(float));
		float *regRecords2D = (float *)malloc(11 * sizeof(float));
		float *d_img2DMax = NULL;
		cudaMalloc((void **)&d_img2DMax, img2DSizeMax*sizeof(float));
		//cudaMalloc((void **)&d_img3D, totalSizeMax *sizeof(float));
		int shiftX = (imx2 - imx1) / 2, shiftY = (imy2 - imy1) / 2, shiftZ = (imz2 - imz1) / 2;
		int flag2Dreg = 1;
		switch (flag2Dreg){
		case 1:
			tmx1[0] = 1; tmx1[1] = 0; tmx1[2] = shiftX;
			tmx1[3] = 0; tmx1[4] = 1; tmx1[5] = shiftY;
			cudaMemcpy(d_img3D, h_img1, totalSize1 * sizeof(float), cudaMemcpyHostToDevice);
			maxprojection(d_img2DMax, d_img3D, imx1, imy1, imz1, 1);
			cudaMemcpy(h_img2D1, d_img2DMax, imx1 * imy1 * sizeof(float), cudaMemcpyDeviceToHost);
			cudaMemcpy(d_img3D, h_img2, totalSize2 * sizeof(float), cudaMemcpyHostToDevice);
			maxprojection(d_img2DMax, d_img3D, imx2, imy2, imz2, 1);
			cudaMemcpy(h_img2D2, d_img2DMax, imx2 * imy2 * sizeof(float), cudaMemcpyDeviceToHost);
			(void)reg_2dshiftaligngpu(h_img2Dreg, tmx1, h_img2D1, h_img2D2, imx1, imy1, imx2, imy2,
				0, 0.3, 15, deviceNum, regRecords2D);
			shiftX = tmx1[2];
			shiftY = tmx1[5];

			tmx2[0] = 1; tmx2[1] = 0; tmx2[2] = shiftZ;
			tmx2[3] = 0; tmx2[4] = 1; tmx2[5] = shiftX;
			cudaMemcpy(d_img3D, h_img1, totalSize1 * sizeof(float), cudaMemcpyHostToDevice);
			maxprojection(d_img2DMax, d_img3D, imx1, imy1, imz1, 2);
			cudaMemcpy(h_img2D1, d_img2DMax, imz1 * imx1 * sizeof(float), cudaMemcpyDeviceToHost);
			cudaMemcpy(d_img3D, h_img2, totalSize2 * sizeof(float), cudaMemcpyHostToDevice);
			maxprojection(d_img2DMax, d_img3D, imx2, imy2, imz2, 2);
			cudaMemcpy(h_img2D2, d_img2DMax, imx2 * imz2 * sizeof(float), cudaMemcpyDeviceToHost);
			(void)reg_2dshiftalignXgpu(h_img2Dreg, tmx2, h_img2D1, h_img2D2, imz1, imx1, imz2, imx2,
				1, 0.3, 15, deviceNum, regRecords2D);
			shiftZ = tmx2[2];
			h_affInitial[3] = shiftX;
			h_affInitial[7] = shiftY;
			h_affInitial[11] = shiftZ;
			printf("...shift translation, X: %f; Y: %f; Z: %f\n", h_affInitial[3], h_affInitial[7], h_affInitial[11]);
			break;
		default:
			break;
		}
		//cudaFree(d_img3D); 
		cudaFree(d_img2DMax);
		free(tmx1); free(tmx2); free(h_img2D1); free(h_img2D2); free(h_img2Dreg);
	}
	else{
		// initial transform matrix
		//			1	0	0	a
		// h_aff =  0	1	0	b
		//			0	0	1	c	
		for (int j = 0; j < NDIM; j++) h_affInitial[j] = 0;
		h_affInitial[0] = 1;
		h_affInitial[5] = 1;
		h_affInitial[10] = 1;
		h_affInitial[3] = (imx2 - imx1) / 2;
		h_affInitial[7] = (imy2 - imy1) / 2;
		h_affInitial[11] = (imz2 - imz1) / 2;
	}

	//****** 3D registration begains********///
	//calculate corrDenominator
	// transfer target image into GPU memory
	if (subBgTrigger)
		cudaMemcpy(d_img3D, h_imgE, totalSize1 * sizeof(float), cudaMemcpyHostToDevice);
	else
		cudaMemcpy(d_img3D, h_img1, totalSize1 * sizeof(float), cudaMemcpyHostToDevice);
	cudaCheckErrors("image1 memcpy fail");
	multi3Dgpu(d_img3D, d_img3D, d_img3D, imx1, imy1, imz1);
	sumSqr1 = sum3Dgpu(d_img3D, d_2D, h_2D, imx1, imy1, imz1);
	corrDenominator = sqrt(sumSqr1);
	/// calculate initial cost function value and time cost for each sub iteration
	if (subBgTrigger)
		cudaMemcpy(d_img3D, h_imgE, totalSize1 * sizeof(float), cudaMemcpyHostToDevice);
	else
		cudaMemcpy(d_img3D, h_img1, totalSize1 * sizeof(float), cudaMemcpyHostToDevice);
	cudaCheckErrors("Memcpy fail");
	// Create 3D texture for source image
	BindTexture(d_Array, channelDesc);

	startTemp = clock();
	dof9Flag = false;
	matrix2p(h_affInitial, p);
	regRecords[4] = costfunc(p);
	printf("...initial cost fun: %f;\n", regRecords[4]);
	endTemp = clock();
	regRecords[9] = (float)(endTemp - startTemp);

	totalItNum = 0;
	if (regMethod){
		// *********registration method:
		///for method 0, no registration 
		///for method 5, directly update initial matrix based on DOF 12 registration 
		///for method 1, 2, 3, 4, 6, 7, perform a affine transformation based on initial matrix and do DOF 9 registration 
		if (regMethod == 5){//***DOF = 12****
			matrix2p(h_affInitial, p);
		}
		else{
			if (inputTmx){// do transforamtion for DOF 9 registration
				//perform transformation based on initial matrix
				CopyTranMatrix(h_affInitial, NDIM * sizeof(float));
				affineTransform(d_img3D, imx2, imy2, imz2, imx2, imy2, imz2);
				UnbindTexture();
				cudacopydevicetoarray(d_Array, channelDesc, d_img3D, imx2, imy2, imz2);
				BindTexture(d_Array, channelDesc);

				if (subBgTrigger)
					cudaMemcpy(d_img3D, h_imgE, totalSize1 * sizeof(float), cudaMemcpyHostToDevice);
				else
					cudaMemcpy(d_img3D, h_img1, totalSize1 * sizeof(float), cudaMemcpyHostToDevice);
				cudaCheckErrors("Memcpy fail");

				//create initial matrix for registration
				p_dof9[0] = 0;
				p_dof9[1] = 0; p_dof9[2] = 0; p_dof9[3] = 0;
				p_dof9[4] = 0; p_dof9[5] = 0; p_dof9[6] = 0;
				p_dof9[7] = 1; p_dof9[8] = 1; p_dof9[9] = 1;
			}
			else{//create initial matrix for registration
				p_dof9[0] = 0;
				p_dof9[1] = (imx2 - imx1) / 2; p_dof9[2] = (imy2 - imy1) / 2; p_dof9[3] = (imz2 - imz1) / 2;
				p_dof9[4] = 0; p_dof9[5] = 0; p_dof9[6] = 0;
				p_dof9[7] = 1; p_dof9[8] = 1; p_dof9[9] = 1;
			}
		}

		// create orthogonal matrix to update searching matrix
		for (int i = 1; i <= NDIM; i++)
			for (int j = 1; j <= NDIM; j++)
				xi[i][j] = (i == j ? 1.0 : 0.0);

		//***DOF <=9***
		for (int i = 1; i <= 9; i++)
			for (int j = 1; j <= 9; j++)
				xi_dof9[i][j] = (i == j ? 1.0 : 0.0);

		totalItNum = 0;
		if (regMethod == 1){
			dof9Flag = true;
			dofNum = 3;
			powell(p_dof9, xi_dof9, dofNum, FTOL, &iter, &fret, costfunc, &totalItNum, itLimit);
		}
		if (regMethod == 2){
			dof9Flag = true;
			dofNum = 6;
			powell(p_dof9, xi_dof9, dofNum, FTOL, &iter, &fret, costfunc, &totalItNum, itLimit);
		}

		if (regMethod == 3){
			dof9Flag = true;
			dofNum = 7;
			powell(p_dof9, xi_dof9, dofNum, FTOL, &iter, &fret, costfunc, &totalItNum, itLimit);
		}

		if (regMethod == 4){
			dof9Flag = true;
			dofNum = 9;
			powell(p_dof9, xi_dof9, dofNum, FTOL, &iter, &fret, costfunc, &totalItNum, itLimit);
		}
		if (regMethod == 5){
			dof9Flag = false;
			dofNum = 12;
			powell(p, xi, dofNum, FTOL, &iter, &fret, costfunc, &totalItNum, itLimit);
		}

		if (regMethod == 6){
			// do 6 DOF --> 12 DOF
			dof9Flag = true;
			dofNum = 6;
			powell(p_dof9, xi_dof9, dofNum, 0.01, &iter, &fret, costfunc, &totalItNum, itLimit);
			regRecords[6] = fret;
			// do DOF 12 registration
			dof9Flag = false;
			dofNum = 12;
			matrix2p(h_aff, p);
			powell(p, xi, dofNum, FTOL, &iter, &fret, costfunc, &totalItNum, itLimit);
		}
		if (regMethod == 7){
			// do 3 DOF --> 6 DOF --> 9 DOF --> 12 DOF
			dof9Flag = true;
			dofNum = 3;
			powell(p_dof9, xi_dof9, dofNum, 0.01, &iter, &fret, costfunc, &totalItNum, itLimit);
			dofNum = 6;
			powell(p_dof9, xi_dof9, dofNum, 0.01, &iter, &fret, costfunc, &totalItNum, itLimit);
			dofNum = 9;
			powell(p_dof9, xi_dof9, dofNum, 0.005, &iter, &fret, costfunc, &totalItNum, itLimit);
			regRecords[6] = fret;
			// do DOF 12 registration
			dof9Flag = false;
			dofNum = 12;
			matrix2p(h_aff, p);
			powell(p, xi, dofNum, FTOL, &iter, &fret, costfunc, &totalItNum, itLimit);
		}
		endTemp = clock();
		if ((inputTmx) && (regMethod != 5)){
			matrixmultiply(h_aff_temp, h_affInitial, h_aff); //final transformation matrix
			memcpy(h_aff, h_aff_temp, NDIM*sizeof(float));
		}
		regRecords[5] = fret; // mimized cost fun value
	}
	else {
		//leave h_aff as it is, and then estimate the cost fuction value
		matrix2p(h_affInitial, p);
		dof9Flag = false;
		fret = costfunc(p);
		regRecords[6] = fret;
		memcpy(h_aff, h_affInitial, NDIM*sizeof(float));
	}
	regRecords[10] = (float)totalItNum;
	//****Perform affine transformation with optimized coefficients****//
	// replace Texture with raw stack_B data
	if ((subBgTrigger) || (inputTmx)){
		UnbindTexture();
		cudacopyhosttoarray(d_Array, channelDesc, h_img2, imx2, imy2, imz2);
		BindTexture(d_Array, channelDesc);
	}
	CopyTranMatrix(h_aff, NDIM * sizeof(float));
	affineTransform(d_img3D, imx1, imy1, imz1, imx2, imy2, imz2);
	UnbindTexture();
	cudaMemcpy(h_reg, d_img3D, totalSize1 * sizeof(float), cudaMemcpyDeviceToHost);
	memcpy(iTmx, h_aff, NDIM*sizeof(float));
	//****write registerred stack and transformation matrix***//
	// always save transformation matrix
	cudaDeviceSynchronize();
	end = clock();
	regRecords[7] = (float)(end - start) / CLOCKS_PER_SEC;
	cudaMemGetInfo(&freeMem, &totalMem);
	regRecords[2] = (float)freeMem / 1048576.0f; // GPU memory after processing

	// release all memory
	//free CPU memory
	free(h_imgE);
	free(h_2D);

	//free(p);
	//free(p_dof9);

	free(h_aff);
	free(h_aff_temp);
	free(h_affInitial);

	free_matrix(xi, 1, NDIM, 1, NDIM);
	free_matrix(xi_dof9, 1, 9, 1, 9);

	//free GPU memory for images
	cudaFree(d_img3D);


	cudaFree(d_2D);
	cudaFree(d_2D2);
	cudaFree(d_2D3);

	cudaFreeArray(d_Array);
	// destroy time recorder
	endWhole = clock();
	regRecords[8] = (float)(endWhole - startWhole) / CLOCKS_PER_SEC;
	cudaMemGetInfo(&freeMem, &totalMem);
	regRecords[3] = (float)freeMem / 1048576.0f; // GPU memory after variables released
	return 0;
}

int reg_3dcpu(float *h_reg, float *iTmx, float *h_img1, float *h_img2, unsigned int *imSize1, unsigned int *imSize2, int regMethod,
	int inputTmx, float FTOL, int itLimit, int subBgTrigger, int deviceNum, float *regRecords){
	//regRecords
	//[0] -[3]: initial GPU memory, after variables allocated, after processing, after variables released ( all in MB)
	//[4] -[6]: initial cost function value, minimized cost function value, intermediate cost function value
	//[7] -[10]: registration time (in s), whole time (in s), single sub iteration time (in ms), total sub iterations

	//************get basic input images information ******************	
	//image size
	imx1 = imSize1[0]; imy1 = imSize1[1]; imz1 = imSize1[2];
	imx2 = imSize2[0]; imy2 = imSize2[1]; imz2 = imSize2[2];
	// total pixel count for each images
	int totalSize1 = imx1*imy1*imz1;
	int totalSize2 = imx2*imy2*imz2;
	int totalSizeMax = (totalSize1 > totalSize2) ? totalSize1 : totalSize2;
	// print GPU devices information
	cudaSetDevice(deviceNum);
	//****************** Processing Starts*****************//
	// variables for memory and time cost records
	clock_t start, startTemp, startWhole, end, endTemp, endWhole;

	/*
	// write records
	freopen("log.txt", "w", stdout);
	printf("3D registration: calling CUDA DLL....\n");
	printf("GPU device: #%d \n", deviceNum);
	printf("...GPU free memory(at beginning) is %.0f MBites\n", (float)freeMem / 1048576.0f);
	printf("\nImage information:\n");
	printf("...Target size %d x %d x %d\n", imx1, imy1, imz1);
	printf("...Source size %d x %d x %d\n", imx2, imy2, imz2);
	printf("...Registered image size %d x %d x %d\n", imx1, imy1, imz1);
	printf("\nRegistration setup:\n");
	if (regTrigger)
	printf("...Perform registration\n");
	else
	printf("...No registration\n");
	switch (regMethod){
	case 1:
	printf("...Registration method: translation only\n"); break;
	case 2:
	printf("...Registration method: rigid body\n"); break;
	case 3:
	printf("...Registration method: 7 degrees of freedom(translation, rotation, scaling equally in 3 dimemsions)\n"); break;
	case 4:
	printf("...Registration method: 9 degrees of freedom(translation, rotation, scaling)\n"); break;
	case 5:
	printf("...Registration method: 12 degrees of freedom\n"); break;
	case 6:
	printf("...Registration method: rigid body first, then do 12 degrees of freedom\n"); break;
	default:
	printf("...no registration method matched!!! please try other method\n"); return 0;
	}
	if (subBgTrigger)
	printf("...Subtract background for registration: Yes\n");
	else
	printf("...Subtract background for registration: No\n");
	if (inputTmx)
	printf("...Initial transformation matrix: Input Matrix\n");
	else
	printf("...Initial transformation matrix: Default\n");
	printf("...Registration convergence threshold:%f\n", FTOL);
	printf("...Maximum iteration number :~ %d\n", itLimit);
	fclose(stdout);
	
	*/
	startWhole = clock();
	//define variables and malloc memory
	//for powell searching
	float costValueInitial = 0;
	h_aff = (float *)malloc((NDIM)* sizeof(float));
	float *h_aff_temp = (float *)malloc((NDIM)* sizeof(float));
	float *h_affInitial = (float *)malloc((NDIM)* sizeof(float));

	static float *p = (float *)malloc((NDIM + 1) * sizeof(float));
	static float *p_dof9 = (float *)malloc((10) * sizeof(float));

	int iter;
	float fret, **xi, **xi_dof9;
	xi = matrix(1, NDIM, 1, NDIM);
	xi_dof9 = matrix(1, 9, 1, 9);

	//**** allocate memory for the images: 
	// CPU memory
	float
		*h_imgE = NULL;
	//cudaError_t cudaStatus;
	h_imgE = (float *)malloc(totalSizeMax * sizeof(float));
	h_s3D = (float *)malloc(totalSize1 * sizeof(float));
	h_t3D = (float *)malloc(totalSize2 * sizeof(float));
	int im2Dsize = (imx1*imy1 > imx2*imy2) ? imx1*imy1 : imx2*imy2;

	double
		sumImg1 = 0,
		sumImg2 = 0,
		sumSqr1 = 0;

	//*****************************************************
	//************** Start processing ******************
	start = clock();
	// ****************Preprocess images for registration ****************************//////
	// subtract mean value
	if (subBgTrigger){
		sumImg2 = sumcpu(h_img2, totalSize2);
		addvaluecpu(h_t3D, h_img2, -float(sumImg2) / float(totalSize2), totalSize2);
		maxvaluecpu(h_t3D, h_t3D, float(SMALLVALUE), totalSize2);

		sumImg1 = sumcpu(h_img1, totalSize1);
		addvaluecpu(h_s3D, h_img1, -float(sumImg1) / float(totalSize1), totalSize1);
		maxvaluecpu(h_s3D, h_s3D, float(SMALLVALUE), totalSize1);

	}
	else{
		memcpy(h_s3D, h_img1, totalSize1);
		memcpy(h_t3D, h_img2, totalSize2);
	}



	//****** 3D registration begains********///
	//calculate corrDenominator
	multicpu(h_imgE, h_s3D, h_s3D, totalSize1);
	sumSqr1 = sumcpu(h_imgE, totalSize1);
	corrDenominator = sqrt(sumSqr1);

	//*****initialize transformation matrix*****//
	// use input matrix as initialization if inputTmx is true
	if (inputTmx){
		memcpy(h_affInitial, iTmx, NDIM*sizeof(float));
	}
	else{
		// initial transform matrix
		//			1	0	0	a
		// h_aff =  0	1	0	b
		//			0	0	1	c	
		for (int j = 0; j < NDIM; j++) h_affInitial[j] = 0;
		h_affInitial[0] = 1;
		h_affInitial[5] = 1;
		h_affInitial[10] = 1;
		h_affInitial[3] = (imx2 - imx1) / 2;
		h_affInitial[7] = (imy2 - imy1) / 2;
		h_affInitial[11] = (imz2 - imz1) / 2;
	}
	/// calculate initial cost function value and time cost for each sub iteration
	startTemp = clock();
	dof9Flag = false;
	matrix2p(h_affInitial, p);
	regRecords[4] = costfunccpu(p);
	endTemp = clock();
	regRecords[9] = (float)(endTemp - startTemp);

	totalItNum = 0;
	if (regMethod){
		// *********registration method:
		///for method 5, directly update initial matrix based on DOF 12 registration 
		///for method 1, 2, 3, 4, 6, perform a affine transformation based on initial matrix and do DOF 9 registration 
		if (regMethod == 5){//***DOF = 12****
			matrix2p(h_affInitial, p);
		}
		else{
			if (inputTmx){// do transforamtion for DOF 9 registration
				//perform transformation based on initial matrix
				affinetransformcpu(h_imgE, h_t3D, h_affInitial, imx2, imy2, imz2, imx2, imy2, imz2);
				memcpy(h_t3D, h_imgE, totalSize2);

				//create initial matrix for registration
				p_dof9[0] = 0;
				p_dof9[1] = 0; p_dof9[2] = 0; p_dof9[3] = 0;
				p_dof9[4] = 0; p_dof9[5] = 0; p_dof9[6] = 0;
				p_dof9[7] = 1; p_dof9[8] = 1; p_dof9[9] = 1;
			}
			else{//create initial matrix for registration
				p_dof9[0] = 0;
				p_dof9[1] = (imx2 - imx1) / 2; p_dof9[2] = (imy2 - imy1) / 2; p_dof9[3] = (imz2 - imz1) / 2;
				p_dof9[4] = 0; p_dof9[5] = 0; p_dof9[6] = 0;
				p_dof9[7] = 1; p_dof9[8] = 1; p_dof9[9] = 1;
			}
		}

		// create orthogonal matrix to update searching matrix
		for (int i = 1; i <= NDIM; i++)
			for (int j = 1; j <= NDIM; j++)
				xi[i][j] = (i == j ? 1.0 : 0.0);

		//***DOF <=9***
		for (int i = 1; i <= 9; i++)
			for (int j = 1; j <= 9; j++)
				xi_dof9[i][j] = (i == j ? 1.0 : 0.0);

		totalItNum = 0;
		if (regMethod == 1){
			dof9Flag = true;
			dofNum = 3;
			powell(p_dof9, xi_dof9, dofNum, FTOL, &iter, &fret, costfunccpu, &totalItNum, itLimit);
		}
		if (regMethod == 2){
			dof9Flag = true;
			dofNum = 6;
			powell(p_dof9, xi_dof9, dofNum, FTOL, &iter, &fret, costfunccpu, &totalItNum, itLimit);
		}

		if (regMethod == 3){
			dof9Flag = true;
			dofNum = 7;
			powell(p_dof9, xi_dof9, dofNum, FTOL, &iter, &fret, costfunccpu, &totalItNum, itLimit);
		}

		if (regMethod == 4){
			dof9Flag = true;
			dofNum = 9;
			powell(p_dof9, xi_dof9, dofNum, FTOL, &iter, &fret, costfunccpu, &totalItNum, itLimit);
		}
		if (regMethod == 5){
			dof9Flag = false;
			dofNum = 12;
			powell(p, xi, dofNum, FTOL, &iter, &fret, costfunccpu, &totalItNum, itLimit);
		}

		if (regMethod == 6){
			// do DOF 6 registration
			dof9Flag = true;
			dofNum = 6;
			powell(p_dof9, xi_dof9, dofNum, 0.01, &iter, &fret, costfunccpu, &totalItNum, itLimit);
			regRecords[6] = fret;
			// do DOF 12 registration
			dof9Flag = false;
			dofNum = 12;
			matrix2p(h_aff, p);
			powell(p, xi, dofNum, FTOL, &iter, &fret, costfunccpu, &totalItNum, itLimit);
		}
		if (regMethod == 7){
			// do 3 DOF --> 6 DOF --> 9 DOF --> 12 DOF
			dof9Flag = true;
			dofNum = 3;
			powell(p_dof9, xi_dof9, dofNum, 0.01, &iter, &fret, costfunccpu, &totalItNum, itLimit);
			dofNum = 6;
			powell(p_dof9, xi_dof9, dofNum, 0.01, &iter, &fret, costfunccpu, &totalItNum, itLimit);
			dofNum = 9;
			powell(p_dof9, xi_dof9, dofNum, 0.005, &iter, &fret, costfunccpu, &totalItNum, itLimit);
			regRecords[6] = fret;
			// do DOF 12 registration
			dof9Flag = false;
			dofNum = 12;
			matrix2p(h_aff, p);
			powell(p, xi, dofNum, FTOL, &iter, &fret, costfunccpu, &totalItNum, itLimit);
		}
		endTemp = clock();
		if (inputTmx && (regMethod != 5)){
			matrixmultiply(h_aff_temp, h_affInitial, h_aff); //final transformation matrix
			memcpy(h_aff, h_aff_temp, NDIM*sizeof(float));
		}
		regRecords[5] = fret; // mimized cost fun value
	}
	else {
		//leave h_aff as it is, and then estimate the cost fuction value
		matrix2p(h_affInitial, p);
		dof9Flag = false;
		fret = costfunccpu(p);
		regRecords[6] = fret;
		memcpy(h_aff, h_affInitial, NDIM*sizeof(float));
	}
	regRecords[10] = (float)totalItNum;
	//****Perform affine transformation with optimized coefficients****//
	// replace Texture with raw stack_B data
	if ((subBgTrigger) || (inputTmx)){
		memcpy(h_t3D, h_img2, totalSize2);
	}

	affinetransformcpu(h_reg, h_t3D, h_aff, imx1, imy1, imz1, imx2, imy2, imz2);

	memcpy(iTmx, h_aff, NDIM*sizeof(float));
	//****write registerred stack and transformation matrix***//
	// always save transformation matrix
	end = clock();
	regRecords[7] = (float)(end - start) / CLOCKS_PER_SEC;

	// release all memory
	//free CPU memory
	free(h_imgE);
	free(h_s3D);
	free(h_t3D);

	free(p);
	free(p_dof9);

	free(h_aff);
	free(h_aff_temp);
	free(h_affInitial);

	free_matrix(xi, 1, NDIM, 1, NDIM);
	free_matrix(xi_dof9, 1, 9, 1, 9);



	// destroy time recorder
	endWhole = clock();
	regRecords[8] = (float)(endWhole - startWhole) / CLOCKS_PER_SEC;
	return 0;
}
///// 3D deconlution
int decon_singleview(float *h_decon, float *h_img, unsigned int *imSize, float *h_psf, unsigned int *psfSize, 
	int itNumForDecon, int deviceNum, int gpuMemMode, float *deconRecords, bool flagUnmatch, float *h_psf_bp){
	// gpuMemMode --> 0: Automatically set memory mode based on calculations; 1: sufficient memory; 2: memory optimized.
	//deconRecords: 10 elements
	//[0]:  the actual GPU memory mode used;
	//[1] -[5]: initial GPU memory, after variables partially allocated, during processing, after processing, after variables released ( all in MB);
	//[6] -[9]: initializing time, prepocessing time, decon time, total time;

	float
		*h_StackA,
		*h_StackE,
		*d_StackA,
		*d_StackE,
		*d_StackT;

	fComplex
		*h_PSFSpectrum,
		*h_FlippedPSFSpectrum,
		*h_StackESpectrum,
		*d_PSFSpectrum,
		*d_FlippedPSFSpectrum,
		*d_StackESpectrum;
	cufftHandle
		fftPlanFwd,
		fftPlanInv;
	// image size
	int
		imx, imy, imz;
	imx = imSize[0], imy = imSize[1], imz = imSize[2];

	// PSF size
	int
		PSFx, PSFy, PSFz;
	PSFx = psfSize[0], PSFy = psfSize[1], PSFz = psfSize[2];

	//FFT size
	int
		FFTx, FFTy, FFTz,
		PSFox, PSFoy, PSFoz,
		imox, imoy, imoz;

	FFTx = snapTransformSize(imx);// snapTransformSize(imx + PSFx - 1);
	FFTy = snapTransformSize(imy);// snapTransformSize(imy + PSFy - 1);
	FFTz = snapTransformSize(imz);// snapTransformSize(imz + PSFz - 1);
	// set original points for padding and cropping
	//fftz.y.z·
	PSFox = round(PSFx / 2);// round((FFTx - PSFx) / 2);
	PSFoy = round(PSFy / 2);//round((FFTy - PSFy) / 2);
	PSFoz = round(PSFz / 2);//round((FFTz - PSFz) / 2 );
	imox = round((FFTx - imSize[0]) / 2);
	imoy = round((FFTy - imSize[1]) / 2);
	imoz = round((FFTz - imSize[2]) / 2);

	printf("Image information:\n");
	printf("...Image size %d x %d x %d\n  ", imx, imy, imz);
	printf("...PSF size %d x %d x %d\n  ", PSFx, PSFy, PSFz);
	printf("...FFT size %d x %d x %d\n  ", FFTx, FFTy, FFTz);

	printf("...Output Image size %d x %d x %d \n   ", imSize[0], imSize[1], imSize[2]);

	// total pixel count for each images
	int totalSize = imx*imy*imz; // in floating format
	int totalSizePSF = PSFx*PSFy*PSFz; // in floating format
	int totalSizeFFT = FFTx*FFTy*FFTz; // in floating format
	int totalSizeSpectrum = FFTx * FFTy*(FFTz / 2 + 1); // in complex floating format
	int totalSizeMax = totalSizeSpectrum * 2; // in floating format
	int totalSizeMax2 = totalSizeMax > totalSizePSF ? totalSizeMax : totalSizePSF; // in floating format: in case PSF has a larger size
	// print GPU devices information
	cudaSetDevice(deviceNum);
	//****************** Processing Starts*****************//
	// variables for memory and time cost records
	clock_t start, time1, time2, time3, end;
	size_t totalMem = 0;
	size_t freeMem = 0;
	time1 = time2 = time3 = end = 0;
	start = clock();
	// allocate memory
	cudaMemGetInfo(&freeMem, &totalMem);
	printf("...GPU free memory(at beginning) is %.0f MBites\n", (float)freeMem / 1048576.0f);
	deconRecords[1] = (float)freeMem / 1048576.0f;
	cudaMalloc((void **)&d_StackA, totalSizeMax2 *sizeof(float));
	cudaMalloc((void **)&d_StackE, totalSizeMax2 *sizeof(float));
	cudaMemset(d_StackA, 0, totalSizeMax2*sizeof(float));
	cudaMemset(d_StackE, 0, totalSizeMax2*sizeof(float));
	//check GPU status
	cudaCheckErrors("****Memory allocating fails... GPU out of memory !!!!*****");
	// Create FFT plans
	cufftPlan3d(&fftPlanFwd, FFTx, FFTy, FFTz, CUFFT_R2C);
	cufftPlan3d(&fftPlanInv, FFTx, FFTy, FFTz, CUFFT_C2R);
	cudaCheckErrors("****Memory allocating fails... GPU out of memory !!!!*****");

	cudaMemGetInfo(&freeMem, &totalMem);
	printf("...GPU free memory(after partially mallocing) is %.0f MBites\n", (float)freeMem / 1048576.0f);
	deconRecords[2] = (float)freeMem / 1048576.0f;

	// ***** Set GPU memory use mode based on images size and available GPU memory ****
	// gpuMemMode --> Unified memory in next version???
	// 0: Automatically set memory mode based on calculations; 
	// 1: sufficient memory; 2: memory optimized.
	if (gpuMemMode == 0){ //Automatically set memory mode based on calculations.
		if (freeMem > 4 * totalSizeMax * sizeof(float)){ // 7 more GPU variables
			gpuMemMode = 1;
			printf("\n GPU memory is sufficient, processing in efficient mode !!!\n");
		}
		else {// no more GPU variables needed
			gpuMemMode = 2;
			printf("\n GPU memory is optimized, processing in memory saved mode !!!\n");
		}
	}
	deconRecords[0] = gpuMemMode;
	double mySumPSF = 0;
	switch (gpuMemMode){
	case 1:// efficient GPU calculation
		time1 = clock();
		cudaMalloc((void **)&d_StackT, totalSizeFFT *sizeof(float));
		cudaMalloc((void **)&d_PSFSpectrum, totalSizeSpectrum*sizeof(fComplex));
		cudaMalloc((void **)&d_FlippedPSFSpectrum, totalSizeSpectrum*sizeof(fComplex));
		cudaMalloc((void **)&d_StackESpectrum, totalSizeSpectrum*sizeof(fComplex));
		cudaMemGetInfo(&freeMem, &totalMem);
		printf("...GPU free memory(after mallocing) is %.0f MBites\n", (float)freeMem / 1048576.0f);
		// *** PSF Preparation
		//PSF 
		cudaMemcpy(d_StackE, h_psf, totalSizePSF* sizeof(float), cudaMemcpyHostToDevice);
		changestorageordergpu(d_StackA, d_StackE, PSFx, PSFy, PSFz, 1); //1: change tiff storage order to C storage order
		mySumPSF = sumcpu(h_psf, totalSizePSF);
		multivaluegpu(d_StackE, d_StackA, (float)(1 / mySumPSF), PSFx, PSFy, PSFz); // normalize PSF sum to 1
		if (!flagUnmatch){ // traditional backprojector matched PSF 
			flipPSF(d_StackA, d_StackE, PSFx, PSFy, PSFz); // flip PSF
			cudaMemcpy(h_psf_bp, d_StackA, totalSizePSF* sizeof(float), cudaMemcpyDeviceToHost);
		}
		cudaMemset(d_StackA, 0, totalSizeMax2*sizeof(float));
		padPSF(d_StackA, d_StackE, FFTx, FFTy, FFTz, PSFx, PSFy, PSFz, PSFox, PSFoy, PSFoz); 		
		cufftExecR2C(fftPlanFwd, (cufftReal *)d_StackA, (cufftComplex *)d_PSFSpectrum);
		//PSF bp
		cudaMemcpy(d_StackE, h_psf_bp, totalSizePSF* sizeof(float), cudaMemcpyHostToDevice);
		if (flagUnmatch){
			changestorageordergpu(d_StackA, d_StackE, PSFx, PSFy, PSFz, 1); //1: change tiff storage order to C storage order
			mySumPSF = sumcpu(h_psf_bp, totalSizePSF);
			multivaluegpu(d_StackE, d_StackA, (float)(1 / mySumPSF), PSFx, PSFy, PSFz); // normalize PSF sum to 1
		}
		cudaMemset(d_StackA, 0, totalSizeMax2*sizeof(float));
		padPSF(d_StackA, d_StackE, FFTx, FFTy, FFTz, PSFx, PSFy, PSFz, PSFox, PSFoy, PSFoz);
		cufftExecR2C(fftPlanFwd, (cufftReal *)d_StackA, (cufftComplex *)d_FlippedPSFSpectrum);

		// Prepare Stack Data
		cudaMemcpy(d_StackA, h_img, totalSize* sizeof(float), cudaMemcpyHostToDevice);
		//eliminate 0 in stacks
		maxvalue3Dgpu(d_StackA, d_StackA, (float)(SMALLVALUE), imx, imy, imz);
		changestorageordergpu(d_StackE, d_StackA, imx, imy, imz, 1); //1: change tiff storage order to C storage order
		padStack(d_StackA, d_StackE, FFTx, FFTy, FFTz, imx, imy, imz, imox, imoy, imoz);//
		// initialize estimation
		cudaMemcpy(d_StackE, d_StackA, totalSizeFFT* sizeof(float), cudaMemcpyDeviceToDevice);
		cudaCheckErrors("image preparing fail");
		//printf("...Initializing deconvolution iteration...\n");
		for (int itNum = 1; itNum <= itNumForDecon; itNum++){
			// ### iterate with StackA and PSFA///////////////////
			// convolve StackE with PSFA
			//printf("...Processing iteration %d\n", it);
			cufftExecR2C(fftPlanFwd, (cufftReal *)d_StackE, (cufftComplex *)d_StackESpectrum);
			multicomplex3Dgpu(d_StackESpectrum, d_StackESpectrum, d_PSFSpectrum, FFTx, FFTy, (FFTz / 2 + 1));
			cufftExecC2R(fftPlanInv, (cufftComplex *)d_StackESpectrum, (cufftReal *)d_StackT);
			// divid StackA by StackTemp
			div3Dgpu(d_StackT, d_StackA, d_StackT, FFTx, FFTy, FFTz);   //// div3Dgpu does not work
			// convolve StackTemp with FlippedPSFA
			cufftExecR2C(fftPlanFwd, (cufftReal *)d_StackT, (cufftComplex *)d_StackESpectrum);
			multicomplex3Dgpu(d_StackESpectrum, d_StackESpectrum, d_FlippedPSFSpectrum, FFTx, FFTy, (FFTz / 2 + 1));
			cufftExecC2R(fftPlanInv, (cufftComplex *)d_StackESpectrum, (cufftReal *)d_StackT);//test
			// multiply StackE and StackTemp
			multi3Dgpu(d_StackE, d_StackE, d_StackT, FFTx, FFTy, FFTz);//
			maxvalue3Dgpu(d_StackE, d_StackE, float(SMALLVALUE), FFTx, FFTy, FFTz);
		}
		cropStack(d_StackE, d_StackT, FFTx, FFTy, FFTz, imx, imy, imz, imox, imoy, imoz);//
		//printf("...Deconvolution completed ! ! !\n");
		cudaDeviceSynchronize();
		changestorageordergpu(d_StackE, d_StackT, imx, imy, imz, -1); //-1: change C storage order to tiff storage order
		cudaMemcpy(h_decon, d_StackE, totalSize* sizeof(float), cudaMemcpyDeviceToHost);

		time3 = clock();
		cudaMemGetInfo(&freeMem, &totalMem);
		printf("...GPU free memory (after processing) is %.0f MBites\n", (float)freeMem / 1048576.0f);
		deconRecords[4] = (float)freeMem / 1048576.0f;
		// release CUDA variables
		cudaFree(d_StackT); cudaFree(d_PSFSpectrum); cudaFree(d_FlippedPSFSpectrum); cudaFree(d_StackESpectrum);
		break;
	case 2:
		time1 = clock();
		h_StackA = (float *)malloc(totalSizeMax * sizeof(float));
		h_StackE = (float *)malloc(totalSizeMax * sizeof(float));
		h_PSFSpectrum = (fComplex *)malloc(totalSizeSpectrum*sizeof(fComplex));
		h_FlippedPSFSpectrum = (fComplex *)malloc(totalSizeSpectrum*sizeof(fComplex));
		h_StackESpectrum = (fComplex *)malloc(totalSizeSpectrum*sizeof(fComplex));
		d_StackESpectrum = (fComplex *)d_StackE; // share the same physic memory
		// *** PSF Preparation
		//PSF 
		cudaMemcpy(d_StackE, h_psf, totalSizePSF* sizeof(float), cudaMemcpyHostToDevice);
		changestorageordergpu(d_StackA, d_StackE, PSFx, PSFy, PSFz, 1); //1: change tiff storage order to C storage order
		mySumPSF = sumcpu(h_psf, totalSizePSF);
		multivaluegpu(d_StackE, d_StackA, (float)(1 / mySumPSF), PSFx, PSFy, PSFz); // normalize PSF sum to 1
		if (!flagUnmatch){ // traditional backprojector matched PSF 
			flipPSF(d_StackA, d_StackE, PSFx, PSFy, PSFz); // flip PSF
			cudaMemcpy(h_psf_bp, d_StackA, totalSizePSF* sizeof(float), cudaMemcpyDeviceToHost);
		}
		cudaMemset(d_StackA, 0, totalSizeMax2*sizeof(float));
		padPSF(d_StackA, d_StackE, FFTx, FFTy, FFTz, PSFx, PSFy, PSFz, PSFox, PSFoy, PSFoz);
		cufftExecR2C(fftPlanFwd, (cufftReal *)d_StackA, (cufftComplex *)d_StackESpectrum);
		cudaMemcpy(h_PSFSpectrum, d_StackESpectrum, totalSizeSpectrum*sizeof(fComplex), cudaMemcpyDeviceToHost);
		//PSF bp
		cudaMemcpy(d_StackE, h_psf_bp, totalSizePSF* sizeof(float), cudaMemcpyHostToDevice);
		if (flagUnmatch){
			changestorageordergpu(d_StackA, d_StackE, PSFx, PSFy, PSFz, 1); //1: change tiff storage order to C storage order
			mySumPSF = sumcpu(h_psf_bp, totalSizePSF);
			multivaluegpu(d_StackE, d_StackA, (float)(1 / mySumPSF), PSFx, PSFy, PSFz); // normalize PSF sum to 1
		}
		cudaMemset(d_StackA, 0, totalSizeMax2*sizeof(float));
		padPSF(d_StackA, d_StackE, FFTx, FFTy, FFTz, PSFx, PSFy, PSFz, PSFox, PSFoy, PSFoz);
		cufftExecR2C(fftPlanFwd, (cufftReal *)d_StackA, (cufftComplex *)d_StackESpectrum);
		cudaMemcpy(h_FlippedPSFSpectrum, d_StackESpectrum, totalSizeSpectrum*sizeof(fComplex), cudaMemcpyDeviceToHost);

		// Prepare Stack Data
		cudaMemcpy(d_StackA, h_img, totalSize* sizeof(float), cudaMemcpyHostToDevice);
		//eliminate 0 in stacks
		maxvalue3Dgpu(d_StackA, d_StackA, (float)(SMALLVALUE), imx, imy, imz);
		changestorageordergpu(d_StackE, d_StackA, imx, imy, imz, 1); //1: change tiff storage order to C storage order
		padStack(d_StackA, d_StackE, FFTx, FFTy, FFTz, imx, imy, imz, imox, imoy, imoz);//

		// initialize estimation
		cudaMemcpy(h_StackA, d_StackA, totalSizeFFT * sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy(h_StackE, d_StackA, totalSizeFFT * sizeof(float), cudaMemcpyDeviceToHost);

		d_PSFSpectrum = (fComplex *)d_StackA; // share the same physic memory
		d_FlippedPSFSpectrum = (fComplex *)d_StackA; // share the same physic memory
		cudaCheckErrors("image preparing fail");
		//printf("...Initializing deconvolution iteration...\n");
		for (int itNum = 1; itNum <= itNumForDecon; itNum++){
			// ### iterate with StackA and PSFA///////////////////
			// convolve StackE with PSFA
			//printf("...Processing iteration %d\n", it);
			cufftExecR2C(fftPlanFwd, (cufftReal *)d_StackA, (cufftComplex *)d_StackESpectrum);
			cudaMemcpy(d_PSFSpectrum, h_PSFSpectrum, totalSizeSpectrum*sizeof(fComplex), cudaMemcpyHostToDevice);
			multicomplex3Dgpu(d_StackESpectrum, d_StackESpectrum, d_PSFSpectrum, FFTx, FFTy, (FFTz / 2 + 1));
			cufftExecC2R(fftPlanInv, (cufftComplex *)d_StackESpectrum, (cufftReal *)d_StackA);

			// divid StackA by StackTemp
			cudaMemcpy(d_StackE, h_StackA, totalSizeFFT* sizeof(float), cudaMemcpyHostToDevice);
			div3Dgpu(d_StackA, d_StackE, d_StackA, FFTx, FFTy, FFTz);   //// div3Dgpu does not work
			// convolve StackTemp with FlippedPSFA
			cufftExecR2C(fftPlanFwd, (cufftReal *)d_StackA, (cufftComplex *)d_StackESpectrum);
			cudaMemcpy(d_FlippedPSFSpectrum, h_FlippedPSFSpectrum, totalSizeSpectrum*sizeof(fComplex), cudaMemcpyHostToDevice);
			multicomplex3Dgpu(d_StackESpectrum, d_StackESpectrum, d_FlippedPSFSpectrum, FFTx, FFTy, (FFTz / 2 + 1));
			cufftExecC2R(fftPlanInv, (cufftComplex *)d_StackESpectrum, (cufftReal *)d_StackA);//test
			// multiply StackE and StackTemp
			cudaMemcpy(d_StackE, h_StackE, totalSizeFFT* sizeof(float), cudaMemcpyHostToDevice);
			multi3Dgpu(d_StackA, d_StackE, d_StackA, FFTx, FFTy, FFTz);//
			cudaMemcpy(h_StackE, d_StackA, totalSizeFFT* sizeof(float), cudaMemcpyDeviceToHost);
		}
		cropStack(d_StackA, d_StackE, FFTx, FFTy, FFTz, imx, imy, imz, imox, imoy, imoz);//
		//printf("...Deconvolution completed ! ! !\n");
		cudaDeviceSynchronize();
		//## Write stack to tiff image
		changestorageordergpu(d_StackA, d_StackE, imx, imy, imz, -1); //-1: change C storage order to tiff storage order
		cudaMemcpy(h_decon, d_StackA, totalSize* sizeof(float), cudaMemcpyDeviceToHost);
		time3 = clock();
		cudaMemGetInfo(&freeMem, &totalMem);
		printf("...GPU free memory (after processing) is %.0f MBites\n", (float)freeMem / 1048576.0f);
		deconRecords[4] = (float)freeMem / 1048576.0f;
		// release CPU memory
		free(h_StackA);  free(h_StackE); free(h_PSFSpectrum); free(h_FlippedPSFSpectrum); free(h_StackESpectrum);
		break;
	default:
		printf("\n****Wrong gpuMemMode setup, no deconvolution performed !!! ****\n");
		return -1;
	}
	// release GPU memory
	cudaFree(d_StackA);
	cudaFree(d_StackE);
	// destroy plans
	cufftDestroy(fftPlanFwd);
	cufftDestroy(fftPlanInv);
	end = clock();
	cudaMemGetInfo(&freeMem, &totalMem);
	printf("GPU free memory (after variable released): %.0f MBites\n", (float)freeMem / 1048576.0f);
	deconRecords[5] = (float)freeMem / 1048576.0f;
	deconRecords[6] = (float)(time1 - start) / CLOCKS_PER_SEC;
	deconRecords[7] = (float)(time2 - time1) / CLOCKS_PER_SEC;
	deconRecords[8] = (float)(time3 - time2) / CLOCKS_PER_SEC;
	deconRecords[9] = (float)(end - start) / CLOCKS_PER_SEC;
	return 0;
}

int decon_dualview(float *h_decon, float *h_img1, float *h_img2, unsigned int *imSize, float *h_psf1, float *h_psf2, 
	unsigned int *psfSize, int itNumForDecon, int deviceNum, int gpuMemMode, float *deconRecords, bool flagUnmatch, float *h_psf_bp1, float *h_psf_bp2){
	// gpuMemMode --> 0: Automatically set memory mode based on calculations; 1: sufficient memory; 2: memory optimized; 3: memory further optimized.
	//deconRecords: 10 elements
	//[0]:  the actual memory mode used;
	//[1] -[5]: initial GPU memory, after variables partially allocated, during processing, after processing, after variables released ( all in MB);
	//[6] -[9]: initializing time, prepocessing time, decon time, total time;
	float
		*h_StackA,
		*h_StackB,
		*h_StackE,
		*h_StackT,
		*d_StackA,
		*d_StackB,
		*d_StackE,
		*d_StackT;

	fComplex
		*h_PSFASpectrum,
		*h_PSFBSpectrum,
		*h_FlippedPSFASpectrum,
		*h_FlippedPSFBSpectrum,
		*h_StackESpectrum,
		*d_PSFSpectrum,
		*d_PSFASpectrum,
		*d_PSFBSpectrum,
		*d_FlippedPSFASpectrum,
		*d_FlippedPSFBSpectrum,
		*d_StackESpectrum;
	cufftHandle
		fftPlanFwd,
		fftPlanInv;
	// image size
	int
		imx, imy, imz;
	imx = imSize[0], imy = imSize[1], imz = imSize[2];

	// PSF size
	int
		PSFx, PSFy, PSFz;
	PSFx = psfSize[0], PSFy = psfSize[1], PSFz = psfSize[2];

	//FFT size
	int
		FFTx, FFTy, FFTz,
		PSFox, PSFoy, PSFoz,
		imox, imoy, imoz;

	FFTx = snapTransformSize(imx);// snapTransformSize(imx + PSFx - 1);
	FFTy = snapTransformSize(imy);// snapTransformSize(imy + PSFy - 1);
	FFTz = snapTransformSize(imz);// snapTransformSize(imz + PSFz - 1);
	// set original points for padding and cropping
	//fftz.y.z·
	PSFox = round(PSFx / 2);// round((FFTx - PSFx) / 2);
	PSFoy = round(PSFy / 2);//round((FFTy - PSFy) / 2);
	PSFoz = round(PSFz / 2);//round((FFTz - PSFz) / 2 );
	imox = round((FFTx - imSize[0]) / 2);
	imoy = round((FFTy - imSize[1]) / 2);
	imoz = round((FFTz - imSize[2]) / 2);
	/*
	printf("Image information:\n");
	printf("...Image size %d x %d x %d\n  ", imx, imy, imz);
	printf("...PSF size %d x %d x %d\n  ", PSFx, PSFy, PSFz);
	printf("...FFT size %d x %d x %d\n  ", FFTx, FFTy, FFTz);

	printf("...Output Image size %d x %d x %d \n   ", imSize[0], imSize[1], imSize[2]);
	*/
	// total pixel count for each images
	int totalSize = imx*imy*imz; // in floating format
	int totalSizePSF = PSFx*PSFy*PSFz; // in floating format
	int totalSizeFFT = FFTx*FFTy*FFTz; // in floating format
	int totalSizeSpectrum = FFTx * FFTy*(FFTz / 2 + 1); // in complex floating format
	int totalSizeMax = totalSizeSpectrum * 2; // in floating format
	int totalSizeMax2 = totalSizeMax > totalSizePSF ? totalSizeMax : totalSizePSF; // in floating format: in case PSF has a larger size
	// print GPU devices information
	cudaSetDevice(deviceNum);
	//****************** Processing Starts*****************//
	// variables for memory and time cost records
	clock_t start, time1, time2, time3, end;
	size_t totalMem = 0;
	size_t freeMem = 0;
	time1 = time2 = time3 = end = 0;
	start = clock();
	// allocate memory
	cudaMemGetInfo(&freeMem, &totalMem);
	//printf("...GPU free memory(at beginning) is %.0f MBites\n", (float)freeMem / 1048576.0f);
	deconRecords[1] = (float)freeMem / 1048576.0f;
	cudaMalloc((void **)&d_StackA, totalSizeMax2 *sizeof(float));
	cudaMalloc((void **)&d_StackE, totalSizeMax2 *sizeof(float));
	cudaMemset(d_StackA, 0, totalSizeMax2*sizeof(float));
	cudaMemset(d_StackE, 0, totalSizeMax2*sizeof(float));
	//check GPU status
	cudaCheckErrors("****Memory allocating fails... GPU out of memory !!!!*****");
	// Create FFT plans
	cufftPlan3d(&fftPlanFwd, FFTx, FFTy, FFTz, CUFFT_R2C);
	cufftPlan3d(&fftPlanInv, FFTx, FFTy, FFTz, CUFFT_C2R);
	cudaCheckErrors("****Memory allocating fails... GPU out of memory !!!!*****");

	cudaMemGetInfo(&freeMem, &totalMem);
	//printf("...GPU free memory(after partially mallocing) is %.0f MBites\n", (float)freeMem / 1048576.0f);
	deconRecords[2] = (float)freeMem / 1048576.0f;

	// ***** Set GPU memory use mode based on images size and available GPU memory ****
	// gpuMemMode --> Unified memory in next version???
	// 0: Automatically set memory mode based on calculations; 
	// 1: sufficient memory; 2: memory optimized; 3: memory further optimized.
	if (gpuMemMode == 0){ //Automatically set memory mode based on calculations.
		if (freeMem > 7 * totalSizeMax * sizeof(float)){ // 7 more GPU variables
			gpuMemMode = 1;
			//printf("\n GPU memory is sufficient, processing in efficient mode !!!\n");
		}
		else if (freeMem > 4 * totalSizeMax * sizeof(float)){// 4 more GPU variables
			gpuMemMode = 2;
			//printf("\n GPU memory is optimized, processing in memory saved mode !!!\n");
		}
		else {// no more GPU variables needed
			gpuMemMode = 3;
			//printf("\n GPU memory is futher optimized, processing in memory saved mode !!!\n");
		}
	}
	deconRecords[0] = gpuMemMode;
	double mySumPSF = 0;
	switch (gpuMemMode){
		case 1:// efficient GPU calculation
			time1 = clock();
			cudaMalloc((void **)&d_StackB, totalSizeFFT *sizeof(float));
			cudaMalloc((void **)&d_StackT, totalSizeFFT *sizeof(float));
			cudaMalloc((void **)&d_PSFASpectrum, totalSizeSpectrum*sizeof(fComplex));
			cudaMalloc((void **)&d_PSFBSpectrum, totalSizeSpectrum*sizeof(fComplex));
			cudaMalloc((void **)&d_FlippedPSFASpectrum, totalSizeSpectrum*sizeof(fComplex));
			cudaMalloc((void **)&d_FlippedPSFBSpectrum, totalSizeSpectrum*sizeof(fComplex));
			cudaMalloc((void **)&d_StackESpectrum, totalSizeSpectrum*sizeof(fComplex));
			cudaMemGetInfo(&freeMem, &totalMem);
			//printf("...GPU free memory(after mallocing) is %.0f MBites\n", (float)freeMem / 1048576.0f);
			// *** PSF Preparation
			//PSF A 
			cudaMemcpy(d_StackE, h_psf1, totalSizePSF* sizeof(float), cudaMemcpyHostToDevice);
			changestorageordergpu(d_StackA, d_StackE, PSFx, PSFy, PSFz, 1); //1: change tiff storage order to C storage order
			mySumPSF = sumcpu(h_psf1, totalSizePSF);
			multivaluegpu(d_StackE, d_StackA, (float)(1 / mySumPSF), PSFx, PSFy, PSFz); // normalize PSF sum to 1
			if (!flagUnmatch){ // traditional backprojector matched PSF 
				flipPSF(d_StackA, d_StackE, PSFx, PSFy, PSFz); // flip PSF
				cudaMemcpy(h_psf_bp1, d_StackA, totalSizePSF* sizeof(float), cudaMemcpyDeviceToHost);
			}
			cudaMemset(d_StackA, 0, totalSizeMax2*sizeof(float));
			padPSF(d_StackA, d_StackE, FFTx, FFTy, FFTz, PSFx, PSFy, PSFz, PSFox, PSFoy, PSFoz);
			cufftExecR2C(fftPlanFwd, (cufftReal *)d_StackA, (cufftComplex *)d_PSFASpectrum);
			//PSF B 
			cudaMemcpy(d_StackE, h_psf2, totalSizePSF* sizeof(float), cudaMemcpyHostToDevice);
			changestorageordergpu(d_StackA, d_StackE, PSFx, PSFy, PSFz, 1); //1: change tiff storage order to C storage order
			mySumPSF = sumcpu(h_psf2, totalSizePSF);
			multivaluegpu(d_StackE, d_StackA, (float)(1 / mySumPSF), PSFx, PSFy, PSFz); // normalize PSF sum to 1
			if (!flagUnmatch){ // traditional backprojector matched PSF 
				flipPSF(d_StackA, d_StackE, PSFx, PSFy, PSFz); // flip PSF
				cudaMemcpy(h_psf_bp2, d_StackA, totalSizePSF* sizeof(float), cudaMemcpyDeviceToHost);
			}
			cudaMemset(d_StackA, 0, totalSizeMax2*sizeof(float));
			padPSF(d_StackA, d_StackE, FFTx, FFTy, FFTz, PSFx, PSFy, PSFz, PSFox, PSFoy, PSFoz);
			cufftExecR2C(fftPlanFwd, (cufftReal *)d_StackA, (cufftComplex *)d_PSFBSpectrum);
			// PSF bp A
			cudaMemcpy(d_StackE, h_psf_bp1, totalSizePSF* sizeof(float), cudaMemcpyHostToDevice);
			if (flagUnmatch){
				changestorageordergpu(d_StackA, d_StackE, PSFx, PSFy, PSFz, 1); //1: change tiff storage order to C storage order
				mySumPSF = sumcpu(h_psf_bp1, totalSizePSF);
				multivaluegpu(d_StackE, d_StackA, (float)(1 / mySumPSF), PSFx, PSFy, PSFz); // normalize PSF sum to 1
			}
			cudaMemset(d_StackA, 0, totalSizeMax2*sizeof(float));
			padPSF(d_StackA, d_StackE, FFTx, FFTy, FFTz, PSFx, PSFy, PSFz, PSFox, PSFoy, PSFoz);
			cufftExecR2C(fftPlanFwd, (cufftReal *)d_StackA, (cufftComplex *)d_FlippedPSFASpectrum);
			// PSF bp B
			cudaMemcpy(d_StackE, h_psf_bp2, totalSizePSF* sizeof(float), cudaMemcpyHostToDevice);
			if (flagUnmatch){
				changestorageordergpu(d_StackA, d_StackE, PSFx, PSFy, PSFz, 1); //1: change tiff storage order to C storage order
				mySumPSF = sumcpu(h_psf_bp2, totalSizePSF);
				multivaluegpu(d_StackE, d_StackA, (float)(1 / mySumPSF), PSFx, PSFy, PSFz); // normalize PSF sum to 1
			}
			cudaMemset(d_StackA, 0, totalSizeMax2*sizeof(float));
			padPSF(d_StackA, d_StackE, FFTx, FFTy, FFTz, PSFx, PSFy, PSFz, PSFox, PSFoy, PSFoz);
			cufftExecR2C(fftPlanFwd, (cufftReal *)d_StackA, (cufftComplex *)d_FlippedPSFBSpectrum);
			cudaCheckErrors("PSF preparing fail");
			cudaMemGetInfo(&freeMem, &totalMem);
			//printf("...GPU free memory (during processing) is %.0f MBites\n", (float)freeMem / 1048576.0f);
			deconRecords[3] = (float)freeMem / 1048576.0f;
			// Prepare Stack Data
			cudaMemcpy(d_StackA, h_img1, totalSize* sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(d_StackB, h_img2, totalSize* sizeof(float), cudaMemcpyHostToDevice);
			//eliminate 0 in stacks
			maxvalue3Dgpu(d_StackA, d_StackA, (float)(SMALLVALUE), imx, imy, imz);
			maxvalue3Dgpu(d_StackB, d_StackB, (float)(SMALLVALUE), imx, imy, imz);
			/*
			double sumStackA = sum3Dgpu(d_StackA, d_2D, h_2D, imx, imy, imz);
			double sumStackB = sum3Dgpu(d_StackB, d_2D, h_2D, imx, imy, imz);
			//printf("Sum of Stack A: %.2f \n ", sumStackA);
			//printf("Sum of Stack B: %.2f \n ", sumStackB);
			multivaluegpu(d_StackB, d_StackB, (float)(sumStackA / sumStackB), imx, imy, imz);
			*/

			changestorageordergpu(d_StackE, d_StackA, imx, imy, imz, 1); //1: change tiff storage order to C storage order
			padStack(d_StackA, d_StackE, FFTx, FFTy, FFTz, imx, imy, imz, imox, imoy, imoz);//
			changestorageordergpu(d_StackE, d_StackB, imx, imy, imz, 1); //1: change tiff storage order to C storage order
			padStack(d_StackB, d_StackE, FFTx, FFTy, FFTz, imx, imy, imz, imox, imoy, imoz);//	

			// initialize estimation by average of StackA and StackB
			add3Dgpu(d_StackE, d_StackA, d_StackB, FFTx, FFTy, FFTz);
			multivaluegpu(d_StackE, d_StackE, (float)0.5, FFTx, FFTy, FFTz);
			cudaCheckErrors("image preparing fail");
			time2 = clock();
			// *****Joint deconvoultion	
			for (int itNum = 1; itNum <= itNumForDecon; itNum++){
				// ### iterate with StackA and PSFA///////////////////
				// convolve StackE with PSFA
				cufftExecR2C(fftPlanFwd, (cufftReal *)d_StackE, (cufftComplex *)d_StackESpectrum);
				multicomplex3Dgpu(d_StackESpectrum, d_StackESpectrum, d_PSFASpectrum, FFTx, FFTy, (FFTz / 2 + 1));
				cufftExecC2R(fftPlanInv, (cufftComplex *)d_StackESpectrum, (cufftReal *)d_StackT);
				// divid StackA by StackTemp
				div3Dgpu(d_StackT, d_StackA, d_StackT, FFTx, FFTy, FFTz);   //// div3Dgpu does not work
				// convolve StackTemp with FlippedPSFA
				cufftExecR2C(fftPlanFwd, (cufftReal *)d_StackT, (cufftComplex *)d_StackESpectrum);
				multicomplex3Dgpu(d_StackESpectrum, d_StackESpectrum, d_FlippedPSFASpectrum, FFTx, FFTy, (FFTz / 2 + 1));
				cufftExecC2R(fftPlanInv, (cufftComplex *)d_StackESpectrum, (cufftReal *)d_StackT);//test
				// multiply StackE and StackTemp
				multi3Dgpu(d_StackE, d_StackE, d_StackT, FFTx, FFTy, FFTz);//
				maxvalue3Dgpu(d_StackE, d_StackE, float(SMALLVALUE), FFTx, FFTy, FFTz);

				// ### iterate with StackB and PSFB /////////////////
				// convolve StackE with PSFB
				cufftExecR2C(fftPlanFwd, (cufftReal *)d_StackE, (cufftComplex *)d_StackESpectrum);//
				multicomplex3Dgpu(d_StackESpectrum, d_StackESpectrum, d_PSFBSpectrum, FFTx, FFTy, (FFTz / 2 + 1));
				cufftExecC2R(fftPlanInv, (cufftComplex *)d_StackESpectrum, (cufftReal *)d_StackT);
				// divid StackB by StackTemp
				div3Dgpu(d_StackT, d_StackB, d_StackT, FFTx, FFTy, FFTz);//
				// convolve StackTemp with FlippedPSFB
				cufftExecR2C(fftPlanFwd, (cufftReal *)d_StackT, (cufftComplex *)d_StackESpectrum);
				multicomplex3Dgpu(d_StackESpectrum, d_StackESpectrum, d_FlippedPSFBSpectrum, FFTx, FFTy, (FFTz / 2 + 1));
				cufftExecC2R(fftPlanInv, (cufftComplex *)d_StackESpectrum, (cufftReal *)d_StackT);
				// multiply StackE and StackTemp
				multi3Dgpu(d_StackE, d_StackE, d_StackT, FFTx, FFTy, FFTz);
				maxvalue3Dgpu(d_StackE, d_StackE, float(SMALLVALUE), FFTx, FFTy, FFTz);
			}
			cropStack(d_StackE, d_StackT, FFTx, FFTy, FFTz, imx, imy, imz, imox, imoy, imoz);//
			//printf("...Deconvolution completed ! ! !\n");
			cudaDeviceSynchronize();
			changestorageordergpu(d_StackE, d_StackT, imx, imy, imz, -1); //-1: change C storage order to tiff storage order
			cudaMemcpy(h_decon, d_StackE, totalSize* sizeof(float), cudaMemcpyDeviceToHost);
			time3 = clock();
			cudaMemGetInfo(&freeMem, &totalMem);
			//printf("...GPU free memory (after processing) is %.0f MBites\n", (float)freeMem / 1048576.0f);
			deconRecords[4] = (float)freeMem / 1048576.0f;
			// release CUDA variables
			cudaFree(d_StackB); cudaFree(d_StackT); cudaFree(d_PSFASpectrum); cudaFree(d_PSFBSpectrum);
			cudaFree(d_FlippedPSFASpectrum); cudaFree(d_FlippedPSFBSpectrum); cudaFree(d_StackESpectrum);
			break;
		case 2: // memory saved mode 2
			time1 = clock();
			h_PSFASpectrum = (fComplex *)malloc(totalSizeSpectrum*sizeof(fComplex));
			h_PSFBSpectrum = (fComplex *)malloc(totalSizeSpectrum*sizeof(fComplex));
			h_FlippedPSFASpectrum = (fComplex *)malloc(totalSizeSpectrum*sizeof(fComplex));
			h_FlippedPSFBSpectrum = (fComplex *)malloc(totalSizeSpectrum*sizeof(fComplex));
			cudaMalloc((void **)&d_StackB, totalSizeMax *sizeof(float));
			cudaMalloc((void **)&d_StackT, totalSizeMax *sizeof(float));
			cudaMalloc((void **)&d_StackESpectrum, totalSizeSpectrum*sizeof(fComplex));
			cudaMalloc((void **)&d_PSFSpectrum, totalSizeSpectrum*sizeof(fComplex));
			d_PSFASpectrum = d_PSFSpectrum;
			d_PSFBSpectrum = d_PSFSpectrum;
			d_FlippedPSFASpectrum = d_PSFSpectrum;
			d_FlippedPSFBSpectrum = d_PSFSpectrum;

			// *** PSF Preparation
			//PSF A 
			cudaMemcpy(d_StackE, h_psf1, totalSizePSF* sizeof(float), cudaMemcpyHostToDevice);
			changestorageordergpu(d_StackA, d_StackE, PSFx, PSFy, PSFz, 1); //1: change tiff storage order to C storage order
			mySumPSF = sumcpu(h_psf1, totalSizePSF);
			multivaluegpu(d_StackE, d_StackA, (float)(1 / mySumPSF), PSFx, PSFy, PSFz); // normalize PSF sum to 1
			if (!flagUnmatch){ // traditional backprojector matched PSF 
				flipPSF(d_StackA, d_StackE, PSFx, PSFy, PSFz); // flip PSF
				cudaMemcpy(h_psf_bp1, d_StackA, totalSizePSF* sizeof(float), cudaMemcpyDeviceToHost);
			}
			cudaMemset(d_StackA, 0, totalSizeMax2*sizeof(float));
			padPSF(d_StackA, d_StackE, FFTx, FFTy, FFTz, PSFx, PSFy, PSFz, PSFox, PSFoy, PSFoz);
			cufftExecR2C(fftPlanFwd, (cufftReal *)d_StackA, (cufftComplex *)d_StackESpectrum);
			cudaMemcpy(h_PSFASpectrum, d_StackESpectrum, totalSizeSpectrum*sizeof(fComplex), cudaMemcpyDeviceToHost);
			//PSF B 
			cudaMemcpy(d_StackE, h_psf2, totalSizePSF* sizeof(float), cudaMemcpyHostToDevice);
			changestorageordergpu(d_StackA, d_StackE, PSFx, PSFy, PSFz, 1); //1: change tiff storage order to C storage order
			mySumPSF = sumcpu(h_psf2, totalSizePSF);
			multivaluegpu(d_StackE, d_StackA, (float)(1 / mySumPSF), PSFx, PSFy, PSFz); // normalize PSF sum to 1
			if (!flagUnmatch){ // traditional backprojector matched PSF 
				flipPSF(d_StackA, d_StackE, PSFx, PSFy, PSFz); // flip PSF
				cudaMemcpy(h_psf_bp2, d_StackA, totalSizePSF* sizeof(float), cudaMemcpyDeviceToHost);
			}
			cudaMemset(d_StackA, 0, totalSizeMax2*sizeof(float));
			padPSF(d_StackA, d_StackE, FFTx, FFTy, FFTz, PSFx, PSFy, PSFz, PSFox, PSFoy, PSFoz);
			cufftExecR2C(fftPlanFwd, (cufftReal *)d_StackA, (cufftComplex *)d_StackESpectrum);
			cudaMemcpy(h_PSFBSpectrum, d_StackESpectrum, totalSizeSpectrum*sizeof(fComplex), cudaMemcpyDeviceToHost);
			// PSF bp A
			cudaMemcpy(d_StackE, h_psf_bp1, totalSizePSF* sizeof(float), cudaMemcpyHostToDevice);
			if (flagUnmatch){
				changestorageordergpu(d_StackA, d_StackE, PSFx, PSFy, PSFz, 1); //1: change tiff storage order to C storage order
				mySumPSF = sumcpu(h_psf_bp1, totalSizePSF);
				multivaluegpu(d_StackE, d_StackA, (float)(1 / mySumPSF), PSFx, PSFy, PSFz); // normalize PSF sum to 1
			}
			cudaMemset(d_StackA, 0, totalSizeMax2*sizeof(float));
			padPSF(d_StackA, d_StackE, FFTx, FFTy, FFTz, PSFx, PSFy, PSFz, PSFox, PSFoy, PSFoz);
			cufftExecR2C(fftPlanFwd, (cufftReal *)d_StackA, (cufftComplex *)d_StackESpectrum);
			cudaMemcpy(h_FlippedPSFASpectrum, d_StackESpectrum, totalSizeSpectrum*sizeof(fComplex), cudaMemcpyDeviceToHost);
			// PSF bp B
			cudaMemcpy(d_StackE, h_psf_bp2, totalSizePSF* sizeof(float), cudaMemcpyHostToDevice);
			if (flagUnmatch){
				changestorageordergpu(d_StackA, d_StackE, PSFx, PSFy, PSFz, 1); //1: change tiff storage order to C storage order
				mySumPSF = sumcpu(h_psf_bp2, totalSizePSF);
				multivaluegpu(d_StackE, d_StackA, (float)(1 / mySumPSF), PSFx, PSFy, PSFz); // normalize PSF sum to 1
			}
			cudaMemset(d_StackA, 0, totalSizeMax2*sizeof(float));
			padPSF(d_StackA, d_StackE, FFTx, FFTy, FFTz, PSFx, PSFy, PSFz, PSFox, PSFoy, PSFoz);
			cufftExecR2C(fftPlanFwd, (cufftReal *)d_StackA, (cufftComplex *)d_StackESpectrum);
			cudaMemcpy(h_FlippedPSFBSpectrum, d_StackESpectrum, totalSizeSpectrum*sizeof(fComplex), cudaMemcpyDeviceToHost);
			cudaCheckErrors("PSF preparing fail");
			cudaMemGetInfo(&freeMem, &totalMem);
			//printf("...GPU free memory (during processing) is %.0f MBites\n", (float)freeMem / 1048576.0f);
			deconRecords[3] = (float)freeMem / 1048576.0f;
			// Prepare Stack Data
			//eliminate 0 in stacks
			cudaMemcpy(d_StackA, h_img1, totalSize* sizeof(float), cudaMemcpyHostToDevice);
			maxvalue3Dgpu(d_StackA, d_StackA, (float)(SMALLVALUE), imx, imy, imz);
			changestorageordergpu(d_StackE, d_StackA, imx, imy, imz, 1); //1: change tiff storage order to C storage order
			padStack(d_StackA, d_StackE, FFTx, FFTy, FFTz, imx, imy, imz, imox, imoy, imoz);//

			cudaMemcpy(d_StackB, h_img2, totalSize* sizeof(float), cudaMemcpyHostToDevice);
			maxvalue3Dgpu(d_StackB, d_StackB, (float)(SMALLVALUE), imx, imy, imz);
			changestorageordergpu(d_StackE, d_StackB, imx, imy, imz, 1); //1: change tiff storage order to C storage order
			padStack(d_StackB, d_StackE, FFTx, FFTy, FFTz, imx, imy, imz, imox, imoy, imoz);//

			// initialize estimation by average of StackA and StackB
			add3Dgpu(d_StackE, d_StackA, d_StackB, FFTx, FFTy, FFTz);
			multivaluegpu(d_StackE, d_StackE, (float)0.5, FFTx, FFTy, FFTz);
			cudaCheckErrors("image preparing fail");
			time2 = clock();
			for (int itNum = 1; itNum <= itNumForDecon; itNum++){
				// ### iterate with StackA and PSFA///////////////////
				// convolve StackE with PSFA
				cufftExecR2C(fftPlanFwd, (cufftReal *)d_StackE, (cufftComplex *)d_StackESpectrum);
				cudaMemcpy(d_PSFASpectrum, h_PSFASpectrum, FFTx*FFTy*(FFTz / 2 + 1)*sizeof(fComplex), cudaMemcpyHostToDevice);
				multicomplex3Dgpu(d_StackESpectrum, d_StackESpectrum, d_PSFASpectrum, FFTx, FFTy, (FFTz / 2 + 1));
				cufftExecC2R(fftPlanInv, (cufftComplex *)d_StackESpectrum, (cufftReal *)d_StackT);
				// divid StackA by StackTemp
				div3Dgpu(d_StackT, d_StackA, d_StackT, FFTx, FFTy, FFTz);   //// div3Dgpu does not work
				// convolve StackTemp with FlippedPSFA
				cufftExecR2C(fftPlanFwd, (cufftReal *)d_StackT, (cufftComplex *)d_StackESpectrum);
				cudaMemcpy(d_FlippedPSFASpectrum, h_FlippedPSFASpectrum, FFTx*FFTy*(FFTz / 2 + 1)*sizeof(fComplex), cudaMemcpyHostToDevice);
				multicomplex3Dgpu(d_StackESpectrum, d_StackESpectrum, d_FlippedPSFASpectrum, FFTx, FFTy, (FFTz / 2 + 1));
				cufftExecC2R(fftPlanInv, (cufftComplex *)d_StackESpectrum, (cufftReal *)d_StackT);//test
				// multiply StackE and StackTemp
				multi3Dgpu(d_StackE, d_StackE, d_StackT, FFTx, FFTy, FFTz);//
				maxvalue3Dgpu(d_StackE, d_StackE, float(SMALLVALUE), FFTx, FFTy, FFTz);

				// ### iterate with StackB and PSFB /////////////////
				// convolve StackE with PSFB
				cufftExecR2C(fftPlanFwd, (cufftReal *)d_StackE, (cufftComplex *)d_StackESpectrum);//
				cudaMemcpy(d_PSFBSpectrum, h_PSFBSpectrum, FFTx*FFTy*(FFTz / 2 + 1)*sizeof(fComplex), cudaMemcpyHostToDevice);
				multicomplex3Dgpu(d_StackESpectrum, d_StackESpectrum, d_PSFBSpectrum, FFTx, FFTy, (FFTz / 2 + 1));
				cufftExecC2R(fftPlanInv, (cufftComplex *)d_StackESpectrum, (cufftReal *)d_StackT);
				// divid StackB by StackTemp
				div3Dgpu(d_StackT, d_StackB, d_StackT, FFTx, FFTy, FFTz);//
				// convolve StackTemp with FlippedPSFB
				cufftExecR2C(fftPlanFwd, (cufftReal *)d_StackT, (cufftComplex *)d_StackESpectrum);
				cudaMemcpy(d_FlippedPSFBSpectrum, h_FlippedPSFBSpectrum, FFTx*FFTy*(FFTz / 2 + 1)*sizeof(fComplex), cudaMemcpyHostToDevice);
				multicomplex3Dgpu(d_StackESpectrum, d_StackESpectrum, d_FlippedPSFBSpectrum, FFTx, FFTy, (FFTz / 2 + 1));
				cufftExecC2R(fftPlanInv, (cufftComplex *)d_StackESpectrum, (cufftReal *)d_StackT);
				// multiply StackE and StackTemp
				multi3Dgpu(d_StackE, d_StackE, d_StackT, FFTx, FFTy, FFTz);
				maxvalue3Dgpu(d_StackE, d_StackE, float(SMALLVALUE), FFTx, FFTy, FFTz);

			}
			cropStack(d_StackE, d_StackT, FFTx, FFTy, FFTz, imx, imy, imz, imox, imoy, imoz);//
			cudaDeviceSynchronize();
			changestorageordergpu(d_StackE, d_StackT, imx, imy, imz, -1); //-1: change C storage order to tiff storage order
			cudaMemcpy(h_decon, d_StackE, totalSize* sizeof(float), cudaMemcpyDeviceToHost);
			time3 = clock();
			cudaMemGetInfo(&freeMem, &totalMem);
			//printf("...GPU free memory (after processing) is %.0f MBites\n", (float)freeMem / 1048576.0f);
			deconRecords[4] = (float)freeMem / 1048576.0f;
			// release CPU and GPU memory
			free(h_PSFASpectrum); free(h_PSFBSpectrum); free(h_FlippedPSFASpectrum); free(h_FlippedPSFBSpectrum);
			cudaFree(d_StackB); cudaFree(d_StackT); cudaFree(d_PSFSpectrum); cudaFree(d_StackESpectrum);
			break;
		case 3: // memory saved mode 3
			time1 = clock();
			h_StackA = (float *)malloc(totalSizeFFT * sizeof(float));
			h_StackB = (float *)malloc(totalSizeFFT * sizeof(float));
			h_StackE = (float *)malloc(totalSizeFFT * sizeof(float));
			h_StackT = (float *)malloc(totalSizeFFT * sizeof(float));
			h_PSFASpectrum = (fComplex *)malloc(totalSizeSpectrum*sizeof(fComplex));
			h_PSFBSpectrum = (fComplex *)malloc(totalSizeSpectrum*sizeof(fComplex));
			h_FlippedPSFASpectrum = (fComplex *)malloc(totalSizeSpectrum*sizeof(fComplex));
			h_FlippedPSFBSpectrum = (fComplex *)malloc(totalSizeSpectrum*sizeof(fComplex));
			h_StackESpectrum = (fComplex *)malloc(totalSizeSpectrum*sizeof(fComplex));

			d_StackESpectrum = (fComplex *)d_StackA;
			d_PSFSpectrum = (fComplex *)d_StackE;
			// *** PSF Preparation
			//PSF A 
			cudaMemcpy(d_StackE, h_psf1, totalSizePSF* sizeof(float), cudaMemcpyHostToDevice);
			changestorageordergpu(d_StackA, d_StackE, PSFx, PSFy, PSFz, 1); //1: change tiff storage order to C storage order
			mySumPSF = sumcpu(h_psf1, totalSizePSF);
			multivaluegpu(d_StackE, d_StackA, (float)(1 / mySumPSF), PSFx, PSFy, PSFz); // normalize PSF sum to 1
			if (!flagUnmatch){ // traditional backprojector matched PSF 
				flipPSF(d_StackA, d_StackE, PSFx, PSFy, PSFz); // flip PSF
				cudaMemcpy(h_psf_bp1, d_StackA, totalSizePSF* sizeof(float), cudaMemcpyDeviceToHost);
			}
			cudaMemset(d_StackA, 0, totalSizeMax2*sizeof(float));
			padPSF(d_StackA, d_StackE, FFTx, FFTy, FFTz, PSFx, PSFy, PSFz, PSFox, PSFoy, PSFoz);
			cufftExecR2C(fftPlanFwd, (cufftReal *)d_StackA, (cufftComplex *)d_PSFSpectrum);
			cudaMemcpy(h_PSFASpectrum, d_PSFSpectrum, totalSizeSpectrum*sizeof(fComplex), cudaMemcpyDeviceToHost);
			//PSF B 
			cudaMemcpy(d_StackE, h_psf2, totalSizePSF* sizeof(float), cudaMemcpyHostToDevice);
			changestorageordergpu(d_StackA, d_StackE, PSFx, PSFy, PSFz, 1); //1: change tiff storage order to C storage order
			mySumPSF = sumcpu(h_psf2, totalSizePSF);
			multivaluegpu(d_StackE, d_StackA, (float)(1 / mySumPSF), PSFx, PSFy, PSFz); // normalize PSF sum to 1
			if (!flagUnmatch){ // traditional backprojector matched PSF 
				flipPSF(d_StackA, d_StackE, PSFx, PSFy, PSFz); // flip PSF
				cudaMemcpy(h_psf_bp2, d_StackA, totalSizePSF* sizeof(float), cudaMemcpyDeviceToHost);
			}
			cudaMemset(d_StackA, 0, totalSizeMax2*sizeof(float));
			padPSF(d_StackA, d_StackE, FFTx, FFTy, FFTz, PSFx, PSFy, PSFz, PSFox, PSFoy, PSFoz);
			cufftExecR2C(fftPlanFwd, (cufftReal *)d_StackA, (cufftComplex *)d_PSFSpectrum);
			cudaMemcpy(h_PSFBSpectrum, d_PSFSpectrum, totalSizeSpectrum*sizeof(fComplex), cudaMemcpyDeviceToHost);
			// PSF bp A
			cudaMemcpy(d_StackE, h_psf_bp1, totalSizePSF* sizeof(float), cudaMemcpyHostToDevice);
			if (flagUnmatch){
				changestorageordergpu(d_StackA, d_StackE, PSFx, PSFy, PSFz, 1); //1: change tiff storage order to C storage order
				mySumPSF = sumcpu(h_psf_bp1, totalSizePSF);
				multivaluegpu(d_StackE, d_StackA, (float)(1 / mySumPSF), PSFx, PSFy, PSFz); // normalize PSF sum to 1
			}
			cudaMemset(d_StackA, 0, totalSizeMax2*sizeof(float));
			padPSF(d_StackA, d_StackE, FFTx, FFTy, FFTz, PSFx, PSFy, PSFz, PSFox, PSFoy, PSFoz);
			cufftExecR2C(fftPlanFwd, (cufftReal *)d_StackA, (cufftComplex *)d_PSFSpectrum);
			cudaMemcpy(h_FlippedPSFASpectrum, d_PSFSpectrum, totalSizeSpectrum*sizeof(fComplex), cudaMemcpyDeviceToHost);
			// PSF bp B
			cudaMemcpy(d_StackE, h_psf_bp2, totalSizePSF* sizeof(float), cudaMemcpyHostToDevice);
			if (flagUnmatch){
				changestorageordergpu(d_StackA, d_StackE, PSFx, PSFy, PSFz, 1); //1: change tiff storage order to C storage order
				mySumPSF = sumcpu(h_psf_bp2, totalSizePSF);
				multivaluegpu(d_StackE, d_StackA, (float)(1 / mySumPSF), PSFx, PSFy, PSFz); // normalize PSF sum to 1
			}
			cudaMemset(d_StackA, 0, totalSizeMax2*sizeof(float));
			padPSF(d_StackA, d_StackE, FFTx, FFTy, FFTz, PSFx, PSFy, PSFz, PSFox, PSFoy, PSFoz);
			cufftExecR2C(fftPlanFwd, (cufftReal *)d_StackA, (cufftComplex *)d_PSFSpectrum);
			cudaMemcpy(h_FlippedPSFBSpectrum, d_PSFSpectrum, totalSizeSpectrum*sizeof(fComplex), cudaMemcpyDeviceToHost);
			cudaCheckErrors("PSF preparing fail");
			cudaMemGetInfo(&freeMem, &totalMem);
			//printf("...GPU free memory (during processing) is %.0f MBites\n", (float)freeMem / 1048576.0f);
			deconRecords[3] = (float)freeMem / 1048576.0f;
			// Prepare Stack Data
			//eliminate 0 in stacks
			cudaMemcpy(d_StackA, h_img1, totalSize* sizeof(float), cudaMemcpyHostToDevice);
			maxvalue3Dgpu(d_StackA, d_StackA, (float)(SMALLVALUE), imx, imy, imz);
			changestorageordergpu(d_StackE, d_StackA, imx, imy, imz, 1); //1: change tiff storage order to C storage order
			padStack(d_StackA, d_StackE, FFTx, FFTy, FFTz, imx, imy, imz, imox, imoy, imoz);//
			cudaMemcpy(h_StackA, d_StackA, totalSizeFFT * sizeof(float), cudaMemcpyDeviceToHost);

			cudaMemcpy(d_StackA, h_img2, totalSize* sizeof(float), cudaMemcpyHostToDevice);
			maxvalue3Dgpu(d_StackA, d_StackA, (float)(SMALLVALUE), imx, imy, imz);
			changestorageordergpu(d_StackE, d_StackA, imx, imy, imz, 1); //1: change tiff storage order to C storage order
			padStack(d_StackA, d_StackE, FFTx, FFTy, FFTz, imx, imy, imz, imox, imoy, imoz);//
			cudaMemcpy(h_StackB, d_StackA, totalSizeFFT * sizeof(float), cudaMemcpyDeviceToHost);

			// initialize estimation by average of StackA and StackB
			cudaMemcpy(d_StackE, h_StackA, totalSizeFFT * sizeof(float), cudaMemcpyHostToDevice);
			add3Dgpu(d_StackE, d_StackA, d_StackE, FFTx, FFTy, FFTz);
			multivaluegpu(d_StackE, d_StackE, (float)0.5, FFTx, FFTy, FFTz);
			cudaMemcpy(h_StackE, d_StackE, totalSizeFFT * sizeof(float), cudaMemcpyDeviceToHost);
			cudaCheckErrors("image preparing fail");
			time2 = clock();
			for (int itNum = 1; itNum <= itNumForDecon; itNum++){
				//printf("...Processing iteration %d\n", it);
				// ### iterate with StackA and PSFA///////////////////
				// convolve StackE with PSFA
				cufftExecR2C(fftPlanFwd, (cufftReal *)d_StackE, (cufftComplex *)d_StackESpectrum);
				cudaMemcpy(h_StackE, d_StackE, totalSizeFFT * sizeof(float), cudaMemcpyDeviceToHost);

				cudaMemcpy(d_PSFSpectrum, h_PSFASpectrum, totalSizeSpectrum*sizeof(fComplex), cudaMemcpyHostToDevice);
				multicomplex3Dgpu(d_StackESpectrum, d_StackESpectrum, d_PSFSpectrum, FFTx, FFTy, (FFTz / 2 + 1));
				cufftExecC2R(fftPlanInv, (cufftComplex *)d_StackESpectrum, (cufftReal *)d_StackE);

				// divid StackA by StackTemp
				cudaMemcpy(d_StackA, h_StackA, totalSizeFFT* sizeof(float), cudaMemcpyHostToDevice);
				div3Dgpu(d_StackE, d_StackA, d_StackE, FFTx, FFTy, FFTz);   //// div3Dgpu does not work
				// convolve StackTemp with FlippedPSFA
				cufftExecR2C(fftPlanFwd, (cufftReal *)d_StackE, (cufftComplex *)d_StackESpectrum);
				cudaMemcpy(d_PSFSpectrum, h_FlippedPSFASpectrum, totalSizeSpectrum*sizeof(fComplex), cudaMemcpyHostToDevice);
				multicomplex3Dgpu(d_StackESpectrum, d_StackESpectrum, d_PSFSpectrum, FFTx, FFTy, (FFTz / 2 + 1));
				cufftExecC2R(fftPlanInv, (cufftComplex *)d_StackESpectrum, (cufftReal *)d_StackE);//test
				// multiply StackE and StackTemp
				cudaMemcpy(d_StackA, h_StackE, totalSizeFFT* sizeof(float), cudaMemcpyHostToDevice);
				multi3Dgpu(d_StackE, d_StackE, d_StackA, FFTx, FFTy, FFTz);//
				maxvalue3Dgpu(d_StackE, d_StackE, float(SMALLVALUE), FFTx, FFTy, FFTz);

				// ### iterate with StackB and PSFB /////////////////
				// convolve StackE with PSFB		
				cufftExecR2C(fftPlanFwd, (cufftReal *)d_StackE, (cufftComplex *)d_StackESpectrum);//
				cudaMemcpy(h_StackE, d_StackE, totalSizeFFT * sizeof(float), cudaMemcpyDeviceToHost);

				cudaMemcpy(d_PSFSpectrum, h_PSFBSpectrum, totalSizeSpectrum*sizeof(fComplex), cudaMemcpyHostToDevice);
				multicomplex3Dgpu(d_StackESpectrum, d_StackESpectrum, d_PSFSpectrum, FFTx, FFTy, (FFTz / 2 + 1));
				cufftExecC2R(fftPlanInv, (cufftComplex *)d_StackESpectrum, (cufftReal *)d_StackE);
				// divid StackB by StackTemp
				cudaMemcpy(d_StackA, h_StackB, totalSizeFFT* sizeof(float), cudaMemcpyHostToDevice);
				div3Dgpu(d_StackE, d_StackA, d_StackE, FFTx, FFTy, FFTz);//

				// convolve StackTemp with FlippedPSFB
				cufftExecR2C(fftPlanFwd, (cufftReal *)d_StackE, (cufftComplex *)d_StackESpectrum);
				cudaMemcpy(d_PSFSpectrum, h_FlippedPSFBSpectrum, totalSizeSpectrum*sizeof(fComplex), cudaMemcpyHostToDevice);
				multicomplex3Dgpu(d_StackESpectrum, d_StackESpectrum, d_PSFSpectrum, FFTx, FFTy, (FFTz / 2 + 1));
				cufftExecC2R(fftPlanInv, (cufftComplex *)d_StackESpectrum, (cufftReal *)d_StackE);
				// multiply StackE and StackTemp
				cudaMemcpy(d_StackA, h_StackE, totalSizeFFT* sizeof(float), cudaMemcpyHostToDevice);
				multi3Dgpu(d_StackE, d_StackE, d_StackA, FFTx, FFTy, FFTz);
				maxvalue3Dgpu(d_StackE, d_StackE, float(SMALLVALUE), FFTx, FFTy, FFTz);
			}
			cropStack(d_StackE, d_StackA, FFTx, FFTy, FFTz, imx, imy, imz, imox, imoy, imoz);//
			cudaDeviceSynchronize();
			changestorageordergpu(d_StackE, d_StackA, imx, imy, imz, -1); //-1: change C storage order to tiff storage order
			cudaMemcpy(h_decon, d_StackE, totalSize* sizeof(float), cudaMemcpyDeviceToHost);
			time3 = clock();
			cudaMemGetInfo(&freeMem, &totalMem);
			//printf("...GPU free memory (after processing) is %.0f MBites\n", (float)freeMem / 1048576.0f);
			deconRecords[4] = (float)freeMem / 1048576.0f;
			// release CPU memory
			free(h_StackA); free(h_StackB); free(h_StackE); free(h_StackT); free(h_PSFASpectrum);
			free(h_PSFBSpectrum); free(h_FlippedPSFASpectrum); free(h_FlippedPSFBSpectrum); free(h_StackESpectrum);
			break;
		default:
			//printf("\n****Wrong gpuMemMode setup, no deconvolution performed !!! ****\n");
			return -1;
	}
	// release GPU memory
	cudaFree(d_StackA);
	cudaFree(d_StackE);
	// destroy plans
	cufftDestroy(fftPlanFwd);
	cufftDestroy(fftPlanInv);
	end = clock();
	cudaMemGetInfo(&freeMem, &totalMem);
	//printf("GPU free memory (after variable released): %.0f MBites\n", (float)freeMem / 1048576.0f);
	deconRecords[5] = (float)freeMem / 1048576.0f;
	deconRecords[6] = (float)(time1 - start) / CLOCKS_PER_SEC;
	deconRecords[7] = (float)(time2 - time1) / CLOCKS_PER_SEC;
	deconRecords[8] = (float)(time3 - time2) / CLOCKS_PER_SEC;
	deconRecords[9] = (float)(end - start) / CLOCKS_PER_SEC;
	return 0;
}

//// 3D fusion: registration and deconvolution
int fusion_dualview(float *h_decon, float *h_reg, float *iTmx, float *h_img1, float *h_img2, unsigned int *imSizeIn1, unsigned int *imSizeIn2, 
	float *pixelSize1, float *pixelSize2, int imRotation, int regMethod, int flagInitialTmx, float FTOL, int itLimit, float *h_psf1, float *h_psf2, 
	unsigned int *psfSizeIn, int itNumForDecon, int deviceNum, int gpuMemMode, float *fusionRecords, bool flagUnmatch, float *h_psf_bp1, float *h_psf_bp2){
	// imBRotation: image B rotation--> 0: no rotation; 1: 90deg rotation by y axis ; -1: -90deg rotation by y axis ;
	//
	// regMethod --> 0: no registration, transform imageB based on input matrix;1: translation only; 2: rigid body; 
	//  3: 7 degrees of freedom (translation, rotation, scaling equally in 3 dimemsions)  4: 9 degrees of freedom(translation, rotation, scaling); 
	//  5: 12 degrees of freedom; 6:rigid body first, then do 12 degrees of freedom; 7:3 DOF --> 6 DOF --> 9 DOF --> 12 DOF
	//
	// flagInitialTmx --> 0: default matrix; 1: use input matrix ; 2: translation based on phase image; 3: do 2D registration or alignment;
	//
	//fusionRecords --> 0-10: regRecords; 11-20: deconRecords; 21: total time;
	//[0] -[3]: initial GPU memory, after variables allocated, after processing, after variables released ( all in MB)
	//[4] -[6]: initial cost function value, minimized cost function value, intermediate cost function value
	//[7] -[10]: registration time (in s), whole time (in s), single sub iteration time (in ms), total sub iterations
	//[11] -[10]: decon time, whole time, single sub iteration time (in ms), total sub iterations
	//
	// flagUnmatach --> 1: use unmatch back projector; 0: use traditional backprojector (flipped PSF)


	//************get basic input images information ******************	

	//****************** calculate images' size *************************//
	int imx, imy, imz;
	unsigned int imSize[3], imSize1[3], imSize2[3], imSizeTemp[3];
	float pixelSize[3], pixelSizeTemp[3];
	bool flagInterp1 = true, flagInterp2 = true;
	int inputTmx = 0;
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
	int
		PSFx, PSFy, PSFz;
	PSFx = psfSizeIn[0], PSFy = psfSizeIn[1], PSFz = psfSizeIn[2];

	//FFT size
	int
		FFTx, FFTy, FFTz;

	FFTx = snapTransformSize(imx);// snapTransformSize(imx + PSFx - 1);
	FFTy = snapTransformSize(imy);// snapTransformSize(imy + PSFy - 1);
	FFTz = snapTransformSize(imz);// snapTransformSize(imz + PSFz - 1);

	// total pixel count for each images
	int totalSizeIn1 = imSizeIn1[0] * imSizeIn1[1] * imSizeIn1[2]; // in floating format
	int totalSizeIn2 = imSizeIn2[0] * imSizeIn2[1] * imSizeIn2[2]; // in floating format
	int totalSize1 = imx1*imy1*imz1; // in floating format
	int totalSize2 = imx2*imy2*imz2; // in floating format
	int totalSize = totalSize1; // in floating format
	int totalSizeFFT = FFTx*FFTy*(FFTz / 2 + 1); // in complex floating format
	int totalSize12 = (totalSize1 > totalSize2) ? totalSize1 : totalSize2;
	int totalSizeMax = (totalSize1 > totalSizeFFT * 2) ? totalSize1 : totalSizeFFT * 2; // in floating format
	
	// print GPU devices information
	cudaSetDevice(deviceNum);
	//****************** Processing Starts*****************//
	// variables for memory and time cost records
	clock_t startWhole, endWhole;
	size_t totalMem = 0;
	size_t freeMem = 0;
	float gpuTimeCost = 0;
	cudaMemGetInfo(&freeMem, &totalMem);
	printf("...GPU free memory at beginning is %.0f MBites\n", (float)freeMem / 1048576.0f);
	startWhole = clock();
	// ***** Set GPU memory use mode based on images size and available GPU memory ****
	// gpuMemMode--> -1: not enough gpu memory; 0-3: enough GPU memory; 
	if (freeMem < 4 * totalSizeMax * sizeof(float)){ // 2 variables + 2 FFT calculation space + a few additional space
		gpuMemMode = -1;
		printf("\n Available GPU memory is insufficient, processing terminated !!!\n");
	}
	if (gpuMemMode < 0) return -1;

	// ************** Registration *************
	// ***interpolation and rotation
	float
		*h_StackA,
		*h_StackB;
	float
		*d_imgE;
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
	cudaArray *d_Array1, *d_Array2;
	float *h_aff12 = (float *)malloc((NDIM)* sizeof(float));
	h_StackA = (float *)malloc(totalSize12 * sizeof(float));
	h_StackB = (float *)malloc(totalSize12 * sizeof(float));
	cudaMalloc((void **)&d_img3D, totalSize12 *sizeof(float));
	if ((imRotation == 1) || (imRotation == -1))
		cudaMalloc((void **)&d_imgE, totalSizeIn1 * sizeof(float));
	if (flagInterp1)
		cudaMalloc3DArray(&d_Array1, &channelDesc, make_cudaExtent(imSizeIn1[0], imSizeIn1[1], imSizeIn1[2]));
	if (flagInterp2)
		cudaMalloc3DArray(&d_Array2, &channelDesc, make_cudaExtent(imSizeIn2[0], imSizeIn2[1], imSizeIn2[2]));
	/// image 1
	if (flagInterp1){
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
	cudaDeviceSynchronize();

	// image 2
	// rotation
	if ((imRotation == 1) || (imRotation == -1)){
		cudaMemcpy(d_imgE, h_img2, totalSizeIn2 * sizeof(float), cudaMemcpyHostToDevice);
		rotbyyaxis(d_img3D, d_imgE, imSizeIn2[0], imSizeIn2[1], imSizeIn2[2], imRotation);
		cudaMemcpy(h_StackB, d_img3D, imSizeIn2[0] * imSizeIn2[1] * imSizeIn2[2] * sizeof(float), cudaMemcpyDeviceToHost);
		cudaFree(d_imgE);
	}
	if (flagInterp2){
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
	cudaDeviceSynchronize();

	// ***** Do 2D registration or alignment
	// only if 3D registration is set and flagInitialTmx = 2
	if (regMethod && (flagInitialTmx==3)){
		float *tmx1 = (float *)malloc(6 * sizeof(float));
		float *tmx2 = (float *)malloc(6 * sizeof(float));
		int img2DSizeMax1 = ((imx1 * imy1) > (imz1 * imx1)) ? (imx1 * imy1) : (imz1 * imx1);
		int img2DSizeMax2 = ((imx2 * imy2) > (imz2 * imx2)) ? (imx2 * imy2) : (imz2 * imx2);
		int img2DSizeMax = (img2DSizeMax1 > img2DSizeMax2) ? img2DSizeMax1 : img2DSizeMax2;
		float *h_img2D1 = (float *)malloc(img2DSizeMax1 * sizeof(float));
		float *h_img2D2 = (float *)malloc(img2DSizeMax2 * sizeof(float));
		float *h_img2Dreg = (float *)malloc(img2DSizeMax1 * sizeof(float));
		float *regRecords2D = (float *)malloc(11 * sizeof(float));
		float *d_img2DMax = NULL;
		cudaMalloc((void **)&d_img2DMax, img2DSizeMax*sizeof(float));
		float shiftX = (imx2 - imx1) / 2, shiftY = (imy2 - imy1) / 2, shiftZ = (imz2 - imz1) / 2;

		int flag2Dreg = 1;
		switch (flag2Dreg){
		case 1:
			tmx1[0] = 1; tmx1[1] = 0; tmx1[2] = shiftX;
			tmx1[3] = 0; tmx1[4] = 1; tmx1[5] = shiftY;
			cudaMemcpy(d_img3D, h_StackA, totalSize1 * sizeof(float), cudaMemcpyHostToDevice);
			maxprojection(d_img2DMax, d_img3D, imx1, imy1, imz1, 1);
			cudaMemcpy(h_img2D1, d_img2DMax, imx1 * imy1 * sizeof(float), cudaMemcpyDeviceToHost);
			cudaMemcpy(d_img3D, h_StackB, totalSize2 * sizeof(float), cudaMemcpyHostToDevice);
			maxprojection(d_img2DMax, d_img3D, imx2, imy2, imz2, 1);
			cudaMemcpy(h_img2D2, d_img2DMax, imx2 * imy2 * sizeof(float), cudaMemcpyDeviceToHost);
			(void)reg_2dshiftaligngpu(h_img2Dreg, tmx1, h_img2D1, h_img2D2, imx1, imy1, imx2, imy2,
				0, 0.3, 15, deviceNum, regRecords2D);
			shiftX = tmx1[2];
			shiftY = tmx1[5];

			tmx2[0] = 1; tmx2[1] = 0; tmx2[2] = shiftZ;
			tmx2[3] = 0; tmx2[4] = 1; tmx2[5] = shiftX;
			cudaMemcpy(d_img3D, h_StackA, totalSize1 * sizeof(float), cudaMemcpyHostToDevice);
			maxprojection(d_img2DMax, d_img3D, imx1, imy1, imz1, 2);
			cudaMemcpy(h_img2D1, d_img2DMax, imz1 * imx1 * sizeof(float), cudaMemcpyDeviceToHost);
			cudaMemcpy(d_img3D, h_StackB, totalSize2 * sizeof(float), cudaMemcpyHostToDevice);
			maxprojection(d_img2DMax, d_img3D, imx2, imy2, imz2, 2);
			cudaMemcpy(h_img2D2, d_img2DMax, imx2 * imz2 * sizeof(float), cudaMemcpyDeviceToHost);
			(void)reg_2dshiftalignXgpu(h_img2Dreg, tmx2, h_img2D1, h_img2D2, imz1, imx1, imz2, imx2,
				1, 0.3, 15, deviceNum, regRecords2D);
			shiftZ = tmx2[2];
			break;
		default:
			break;
		}
		for (int j = 0; j < NDIM; j++) iTmx[j] = 0;
		iTmx[0] = 1;
		iTmx[5] = 1;
		iTmx[10] = 1;
		iTmx[3] = shiftX;
		iTmx[7] = shiftY;
		iTmx[11] = shiftZ;
		free(tmx1); free(tmx2); free(h_img2D1); free(h_img2D2); free(h_img2Dreg); free(regRecords2D);
	}
	cudaFree(d_img3D);
	cudaMemGetInfo(&freeMem, &totalMem);
	printf("...GPU free memory before registration is %.0f MBites\n", (float)freeMem / 1048576.0f);
	// ***** Do 3D registration
	printf("Running 3D registration ...\n");

	if (flagInitialTmx == 3)
		inputTmx = 1;
	else
		inputTmx = flagInitialTmx;
	int subBgTrigger = 1;
	float *regRecords = (float *)malloc(11 * sizeof(float));
	int regStatus = reg_3dgpu(h_reg, iTmx, h_StackA, h_StackB, imSize, imSize2, regMethod,
		inputTmx, FTOL, itLimit, subBgTrigger, deviceNum, regRecords);

	memcpy(fusionRecords, regRecords, 11 * sizeof(float));
	free(h_StackB);
	free(regRecords);
	cudaMemGetInfo(&freeMem, &totalMem);
	printf("...GPU free memory before deconvolution is %.0f MBites\n", (float)freeMem / 1048576.0f);
	
	// ***** Joint deconvolution
	float *deconRecords = (float *)malloc(10 * sizeof(float));
	int deconStatus =  decon_dualview(h_decon, h_StackA, h_reg, imSize, h_psf1, h_psf2,
		psfSizeIn, itNumForDecon, deviceNum, gpuMemMode, deconRecords, flagUnmatch, h_psf_bp1, h_psf_bp2);
	memcpy(&fusionRecords[11], deconRecords, 10 * sizeof(float));
	free(deconRecords);
	free(h_StackA);
	free(h_aff12);
	cudaMemGetInfo(&freeMem, &totalMem);
	printf("...GPU free memory after whole processing is %.0f MBites\n", (float)freeMem / 1048576.0f);
	endWhole = clock();
	deconRecords[21] = (float)(endWhole - startWhole) / CLOCKS_PER_SEC;
	return 0;	
}
/// maximum intensity projectoions
int mp2Dgpu(float *h_MP, unsigned int *sizeMP, float *h_img, unsigned int *sizeImg, bool flagZProj, bool flagXProj, bool flagYProj){
	// sizeMP: sx, sy, sy, sz, sz, sx
	int sx = sizeImg[0], sy = sizeImg[1], sz = sizeImg[2];
	int totalSizeImg = sx*sy*sz; 
	int totalSizeMP = sx*sy + sy*sz + sz*sx;
	float *d_img, *d_MP;
	cudaMalloc((void **)&d_img, totalSizeImg * sizeof(float));
	cudaMalloc((void **)&d_MP, totalSizeMP * sizeof(float));
	cudaMemset(d_MP, 0, totalSizeMP*sizeof(float));
	cudaMemcpy(d_img, h_img, totalSizeImg* sizeof(float), cudaMemcpyHostToDevice);

	if(flagZProj) maxprojection(d_MP, d_img, sx, sy, sz, 1);
	if(flagXProj) maxprojection(&d_MP[sx*sy], d_img, sx, sy, sz, 3);
	if (flagZProj) maxprojection(&d_MP[sx*sy+sy*sz], d_img, sx, sy, sz, 2);
	sizeMP[0] = sx; sizeMP[1] = sy; sizeMP[2] = sy; 
	sizeMP[3] = sz; sizeMP[4] = sz; sizeMP[5] = sx;
	cudaMemcpy(h_MP, d_MP, totalSizeMP * sizeof(float), cudaMemcpyDeviceToHost);
	cudaFree(d_img);
	cudaFree(d_MP);
	return 0;
}

int mp3Dgpu(float *h_MP, unsigned int *sizeMP, float *h_img, unsigned int *sizeImg, bool flagXaxis, bool flagYaxis, int projectNum){
	//sizeMP: sx, imRotationy, projectNum, imRotationx, sy, projectNum
	int sx = sizeImg[0], sy = sizeImg[1], sz = sizeImg[2];
	int imRotationx = round(sqrt(sx*sx + sz*sz));
	int imRotationy = round(sqrt(sy*sy + sz*sz));
	float projectAng = 0;
	float projectStep = 3.14159 * 2 / projectNum;
	float *h_affRot = (float *)malloc(NDIM * sizeof(float));
	float *d_StackProject, *d_StackRotation;
	int totalSizeProjectX = sx * imRotationy;
	int totalSizeProjectY = imRotationx * sy;
	int totalSizeProjectMax = totalSizeProjectX > totalSizeProjectY ? totalSizeProjectX : totalSizeProjectY;
	int totalSizeRotationX = sx * imRotationy * imRotationy; 
	int totalSizeRotationY = imRotationx * sy * imRotationx;
	int totalSizeRotationMax;
	if (flagXaxis&&flagYaxis) 
		totalSizeRotationMax = totalSizeRotationX > totalSizeRotationY ? totalSizeRotationX : totalSizeRotationY;
	else if (flagXaxis) 
		totalSizeRotationMax = totalSizeRotationX;
	else if (flagYaxis)
		totalSizeRotationMax = totalSizeRotationY;
	else
		return -1;
	cudaMalloc((void **)&d_StackRotation, totalSizeRotationMax * sizeof(float));
	cudaMalloc((void **)&d_StackProject, totalSizeProjectMax * sizeof(float));
	cudaChannelFormatDesc channelDescT = cudaCreateChannelDesc<float>();
	cudaArray *d_Array;
	cudaMalloc3DArray(&d_Array, &channelDescT, make_cudaExtent(sx, sy, sz));
	cudacopyhosttoarray(d_Array, channelDescT, h_img, sx, sy, sz);
	BindTexture(d_Array, channelDescT);
	cudaCheckErrors("Texture create fail");
	if (flagXaxis){// 3D projection by X axis
		for (int iProj = 0; iProj < projectNum; iProj++){
			projectAng = projectStep * iProj;
			rot2matrix(h_affRot, projectAng, sx, sy, sz, 1);
			//rot3Dbyyaxis(h_aff_temp, projectAng, imx, imz, imRotationx, imRotationx);
			CopyTranMatrix(h_affRot, NDIM * sizeof(float));
			affineTransform(d_StackRotation, sx, imRotationy, imRotationy, sx, sy, sz);
			maxprojection(d_StackProject, d_StackRotation, sx, imRotationy, imRotationy, 1);
			cudaMemcpy(&h_MP[totalSizeProjectX*iProj], d_StackProject, totalSizeProjectX * sizeof(float), cudaMemcpyDeviceToHost);
			cudaDeviceSynchronize();
		}
		sizeMP[0] = sx; sizeMP[1] = imRotationy; sizeMP[2] = projectNum;
	}

	if (flagYaxis){// 3D projection by Y axis
		int Ystart = sx * imRotationy * projectNum;
		// 3D projection by Y axis
		for (int iProj = 0; iProj < projectNum; iProj++){
			projectAng = projectStep * iProj;
			rot2matrix(h_affRot, projectAng, sx, sy, sz, 2);
			//rot3Dbyyaxis(h_aff_temp, projectAng, imx, imz, imRotationx, imRotationx);
			CopyTranMatrix(h_affRot, NDIM * sizeof(float));
			affineTransform(d_StackRotation, imRotationx, sy, imRotationx, sx, sy, sz);
			maxprojection(d_StackProject, d_StackRotation, imRotationx, sy, imRotationx, 1);
			cudaMemcpy(&h_MP[Ystart + totalSizeProjectY*iProj], d_StackProject, totalSizeProjectY * sizeof(float), cudaMemcpyDeviceToHost);
			cudaDeviceSynchronize();
		}
		sizeMP[3] = imRotationx; sizeMP[4] = sy; sizeMP[5] = projectNum;
	}
	UnbindTexture();
	free(h_affRot);
	cudaFree(d_StackRotation);
	cudaFree(d_StackProject);
	cudaFreeArray(d_Array);

	return 0;
}

//// 3D reg, decon and fusion: batch processing
int reg_3dgpu_batch(char *outMainFolder, char *folder1, char *folder2, char *fileNamePrefix1, char *fileNamePrefix2, int imgNumStart, int imgNumEnd, int imgNumInterval, int imgNumTest,
	float *pixelSize1, float *pixelSize2, int regMode, int imRotation, int flagInitialTmx, float *iTmx, float FTOL, int itLimit, int deviceNum, int *flagSaveInterFiles, float *records){ 
	// Next version: variable arguments input
	// regMode--> 0: no registration; 1: one image only
	//			2: dependently, based on the results of last time point; 3: independently
	// flagInitialTmx --> 0: default matrix; 1: input matrix; 2: phase translation; 3: 2D registration
	// flagSaveInterFiles: 3 elements --> 1: save files; 0: not; [0] is currently not used.
	//					[0]: Intermediate outputs; [1]: reg A; [2]: reg B;

	char *outFolder, *inFolder1, *inFolder2;
	// ***** check if multitple color processing
	char mainFolder[MAX_PATH];
	bool flagMultiColor = false;
	int multiColor = atoi(folder1);
	int subFolderCount = 1;
	char subFolderNames[20][MAX_PATH];
	if (multiColor == 1){ // trigger multiple color
#ifdef _WIN32 
		strcpy(mainFolder, folder2);
		flagMultiColor = true;
#else
		fprintf(stderr, "*** Multi-color processing is currently not supported on Linux\n");
#endif
	}
	if (flagMultiColor){
#ifdef _WIN32 
		subFolderCount = findSubFolders(&subFolderNames[0][0], mainFolder);

		if (subFolderCount > 20)
			fprintf(stderr, "*** Number of subfolders: %d; two many subfolders\n", subFolderCount);
		else{
			printf("Procecing multicolor data: %d colors\n", subFolderCount);
			for (int j = 0; j < subFolderCount; j++)
				printf("...Subfolders %d: %s\n", j + 1, &subFolderNames[j][0]);
		}
		inFolder1 = concat(3, mainFolder, &subFolderNames[0][0], "/SPIMA/");
		inFolder2 = concat(3, mainFolder, &subFolderNames[0][0], "/SPIMB/");
#endif
	}
	else{
		inFolder1 = folder1;
		inFolder2 = folder2;
	}

	//************get basic input images and PSFs information ******************
	unsigned int  imSizeIn1[3], imSizeIn2[3], imSizeTemp[3];
	int imgNum = imgNumStart;
	if (regMode == 3)
		imgNum = imgNumTest;
	char imgNumStr[20];
	sprintf(imgNumStr, "%d", imgNum);
	char *fileStack1 = concat(4, inFolder1, fileNamePrefix1, imgNumStr, ".tif"); // TIFF file to get image information
	char *fileStack2 = concat(4, inFolder2, fileNamePrefix2, imgNumStr, ".tif");
	//**** check image files and image size ***
	unsigned short bitPerSample_input;
	if (!fexists(fileStack1)){
		fprintf(stderr, "***File does not exist: %s\n", fileStack1);
		fprintf(stderr, "*** FAILED - ABORTING\n");
		exit(1);
	}
	if (!fexists(fileStack2)){
		fprintf(stderr, "***File does not exist: %s\n", fileStack2);
		fprintf(stderr, "*** FAILED - ABORTING\n");
		exit(1);
	}
	

	bitPerSample_input = gettifinfo(fileStack1, &imSizeIn1[0]);
	if (bitPerSample_input != 16 && bitPerSample_input != 32){
		fprintf(stderr, "***Input images are not supported, please use 16-bit or 32-bit image !!!\n");
		fprintf(stderr, "*** FAILED - ABORTING\n");
		exit(1);
	}
	(void)gettifinfo(fileStack2, &imSizeIn2[0]);
	// ****************** Create output folders***************** //
	char *tmxFolder, *regFolder1, *regFolder2;
	// flagSaveInterFiles: 3 elements --> 1: save files; 0: not
	//					[0]: Intermediate outputs; [1]: reg A; [2]: reg B;
	if (flagMultiColor){
#ifdef _WIN32 
		CreateDirectory(outMainFolder, NULL);
		for (int j = 0; j < subFolderCount; j++){
			outFolder = concat(3, outMainFolder, &subFolderNames[j][0], "/");
			inFolder1 = concat(3, mainFolder, &subFolderNames[j][0], "/SPIMA/");
			inFolder2 = concat(3, mainFolder, &subFolderNames[j][0], "/SPIMB/");
			tmxFolder = concat(2, outFolder, "TMX/");
			regFolder1 = concat(2, outFolder, "RegA/");
			regFolder2 = concat(2, outFolder, "RegB/");
			CreateDirectory(outFolder, NULL);
			CreateDirectory(tmxFolder, NULL); // currentlly the TMX file is always saved
			if (flagSaveInterFiles[1] == 1) CreateDirectory(regFolder1, NULL);
			if (flagSaveInterFiles[2] == 1) CreateDirectory(regFolder2, NULL);
			free(outFolder); free(inFolder1); free(inFolder2); free(tmxFolder); free(regFolder1); //
			free(regFolder2);
		}
#endif
	}
	else{
		outFolder = outMainFolder;
		inFolder1 = folder1;
		inFolder2 = folder2;
		tmxFolder = concat(2, outFolder, "TMX/");
		regFolder1 = concat(2, outFolder, "RegA/");
		regFolder2 = concat(2, outFolder, "RegB/");

#ifdef _WIN32 
		CreateDirectory(outFolder, NULL);
		CreateDirectory(tmxFolder, NULL); // currentlly the TMX file is always saved
		if (flagSaveInterFiles[1] == 1) CreateDirectory(regFolder1, NULL);
		if (flagSaveInterFiles[2] == 1) CreateDirectory(regFolder2, NULL);
#else
		mkdir(outFolder, 0755);
		mkdir(tmxFolder, 0755);
		if (flagSaveInterFiles[1] == 1) mkdir(regFolder1, 0755);
		if (flagSaveInterFiles[2] == 1) mkdir(regFolder2, 0755);
#endif
	}

	// ****************** calculate images' size ************************* //
	int imx, imy, imz;
	unsigned int imSize[3], imSize1[3], imSize2[3];
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

	// total pixel count for each images
	int totalSizeIn1 = imSizeIn1[0] * imSizeIn1[1] * imSizeIn1[2]; // in floating format
	int totalSizeIn2 = imSizeIn2[0] * imSizeIn2[1] * imSizeIn2[2]; // in floating format
	int totalSize1 = imx1*imy1*imz1; // in floating format
	int totalSize2 = imx2*imy2*imz2; // in floating format
	int totalSize12 = (totalSize1 > totalSize2) ? totalSize1 : totalSize2;
	int totalSize = totalSize1; // in floating format

	// print GPU devices information
	cudaSetDevice(deviceNum);

	// Log file
	FILE *f1 = NULL, *f2 = NULL, *f3 = NULL;
	char *fileLog = concat(2, outMainFolder, "ProcessingLog.txt");

	// print images information
	printf("Image information:\n");
	printf("...Image A size %d x %d x %d\n     ...pixel size %2.4f x %2.4f x %2.4f um\n", imSizeIn1[0], imSizeIn1[1], imSizeIn1[2], pixelSize1[0], pixelSize1[1], pixelSize1[2]);
	printf("...Image B size %d x %d x %d\n     ...pixel size %2.4f x %2.4f x %2.4f um\n", imSizeTemp[0], imSizeTemp[1], imSizeTemp[2], pixelSizeTemp[0], pixelSizeTemp[1], pixelSizeTemp[2]);
	printf("...Output Image size %d x %d x %d \n    ......pixel size %2.4f x %2.4f x %2.4f um\n\n", imSize[0], imSize[1], imSize[2], pixelSize[0], pixelSize[1], pixelSize[2]);
	printf("...Image number from %d to %d with step %d\n", imgNumStart, imgNumEnd, imgNumInterval);
	switch (regMode){
	case 0:
		printf("...No registration\n"); break;
	case 1:
		printf("...One registration for all images, test image number: %d\n", imgNumTest); break;
	case 2:
		printf("...Perform registration for all images dependently\n"); break;
	case 3:
		printf("...Perform registration for all images independently\n"); break;
	default:
		printf("...regMode incorrect !!!\n");
		return 0;
	}

	switch (imRotation){
	case 0:
		printf("...No rotation on image B\n"); break;
	case 1:
		printf("...Rotate image B by 90 degree along Y axis\n"); break;
	case -1:
		printf("...Rotate image B by -90 degree along Y axis\n"); break;
	}

	switch (flagInitialTmx){
	case 1:
		printf("...Initial transformation matrix: based on input matrix\n"); break;
	case 2:
		printf("...Initial transformation matrix: by phase translation\n"); break;
	case 3:
		printf("...Initial transformation matrix: by 2D registration\n"); break;
	default:
		printf("...Initial transformation matrix: Default\n");
	}
	printf("\n...GPU Device %d is used...\n\n", deviceNum);

	time_t now;
	time(&now);
	// ****Write information to log file***
	f1 = fopen(fileLog, "w");
	// print images information
	fprintf(f1, "3D Registration: %s\n", ctime(&now));
	if (flagMultiColor){
		fprintf(f1, "Multicolor data: %d colors\n", subFolderCount);
		fprintf(f1, "...Input directory: %s\n", folder2);
		for (int j = 0; j < subFolderCount; j++)
			fprintf(f1, "     ...Subfolders %d: %s\n", j + 1, &subFolderNames[j][0]);
		fprintf(f1, "...Output directory: %s\n", outMainFolder);
	}
	else{
		fprintf(f1, "Single color data:\n");
		fprintf(f1, "...SPIMA input directory: %s\n", folder1);
		fprintf(f1, "...SPIMB input directory: %s\n", folder2);
		fprintf(f1, "...Output directory: %s\n", outMainFolder);
	}

	fprintf(f1, "\nImage information:\n");
	fprintf(f1, "...Image A size %d x %d x %d\n     ...pixel size %2.4f x %2.4f x %2.4f um\n", imSizeIn1[0], imSizeIn1[1], imSizeIn1[2], pixelSize1[0], pixelSize1[1], pixelSize1[2]);
	fprintf(f1, "...Image B size %d x %d x %d\n     ...pixel size %2.4f x %2.4f x %2.4f um\n", imSizeTemp[0], imSizeTemp[1], imSizeTemp[2], pixelSizeTemp[0], pixelSizeTemp[1], pixelSizeTemp[2]);
	fprintf(f1, "...Output Image size %d x %d x %d \n    ......pixel size %2.4f x %2.4f x %2.4f um\n\n", imSize[0], imSize[1], imSize[2], pixelSize[0], pixelSize[1], pixelSize[2]);
	fprintf(f1, "...Image number from %d to %d with step %d\n", imgNumStart, imgNumEnd, imgNumInterval);
	switch (regMode){
	case 0:
		fprintf(f1, "...No registration\n"); break;
	case 1:
		fprintf(f1, "...One registration for all images, test image number: %d\n", imgNumTest); break;
	case 2:
		fprintf(f1, "...Perform registration for all images dependently\n"); break;
	case 3:
		fprintf(f1, "...Perform registration for all images independently\n"); break;
	default:
		fprintf(f1, "...regMode incorrect !!!\n");
		return 0;
	}

	switch (imRotation){
	case 0:
		fprintf(f1, "...No rotation on image B\n"); break;
	case 1:
		fprintf(f1, "...Rotate image B by 90 degree along Y axis\n"); break;
	case -1:
		fprintf(f1, "...Rotate image B by -90 degree along Y axis\n"); break;
	}

	switch (flagInitialTmx){
	case 1:
		fprintf(f1, "...Initial transformation matrix: based on input matrix\n"); break;
	case 2:
		fprintf(f1, "...Initial transformation matrix: by phase translation\n"); break;
	case 3:
		fprintf(f1, "...Initial transformation matrix: by 2D registration\n"); break;
	default:
		fprintf(f1, "...Initial transformation matrix: Default\n");
	}

	fprintf(f1, "...Registration convergence threshold:%f\n", FTOL);
	fprintf(f1, "...Registration maximum sub-iteration number:%d\n", itLimit);
	fprintf(f1, "\n...GPU Device %d is used...\n\n", deviceNum);
	fclose(f1);
	// ****************** Processing Starts***************** //
	// variables for memory and time cost records
	clock_t startWhole, endWhole, start, end;
	size_t totalMem = 0;
	size_t freeMem = 0;
	printf("\nStart processing...\n");
	cudaMemGetInfo(&freeMem, &totalMem);
	printf("...GPU free memory at beginning is %.0f MBites\n", (float)freeMem / 1048576.0f);
	f1 = fopen(fileLog, "a");
	fprintf(f1, "\nStart processing...\n");
	fprintf(f1, "...GPU free memory at beginning is %.0f MBites\n", (float)freeMem / 1048576.0f);
	fclose(f1);
	startWhole = clock();
	// ***** Set GPU memory use mode based on images size and available GPU memory ****
	int gpuMemMode = 0;
	// gpuMemMode--> -1: not enough gpu memory; 0-3: enough GPU memory; 
	if (freeMem < 4 * totalSize * sizeof(float)){ // 4 variables + 2 FFT calculation space + a few additional space
		gpuMemMode = -1;
		f1 = fopen(fileLog, "a");
		fprintf(f1, "***Available GPU memory is insufficient, processing terminated !!!\n");
		fclose(f1);
		fprintf(stderr, "***Available GPU memory is insufficient, processing terminated !!!\n");
		fprintf(stderr, "*** FAILED - ABORTING\n");
		exit(1);
	}

	// variables
	float
		*h_img1,
		*h_img2,
		*h_StackA,
		*h_StackB,
		*h_reg;
	float
		*d_imgE;
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
	cudaArray *d_Array1, *d_Array2;
	float *h_aff12 = (float *)malloc((NDIM)* sizeof(float));
	h_img1 = (float *)malloc(totalSizeIn1 * sizeof(float));
	h_img2 = (float *)malloc(totalSizeIn2 * sizeof(float));
	h_StackA = (float *)malloc(totalSize12 * sizeof(float));
	h_StackB = (float *)malloc(totalSize12 * sizeof(float));
	h_reg = (float *)malloc(totalSize * sizeof(float));

	// regMode--> 0: no registration; 1: one image only
	//			3: dependently, based on the results of last time point; 4: independently
	float *h_affInitial = (float *)malloc((NDIM)* sizeof(float));
	float *h_affWeighted = (float *)malloc((NDIM)* sizeof(float));
	bool regMode3OffTrigger = false; // registration trigger for regMode 3
	float shiftX, shiftY, shiftZ;
	bool inputTmx;
	int regMethod = 7;
	int subBgTrigger = 1;
	float *regRecords = (float *)malloc(11 * sizeof(float));
	float *deconRecords = (float *)malloc(10 * sizeof(float));
	int regStatus;
	bool mStatus;

	// ******processing in batch*************
	for (imgNum = imgNumStart; imgNum <= imgNumEnd; imgNum += imgNumInterval){
		if (regMode == 0){ // no registration
			regMethod = 0;
		}
		else if (regMode == 1){//in regMode 1, use Test number for registratiion
			imgNum = imgNumTest;
		}
		printf("\n***Image time point number: %d \n", imgNum);
		f1 = fopen(fileLog, "a");
		fprintf(f1, "\n***Image time point number: %d \n", imgNum);
		fclose(f1);
		sprintf(imgNumStr, "%d", imgNum);
		char *fileStackA, *fileStackB, *fileRegA, *fileRegB, *fileTmx;
		for (int iColor = 0; iColor < subFolderCount; iColor++){
			start = clock();
			if (flagMultiColor){
				printf("\n Processing time point %d color %d: %s \n", imgNum, iColor + 1, &subFolderNames[iColor][0]);
				f1 = fopen(fileLog, "a");
				fprintf(f1, "\n Processing time point %d color %d: %s \n", imgNum, iColor + 1, &subFolderNames[iColor][0]);
				fclose(f1);
				outFolder = concat(3, outMainFolder, &subFolderNames[iColor][0], "/");
				inFolder1 = concat(3, mainFolder, &subFolderNames[iColor][0], "/SPIMA/");
				inFolder2 = concat(3, mainFolder, &subFolderNames[iColor][0], "/SPIMB/");
				tmxFolder = concat(2, outFolder, "TMX/");
				regFolder1 = concat(2, outFolder, "RegA/");
				regFolder2 = concat(2, outFolder, "RegB/");
			}
			fileStackA = concat(4, inFolder1, fileNamePrefix1, imgNumStr, ".tif");
			fileStackB = concat(4, inFolder2, fileNamePrefix2, imgNumStr, ".tif");
			fileRegA = concat(5, regFolder1, fileNamePrefix1, "reg_", imgNumStr, ".tif");
			fileRegB = concat(5, regFolder2, fileNamePrefix2, "reg_", imgNumStr, ".tif");
			fileTmx = concat(4, tmxFolder, "Matrix_", imgNumStr, ".tmx");
			///

			printf("...Registration...\n");
			printf("	...Initializing (rotation, interpolation, initial matrix)...\n");
			f1 = fopen(fileLog, "a");
			fprintf(f1, "...Registration...\n");
			fprintf(f1, "	...Initializing (rotation, interpolation, initial matrix)...\n");
			fclose(f1);
			// ****************Interpolation before registration**************** //////
			// ## check files
			if (!fexists(fileStackA)){
				printf("***File does not exist: %s", fileStackA);
				return 0;
			}
			if (!fexists(fileStackB)){
				printf("***File does not exist: %s", fileStackB);
				return 0;
			}
			cudaMalloc((void **)&d_img3D, totalSize12 *sizeof(float));
			if ((imRotation == 1) || (imRotation == -1))
				cudaMalloc((void **)&d_imgE, totalSizeIn1 * sizeof(float));
			if (flagInterp1)
				cudaMalloc3DArray(&d_Array1, &channelDesc, make_cudaExtent(imSizeIn1[0], imSizeIn1[1], imSizeIn1[2]));
			if (flagInterp2)
				cudaMalloc3DArray(&d_Array2, &channelDesc, make_cudaExtent(imSizeIn2[0], imSizeIn2[1], imSizeIn2[2]));

			///##image 1 or Stack A
			readtifstack(h_img1, fileStackA, &imSizeIn1[0]);
			if (flagInterp1){
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
			cudaDeviceSynchronize();

			//## image 2 or Stack B
			readtifstack(h_img2, fileStackB, &imSizeTemp[0]); //something wrong here at the 6th reading
			// rotation
			if ((imRotation == 1) || (imRotation == -1)){
				cudaMemcpy(d_imgE, h_img2, totalSizeIn2 * sizeof(float), cudaMemcpyHostToDevice);
				rotbyyaxis(d_img3D, d_imgE, imSizeIn2[0], imSizeIn2[1], imSizeIn2[2], imRotation);
				cudaMemcpy(h_StackB, d_img3D, totalSizeIn2 * sizeof(float), cudaMemcpyDeviceToHost);
				cudaFree(d_imgE);
			}
			if (flagInterp2){
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
			cudaDeviceSynchronize();

			// ***** Do 2D registration or alignment
			// only if 3D registration is set and flagInitialTmx = 2
			if ((regMode > 0) && (flagInitialTmx == 3)){
				float *tmx1 = (float *)malloc(6 * sizeof(float));
				float *tmx2 = (float *)malloc(6 * sizeof(float));
				int img2DSizeMax1 = ((imx1 * imy1) > (imz1 * imx1)) ? (imx1 * imy1) : (imz1 * imx1);
				int img2DSizeMax2 = ((imx2 * imy2) > (imz2 * imx2)) ? (imx2 * imy2) : (imz2 * imx2);
				int img2DSizeMax = (img2DSizeMax1 > img2DSizeMax2) ? img2DSizeMax1 : img2DSizeMax2;
				float *h_img2D1 = (float *)malloc(img2DSizeMax1 * sizeof(float));
				float *h_img2D2 = (float *)malloc(img2DSizeMax2 * sizeof(float));
				float *h_img2Dreg = (float *)malloc(img2DSizeMax1 * sizeof(float));
				float *regRecords2D = (float *)malloc(11 * sizeof(float));
				float *d_img2DMax = NULL;
				cudaMalloc((void **)&d_img2DMax, img2DSizeMax*sizeof(float));
				shiftX = (imx2 - imx1) / 2, shiftY = (imy2 - imy1) / 2, shiftZ = (imz2 - imz1) / 2;
				int flag2Dreg = 1;
				switch (flag2Dreg){
				case 1:
					tmx1[0] = 1; tmx1[1] = 0; tmx1[2] = shiftX;
					tmx1[3] = 0; tmx1[4] = 1; tmx1[5] = shiftY;
					cudaMemcpy(d_img3D, h_StackA, totalSize1 * sizeof(float), cudaMemcpyHostToDevice);
					maxprojection(d_img2DMax, d_img3D, imx1, imy1, imz1, 1);
					cudaMemcpy(h_img2D1, d_img2DMax, imx1 * imy1 * sizeof(float), cudaMemcpyDeviceToHost);
					cudaMemcpy(d_img3D, h_StackB, totalSize2 * sizeof(float), cudaMemcpyHostToDevice);
					maxprojection(d_img2DMax, d_img3D, imx2, imy2, imz2, 1);
					cudaMemcpy(h_img2D2, d_img2DMax, imx2 * imy2 * sizeof(float), cudaMemcpyDeviceToHost);
					(void)reg_2dshiftaligngpu(h_img2Dreg, tmx1, h_img2D1, h_img2D2, imx1, imy1, imx2, imy2,
						0, 0.3, 15, deviceNum, regRecords2D);
					shiftX = tmx1[2];
					shiftY = tmx1[5];

					tmx2[0] = 1; tmx2[1] = 0; tmx2[2] = shiftZ;
					tmx2[3] = 0; tmx2[4] = 1; tmx2[5] = shiftX;
					cudaMemcpy(d_img3D, h_StackA, totalSize1 * sizeof(float), cudaMemcpyHostToDevice);
					maxprojection(d_img2DMax, d_img3D, imx1, imy1, imz1, 2);
					cudaMemcpy(h_img2D1, d_img2DMax, imz1 * imx1 * sizeof(float), cudaMemcpyDeviceToHost);
					cudaMemcpy(d_img3D, h_StackB, totalSize2 * sizeof(float), cudaMemcpyHostToDevice);
					maxprojection(d_img2DMax, d_img3D, imx2, imy2, imz2, 2);
					cudaMemcpy(h_img2D2, d_img2DMax, imx2 * imz2 * sizeof(float), cudaMemcpyDeviceToHost);
					(void)reg_2dshiftalignXgpu(h_img2Dreg, tmx2, h_img2D1, h_img2D2, imz1, imx1, imz2, imx2,
						1, 0.3, 15, deviceNum, regRecords2D);
					shiftZ = tmx2[2];
					break;
				default:
					break;
				}
				for (int j = 0; j < NDIM; j++) iTmx[j] = 0;
				iTmx[0] = 1;
				iTmx[5] = 1;
				iTmx[10] = 1;
				iTmx[3] = shiftX;
				iTmx[7] = shiftY;
				iTmx[11] = shiftZ;
				if ((regMode == 1) || (regMode == 2)) flagInitialTmx = 1; // perform 2D reg only one time
				free(tmx1); free(tmx2); free(h_img2D1); free(h_img2D2); free(h_img2Dreg); free(regRecords2D);
			}
			cudaFree(d_img3D);

			/// initialize matrix *******
			//  Do 3D registration ******
			// regMode--> 0: no registration; 1: one image only
			//			2: dependently, based on the results of last time point; 3: independently
			cudaMemGetInfo(&freeMem, &totalMem);
			printf("	...GPU free memory before registration is %.0f MBites\n", (float)freeMem / 1048576.0f);
			f1 = fopen(fileLog, "a");
			fprintf(f1, "	...GPU free memory before registration is %.0f MBites\n", (float)freeMem / 1048576.0f);
			fclose(f1);
			if (flagInitialTmx == 3)
				inputTmx = 1;
			else
				inputTmx = flagInitialTmx;
			if (inputTmx==1) memcpy(h_affInitial, iTmx, NDIM*sizeof(float));
			switch (regMode){
			case 0:
				regMethod = 0;
				regStatus = reg_3dgpu(h_reg, iTmx, h_StackA, h_StackB, imSize, imSize2, regMethod,
					inputTmx, FTOL, itLimit, subBgTrigger, deviceNum, regRecords);
				break;
			case 1:
				regStatus = reg_3dgpu(h_reg, iTmx, h_StackA, h_StackB, imSize, imSize2, regMethod,
					inputTmx, FTOL, itLimit, subBgTrigger, deviceNum, regRecords);
				imgNum = imgNumStart;
				regMode = 0; // Don't do more registraion for other time points
				flagInitialTmx = 1; // Apply matrix to all other time points
				continue;
				break;
			case 2:
				if ((imgNum != imgNumStart) || (iColor>0)){
					inputTmx = true; // use previous matrix as input
					memcpy(iTmx, h_affWeighted, NDIM*sizeof(float));
				}
				regStatus = reg_3dgpu(h_reg, iTmx, h_StackA, h_StackB, imSize, imSize2, regMethod,
					inputTmx, FTOL, itLimit, subBgTrigger, deviceNum, regRecords);
				mStatus = checkmatrix(iTmx);//if registration is good
				if (!mStatus){ // apply previous matrix
					memcpy(iTmx, h_affInitial, NDIM*sizeof(float)); // use input or previous matrix
					regStatus = reg_3dgpu(h_reg, iTmx, h_StackA, h_StackB, imSize, imSize2, 0,
						inputTmx, FTOL, itLimit, subBgTrigger, deviceNum, regRecords);
				}
				if ((imgNum == imgNumStart) && (iColor == 0)){
					memcpy(h_affWeighted, iTmx, NDIM*sizeof(float));
				}
				else{
					for (int j = 0; j < NDIM; j++){
						h_affWeighted[j] = 0.8*h_affWeighted[j] + 0.2*iTmx[j]; // weighted matrix for next time point
					}
				}
				break;
			case 3:
				if (inputTmx) memcpy(iTmx, h_affInitial, NDIM*sizeof(float));
				regStatus = reg_3dgpu(h_reg, iTmx, h_StackA, h_StackB, imSize, imSize2, 7,
					inputTmx, FTOL, itLimit, subBgTrigger, deviceNum, regRecords);
				mStatus = checkmatrix(iTmx);//if registration is good
				if (!mStatus){ // apply previous matrix
					memcpy(iTmx, h_affInitial, NDIM*sizeof(float)); // use input or previous matrix
					regStatus = reg_3dgpu(h_reg, iTmx, h_StackA, h_StackB, imSize, imSize2, 0,
						inputTmx, FTOL, itLimit, subBgTrigger, deviceNum, regRecords);
				}
				break;
			default:
				;
			}
			//regRecords
			//[0] -[3]: initial GPU memory, after variables allocated, after processing, after variables released ( all in MB)
			//[4] -[6]: initial cost function value, minimized cost function value, intermediate cost function value
			//[7] -[10]: registration time (in s), whole time (in s), single sub iteration time (in ms), total sub iterations
			printf("	...initial cost function value: %f\n", regRecords[4]);
			printf("	...minimized cost function value: %f\n", regRecords[5]);
			printf("	...total sub-iteration number: %d\n", int(regRecords[10]));
			printf("	...each sub-iteration time cost: %2.3f ms\n", regRecords[9]);
			printf("	...all iteration time cost: %2.3f s\n", regRecords[7]);
			printf("	...registration time cost: %2.3f s\n", regRecords[7]);
			f1 = fopen(fileLog, "a");
			fprintf(f1, "	...initial cost function value: %f\n", regRecords[4]);
			fprintf(f1, "	...minimized cost function value: %f\n", regRecords[5]);
			//fprintf(f1, "	...total sub-iteration number: %d\n", int(regRecords[10]));
			//fprintf(f1, "	...each sub-iteration time cost: %2.3f ms\n", regRecords[9]);
			//fprintf(f1, "	...all iteration time cost: %2.3f s\n", regRecords[7]);
			fprintf(f1, "	...registration time cost: %2.3f s\n", regRecords[7]);
			fclose(f1);

			// always save transformation matrix
			f2 = fopen(fileTmx, "w");
			for (int j = 0; j < NDIM; j++)
			{
				fprintf(f2, "%f\t", iTmx[j]);
				if ((j + 1) % 4 == 0)
					fprintf(f2, "\n");
			}
			fprintf(f2, "%f\t%f\t%f\t%f\n", 0.0, 0.0, 0.0, 1.0);
			fclose(f2);
			memcpy(records, regRecords, 11 * sizeof(float));
			if (flagSaveInterFiles[1] == 1) writetifstack(fileRegA, h_StackA, &imSize[0], bitPerSample_input);//set bitPerSample as input images
			if (flagSaveInterFiles[2] == 1) writetifstack(fileRegB, h_reg, &imSize[0], bitPerSample_input);//set bitPerSample as input images

			end = clock();
			records[12] = (float)(end - start) / CLOCKS_PER_SEC;

			// release file names
			if (flagMultiColor){
				free(outFolder); free(inFolder1); free(inFolder2); free(tmxFolder); free(regFolder1); //
				free(regFolder2); 
			}
			free(fileStackA); free(fileStackB); free(fileRegA); free(fileRegB); //
			free(fileTmx); 
			printf("...Time cost for current image is %2.3f s\n", records[12]);
			f1 = fopen(fileLog, "a");
			fprintf(f1, "...Time cost for current image is %2.3f s\n", records[12]);
			fclose(f1);
		}
	}

	////release CPU memory 
	free(h_affInitial);
	free(h_affWeighted);
	free(regRecords);
	free(deconRecords);
	free(h_aff12);
	free(h_StackA);
	free(h_StackB);
	free(h_reg);

	cudaMemGetInfo(&freeMem, &totalMem);
	printf("\nGPU free memory after whole processing is %.0f MBites\n", (float)freeMem / 1048576.0f);
	endWhole = clock();
	printf("Total time cost for whole processing is %2.3f s\n", (float)(endWhole - startWhole) / CLOCKS_PER_SEC);
	f1 = fopen(fileLog, "a");
	fprintf(f1, "\nGPU free memory after whole processing is %.0f MBites\n", (float)freeMem / 1048576.0f);
	fprintf(f1, "Total time cost for whole processing is %2.3f s\n", (float)(endWhole - startWhole) / CLOCKS_PER_SEC);
	fclose(f1);
	return 0;
}

int decon_singleview_batch(char *outMainFolder, char *folder, char *fileNamePrefix, int imgNumStart, int imgNumEnd, int imgNumInterval,  char *filePSF,
	int itNumForDecon, int deviceNum, int bitPerSample, bool flagMultiColor, float *records, bool flagUnmatch, char *filePSF_bp){
	//deconRecords: 10 elements
	char *outFolder, *inFolder;
	char mainFolder[MAX_PATH];
	int subFolderCount = 1;
	char subFolderNames[20][MAX_PATH];
	if (flagMultiColor){ // trigger multiple color
#ifdef _WIN32 
		strcpy(mainFolder, folder);
#else
		fprintf(stderr, "*** Multi-color processing is currently not supported on Linux\n");
#endif
	}
	if (flagMultiColor){
#ifdef _WIN32 
		subFolderCount = findSubFolders(&subFolderNames[0][0], mainFolder);

		if (subFolderCount > 20)
			fprintf(stderr, "*** Number of subfolders: %d; two many subfolders\n", subFolderCount);
		else{
			printf("Procecing multicolor data: %d colors\n", subFolderCount);
			for (int j = 0; j < subFolderCount; j++)
				printf("...Subfolders %d: %s\n", j + 1, &subFolderNames[j][0]);
		}
		inFolder = concat(3, mainFolder, &subFolderNames[0][0], "/");
#endif
	}
	else{
		inFolder = folder;
	}
	// ************get basic input images and PSFs information ******************
	unsigned int  imSize[3], imSizeTemp[3], psfSize[3];
	int imgNum = imgNumStart;
	char imgNumStr[20];
	sprintf(imgNumStr, "%d", imgNum);
	char *fileStack = concat(4, inFolder, fileNamePrefix, imgNumStr, ".tif"); // TIFF file to get image information
	// **** check image files and image size ***
	unsigned short bitPerSample_input;
	if (!fexists(fileStack)){
		fprintf(stderr, "***File does not exist: %s\n", fileStack);
		fprintf(stderr, "*** FAILED - ABORTING\n");
		exit(1);
	}
	if (!fexists(filePSF)){
		fprintf(stderr, "***File does not exist: %s\n", filePSF);
		fprintf(stderr, "*** FAILED - ABORTING\n");
		exit(1);
	}
	if (flagUnmatch){// use unmatched back projectors
		if (!fexists(filePSF_bp)){
			fprintf(stderr, "***File does not exist: %s\n", filePSF_bp);
			fprintf(stderr, "*** FAILED - ABORTING\n");
			exit(1);
		}
	}

	bitPerSample_input = gettifinfo(fileStack, &imSize[0]);
	if (bitPerSample_input != 16 && bitPerSample_input != 32){
		fprintf(stderr, "***Input images are not supported, please use 16-bit or 32-bit image !!!\n");
		fprintf(stderr, "*** FAILED - ABORTING\n");
		exit(1);
	}
	(void)gettifinfo(filePSF, &psfSize[0]);
	if (flagUnmatch){
		(void)gettifinfo(filePSF_bp, &imSizeTemp[0]);
		if ((psfSize[0] != imSizeTemp[0]) || (psfSize[1] != imSizeTemp[1]) || (psfSize[2] != imSizeTemp[2])){
			fprintf(stderr, "***PSF and PSF_bp image size are not consistent to each other !!!\n");
			fprintf(stderr, "*** FAILED - ABORTING\n");
			exit(1);
		}
	}

	// ****************** Create output folders***************** //
	char *deconFolder;
	// flagSaveProjZ --> 1: save max projections; 0: not.
	if (flagMultiColor){

#ifdef _WIN32 
		CreateDirectory(outMainFolder, NULL);
		for (int j = 0; j < subFolderCount; j++){
			outFolder = concat(3, outMainFolder, &subFolderNames[j][0], "/");
			inFolder = concat(3, mainFolder, &subFolderNames[j][0], "/");
			deconFolder = concat(2, outFolder, "Decon/");
			CreateDirectory(outFolder, NULL);
			CreateDirectory(deconFolder, NULL);
			free(outFolder); free(inFolder); free(deconFolder);
		}
#endif
	}
	else{
		outFolder = outMainFolder;
		inFolder = folder;
		deconFolder = concat(2, outFolder, "Decon/");
#ifdef _WIN32 
		CreateDirectory(outFolder, NULL);
		CreateDirectory(deconFolder, NULL);
#else
		mkdir(outFolder, 0755);
		mkdir(deconFolder, 0755);
#endif
	}

	// ****************** calculate images' size ************************* //
	int imx, imy, imz;
	imx = imSize[0]; imy = imSize[1]; imz = imSize[2]; // also as output size

	// PSF size
	int
		PSFx, PSFy, PSFz;
	PSFx = psfSize[0], PSFy = psfSize[1], PSFz = psfSize[2];

	//FFT size
	int
		FFTx, FFTy, FFTz,
		PSFox, PSFoy, PSFoz,
		imox, imoy, imoz;

	FFTx = snapTransformSize(imx);// snapTransformSize(imx + PSFx - 1);
	FFTy = snapTransformSize(imy);// snapTransformSize(imy + PSFy - 1);
	FFTz = snapTransformSize(imz);// snapTransformSize(imz + PSFz - 1);

	PSFox = round(PSFx / 2);
	PSFoy = round(PSFy / 2);
	PSFoz = round(PSFz / 2);
	imox = round((FFTx - imSize[0]) / 2);
	imoy = round((FFTy - imSize[1]) / 2);
	imoz = round((FFTz - imSize[2]) / 2);

	// total pixel count for each images
	int totalSize = imx*imy*imz; // in floating format
	int totalSizePSF = PSFx*PSFy*PSFz; // in floating format
	int totalSizeFFT = FFTx*FFTy*FFTz; // in floating format
	int totalSizeSpectrum = FFTx * FFTy*(FFTz / 2 + 1); // in complex floating format
	int totalSizeMax = totalSizeSpectrum * 2; // in floating format
	int totalSizeMax2 = totalSizeMax > totalSizePSF ? totalSizeMax : totalSizePSF; // in floating format: in case PSF has a larger size


	float
		*h_StackA,
		*h_StackE,
		*d_StackA,
		*d_StackE,
		*d_StackT;
	float
		*h_img,
		*h_decon,
		*h_psf,
		*h_psf_bp;

	fComplex
		*h_PSFSpectrum,
		*h_FlippedPSFSpectrum,
		*h_StackESpectrum,
		*d_PSFSpectrum,
		*d_FlippedPSFSpectrum,
		*d_StackESpectrum;
	cufftHandle
		fftPlanFwd,
		fftPlanInv;

	// print GPU devices information
	cudaSetDevice(deviceNum);
	int gpuMemMode = 0;

	// Log file
	FILE *f1 = NULL;
	char *fileLog = concat(2, outMainFolder, "ProcessingLog.txt");

	// print images information
	printf("Image information:\n");
	printf("...Image size %d x %d x %d\n ", imSize[0], imSize[1], imSize[2]);
	printf("...PSF size %d x %d x %d\n", psfSize[0], psfSize[1], psfSize[2]);
	printf("...FFT size: %d x %d x %d\n", FFTx, FFTy, FFTz);
	printf("...Output Image size %d x %d x %d \n\n", imSize[0], imSize[1], imSize[2]);
	printf("...Image number from %d to %d with step %d\n", imgNumStart, imgNumEnd, imgNumInterval);

	if (flagUnmatch) printf("\n...Unmatched back projectors for deconvolution: yes\n");
	else printf("\n...Unmatched back projectors for deconvolution: no\n");
	printf("\n...Iteration number for deconvolution:%d\n", itNumForDecon);
	printf("\n...GPU Device %d is used...\n\n", deviceNum);

	time_t now;
	time(&now);
	//****Write information to log file***
	f1 = fopen(fileLog, "w");
	// print images information
	fprintf(f1, "Single-view Deconvolution: %s\n", ctime(&now));
	if (flagMultiColor){
		fprintf(f1, "Multicolor data: %d colors\n", subFolderCount);
		fprintf(f1, "...Input directory: %s\n", folder);
		for (int j = 0; j < subFolderCount; j++)
			fprintf(f1, "     ...Subfolders %d: %s\n", j + 1, &subFolderNames[j][0]);
		fprintf(f1, "...Output directory: %s\n", outMainFolder);
	}
	else{
		fprintf(f1, "Single color data:\n");
		fprintf(f1, "...Input directory: %s\n", folder);
		fprintf(f1, "...Output directory: %s\n", outMainFolder);
	}

	fprintf(f1, "\nImage information:\n");
	fprintf(f1, "...Image size %d x %d x %d\n", imSize[0], imSize[1], imSize[2]);
	fprintf(f1, "...PSF size %d x %d x %d\n", psfSize[0], psfSize[1], psfSize[2]);
	fprintf(f1, "...FFT size: %d x %d x %d\n", FFTx, FFTy, FFTz);
	fprintf(f1, "...Output Image size %d x %d x %d \n\n", imSize[0], imSize[1], imSize[2]);
	fprintf(f1, "...Image number from %d to %d with step %d\n", imgNumStart, imgNumEnd, imgNumInterval);
	if (flagUnmatch) fprintf(f1, "\n...Unmatched back projectors for deconvolution: yes\n");
	else fprintf(f1, "\n...Unmatched back projectors for deconvolution: no\n");
	fprintf(f1, "...Iteration number for deconvolution:%d\n", itNumForDecon);
	fprintf(f1, "\n...GPU Device %d is used...\n\n", deviceNum);
	fclose(f1);
	// ****************** Processing Starts***************** //
	// variables for memory and time cost records
	clock_t startWhole, endWhole;
	size_t totalMem = 0;
	size_t freeMem = 0;
	clock_t start, time1, time2, time3, time4, end;
	time1 = time2 = time3 = end = 0;
	start = clock();
	printf("\nStart processing...\n");
	cudaMemGetInfo(&freeMem, &totalMem);
	printf("...GPU free memory at beginning is %.0f MBites\n", (float)freeMem / 1048576.0f);
	f1 = fopen(fileLog, "a");
	fprintf(f1, "\nStart processing...\n");
	fprintf(f1, "...GPU free memory at beginning is %.0f MBites\n", (float)freeMem / 1048576.0f);
	fclose(f1);

	// ****************** Processing Starts***************** //
	// variables for memory and time cost records
	
	// allocate memory
	cudaMemGetInfo(&freeMem, &totalMem);
	printf("...GPU free memory(at beginning) is %.0f MBites\n", (float)freeMem / 1048576.0f);
	records[1] = (float)freeMem / 1048576.0f;
	cudaMalloc((void **)&d_StackA, totalSizeMax2 *sizeof(float));
	cudaMalloc((void **)&d_StackE, totalSizeMax2 *sizeof(float));
	cudaMemset(d_StackA, 0, totalSizeMax2*sizeof(float));
	cudaMemset(d_StackE, 0, totalSizeMax2*sizeof(float));
	//check GPU status
	cudaCheckErrors("****Memory allocating fails... GPU out of memory !!!!*****");
	// Create FFT plans
	cufftPlan3d(&fftPlanFwd, FFTx, FFTy, FFTz, CUFFT_R2C);
	cufftPlan3d(&fftPlanInv, FFTx, FFTy, FFTz, CUFFT_C2R);
	cudaCheckErrors("****Memory allocating fails... GPU out of memory !!!!*****");

	cudaMemGetInfo(&freeMem, &totalMem);
	printf("...GPU free memory(after partially mallocing) is %.0f MBites\n", (float)freeMem / 1048576.0f);
	records[2] = (float)freeMem / 1048576.0f;

	h_img = (float *)malloc(totalSize * sizeof(float));
	h_psf = (float *)malloc(totalSizePSF * sizeof(float));
	h_psf_bp = (float *)malloc(totalSizePSF * sizeof(float));
	h_decon = (float *)malloc(totalSize * sizeof(float));
	// ***read PSFs ***
	readtifstack(h_psf, filePSF, &psfSize[0]);
	if (flagUnmatch){
		readtifstack(h_psf_bp, filePSF_bp, &psfSize[0]);
	}
	// ***** Set GPU memory use mode based on images size and available GPU memory ****
	// gpuMemMode --> Unified memory in next version???
	// 0: Automatically set memory mode based on calculations; 
	// 1: sufficient memory; 2: memory optimized.
	if (gpuMemMode == 0){ //Automatically set memory mode based on calculations.
		if (freeMem > 4 * totalSizeMax * sizeof(float)){ // 7 more GPU variables
			gpuMemMode = 1;
			printf("\n GPU memory is sufficient, processing in efficient mode !!!\n");
		}
		else {// no more GPU variables needed
			gpuMemMode = 2;
			printf("\n GPU memory is optimized, processing in memory saved mode !!!\n");
		}
	}
	records[0] = gpuMemMode;
	double mySumPSF = 0;
	switch(gpuMemMode){
	case 1:// efficient GPU calculation
		cudaMalloc((void **)&d_StackT, totalSizeFFT *sizeof(float));
		cudaMalloc((void **)&d_PSFSpectrum, totalSizeSpectrum*sizeof(fComplex));
		cudaMalloc((void **)&d_FlippedPSFSpectrum, totalSizeSpectrum*sizeof(fComplex));
		cudaMalloc((void **)&d_StackESpectrum, totalSizeSpectrum*sizeof(fComplex));
		cudaMemGetInfo(&freeMem, &totalMem);
		printf("...GPU free memory(after mallocing) is %.0f MBites\n", (float)freeMem / 1048576.0f);
		// *** PSF Preparation
		//PSF 
		cudaMemcpy(d_StackE, h_psf, totalSizePSF* sizeof(float), cudaMemcpyHostToDevice);
		changestorageordergpu(d_StackA, d_StackE, PSFx, PSFy, PSFz, 1); //1: change tiff storage order to C storage order
		mySumPSF = sumcpu(h_psf, totalSizePSF);
		multivaluegpu(d_StackE, d_StackA, (float)(1 / mySumPSF), PSFx, PSFy, PSFz); // normalize PSF sum to 1
		if (!flagUnmatch){ // traditional backprojector matched PSF 
			flipPSF(d_StackA, d_StackE, PSFx, PSFy, PSFz); // flip PSF
			cudaMemcpy(h_psf_bp, d_StackA, totalSizePSF* sizeof(float), cudaMemcpyDeviceToHost);
		}
		cudaMemset(d_StackA, 0, totalSizeMax2*sizeof(float));
		padPSF(d_StackA, d_StackE, FFTx, FFTy, FFTz, PSFx, PSFy, PSFz, PSFox, PSFoy, PSFoz);
		cufftExecR2C(fftPlanFwd, (cufftReal *)d_StackA, (cufftComplex *)d_PSFSpectrum);
		//PSF bp
		cudaMemcpy(d_StackE, h_psf_bp, totalSizePSF* sizeof(float), cudaMemcpyHostToDevice);
		if (flagUnmatch){
			changestorageordergpu(d_StackA, d_StackE, PSFx, PSFy, PSFz, 1); //1: change tiff storage order to C storage order
			mySumPSF = sumcpu(h_psf_bp, totalSizePSF);
			multivaluegpu(d_StackE, d_StackA, (float)(1 / mySumPSF), PSFx, PSFy, PSFz); // normalize PSF sum to 1
		}
		cudaMemset(d_StackA, 0, totalSizeMax2*sizeof(float));
		padPSF(d_StackA, d_StackE, FFTx, FFTy, FFTz, PSFx, PSFy, PSFz, PSFox, PSFoy, PSFoz);
		cufftExecR2C(fftPlanFwd, (cufftReal *)d_StackA, (cufftComplex *)d_FlippedPSFSpectrum);

		// ******processing in batch*************
		for (imgNum = imgNumStart; imgNum <= imgNumEnd; imgNum += imgNumInterval){
			printf("\n***Image time point number: %d \n", imgNum);
			f1 = fopen(fileLog, "a");
			fprintf(f1, "\n***Image time point number: %d \n", imgNum);
			fclose(f1);
			sprintf(imgNumStr, "%d", imgNum);
			char *fileStack, *fileDecon;
			for (int iColor = 0; iColor < subFolderCount; iColor++){
				time1 = clock();
				if (flagMultiColor){
					printf("\n Processing time point %d color %d: %s \n", imgNum, iColor + 1, &subFolderNames[iColor][0]);
					f1 = fopen(fileLog, "a");
					fprintf(f1, "\n Processing time point %d color %d: %s \n", imgNum, iColor + 1, &subFolderNames[iColor][0]);
					fclose(f1);
					outFolder = concat(3, outMainFolder, &subFolderNames[iColor][0], "/");
					inFolder = concat(3, mainFolder, &subFolderNames[iColor][0], "/");
					deconFolder = concat(2, outFolder, "Decon/");
				}
				fileStack = concat(4, inFolder, fileNamePrefix, imgNumStr, ".tif");
				fileDecon = concat(4, deconFolder, "Decon_", imgNumStr, ".tif");

				// ****************Interpolation before registration**************** //////
				// ## check files
				if (!fexists(fileStack)){
					printf("***File does not exist: %s", fileStack);
					return 0;
				}
				else
					readtifstack(h_img, fileStack, &imSize[0]);
				cudaMemcpy(d_StackA, h_img, totalSize* sizeof(float), cudaMemcpyHostToDevice);
				//eliminate 0 in stacks
				maxvalue3Dgpu(d_StackA, d_StackA, (float)(SMALLVALUE), imx, imy, imz);
				changestorageordergpu(d_StackE, d_StackA, imx, imy, imz, 1); //1: change tiff storage order to C storage order
				padStack(d_StackA, d_StackE, FFTx, FFTy, FFTz, imx, imy, imz, imox, imoy, imoz);//
				// initialize estimation
				cudaMemcpy(d_StackE, d_StackA, totalSizeFFT* sizeof(float), cudaMemcpyDeviceToDevice);
				cudaCheckErrors("image preparing fail");
				//printf("...Initializing deconvolution iteration...\n");
				time2 = clock();
				for (int itNum = 1; itNum <= itNumForDecon; itNum++){
					// ### iterate with StackA and PSFA///////////////////
					// convolve StackE with PSFA
					//printf("...Processing iteration %d\n", it);
					cufftExecR2C(fftPlanFwd, (cufftReal *)d_StackE, (cufftComplex *)d_StackESpectrum);
					multicomplex3Dgpu(d_StackESpectrum, d_StackESpectrum, d_PSFSpectrum, FFTx, FFTy, (FFTz / 2 + 1));
					cufftExecC2R(fftPlanInv, (cufftComplex *)d_StackESpectrum, (cufftReal *)d_StackT);
					// divid StackA by StackTemp
					div3Dgpu(d_StackT, d_StackA, d_StackT, FFTx, FFTy, FFTz);   //// div3Dgpu does not work
					// convolve StackTemp with FlippedPSFA
					cufftExecR2C(fftPlanFwd, (cufftReal *)d_StackT, (cufftComplex *)d_StackESpectrum);
					multicomplex3Dgpu(d_StackESpectrum, d_StackESpectrum, d_FlippedPSFSpectrum, FFTx, FFTy, (FFTz / 2 + 1));
					cufftExecC2R(fftPlanInv, (cufftComplex *)d_StackESpectrum, (cufftReal *)d_StackT);//test
					// multiply StackE and StackTemp
					multi3Dgpu(d_StackE, d_StackE, d_StackT, FFTx, FFTy, FFTz);//
					maxvalue3Dgpu(d_StackE, d_StackE, float(SMALLVALUE), FFTx, FFTy, FFTz);
				}
				time3 = clock();
				cropStack(d_StackE, d_StackT, FFTx, FFTy, FFTz, imx, imy, imz, imox, imoy, imoz);//
				//printf("...Deconvolution completed ! ! !\n");
				cudaThreadSynchronize();
				changestorageordergpu(d_StackE, d_StackT, imx, imy, imz, -1); //-1: change C storage order to tiff storage order
				cudaMemcpy(h_decon, d_StackE, totalSize* sizeof(float), cudaMemcpyDeviceToHost);
				writetifstack(fileDecon, h_decon, &imSize[0], bitPerSample);//set bitPerSample as input images

				time4 = clock();
				cudaMemGetInfo(&freeMem, &totalMem);
				printf("...GPU free memory (during processing) is %.0f MBites\n", (float)freeMem / 1048576.0f);
				records[4] = (float)freeMem / 1048576.0f;

				time4 = clock();
				cudaMemGetInfo(&freeMem, &totalMem);
				printf("...GPU free memory (during processing) is %.0f MBites\n", (float)freeMem / 1048576.0f);
				records[4] = (float)freeMem / 1048576.0f;

				gpuMemMode = int(records[0]);
				switch (gpuMemMode){
				case 1:
					printf("	...Sufficient GPU memory, running in efficient mode !!!\n");
					break;
				case 2:
					printf("	...GPU memory optimized, running in memory saved mode !!!\n");
					break;

				default:
					printf("	...Not enough GPU memory, no deconvolution performed !!!\n");
				}
				printf("	...GPU free memory during deconvolution is %.0f MBites\n", records[4]);
				printf("	...all iteration time cost: %2.3f s\n", (float)(time3 - time2) / CLOCKS_PER_SEC);
				printf("	...time cost for current image: %2.3f s\n", (float)(time4 - time1) / CLOCKS_PER_SEC);

				f1 = fopen(fileLog, "a");
				switch (gpuMemMode){
				case 1:
					fprintf(f1, "	...Sufficient GPU memory, running in efficient mode !!!\n");
					break;
				case 2:
					fprintf(f1, "	...GPU memory optimized, running in memory saved mode !!!\n");
					break;
				default:
					fprintf(f1, "	...Not enough GPU memory, no deconvolution performed !!!\n");
				}
				fprintf(f1, "	...GPU free memory (during processing) is %.0f MBites\n", records[4]);
				fprintf(f1, "	... time cost for current image: %2.3f s\n", (float)(time4 - time1) / CLOCKS_PER_SEC);
				fclose(f1);
			}
		}
		// release CUDA variables
		cudaFree(d_StackT); cudaFree(d_PSFSpectrum); cudaFree(d_FlippedPSFSpectrum); cudaFree(d_StackESpectrum);
		break;
	case 2:
		time1 = clock();
		h_StackA = (float *)malloc(totalSizeMax * sizeof(float));
		h_StackE = (float *)malloc(totalSizeMax * sizeof(float));
		h_PSFSpectrum = (fComplex *)malloc(totalSizeSpectrum*sizeof(fComplex));
		h_FlippedPSFSpectrum = (fComplex *)malloc(totalSizeSpectrum*sizeof(fComplex));
		h_StackESpectrum = (fComplex *)malloc(totalSizeSpectrum*sizeof(fComplex));
		d_StackESpectrum = (fComplex *)d_StackE; // share the same physic memory
		// *** PSF Preparation
		//PSF 
		cudaMemcpy(d_StackE, h_psf, totalSizePSF* sizeof(float), cudaMemcpyHostToDevice);
		changestorageordergpu(d_StackA, d_StackE, PSFx, PSFy, PSFz, 1); //1: change tiff storage order to C storage order
		mySumPSF = sumcpu(h_psf, totalSizePSF);
		multivaluegpu(d_StackE, d_StackA, (float)(1 / mySumPSF), PSFx, PSFy, PSFz); // normalize PSF sum to 1
		if (!flagUnmatch){ // traditional backprojector matched PSF 
			flipPSF(d_StackA, d_StackE, PSFx, PSFy, PSFz); // flip PSF
			cudaMemcpy(h_psf_bp, d_StackA, totalSizePSF* sizeof(float), cudaMemcpyDeviceToHost);
		}
		cudaMemset(d_StackA, 0, totalSizeMax2*sizeof(float));
		padPSF(d_StackA, d_StackE, FFTx, FFTy, FFTz, PSFx, PSFy, PSFz, PSFox, PSFoy, PSFoz);
		cufftExecR2C(fftPlanFwd, (cufftReal *)d_StackA, (cufftComplex *)d_StackESpectrum);
		cudaMemcpy(h_PSFSpectrum, d_StackESpectrum, totalSizeSpectrum*sizeof(fComplex), cudaMemcpyDeviceToHost);
		//PSF bp
		cudaMemcpy(d_StackE, h_psf_bp, totalSizePSF* sizeof(float), cudaMemcpyHostToDevice);
		if (flagUnmatch){
			changestorageordergpu(d_StackA, d_StackE, PSFx, PSFy, PSFz, 1); //1: change tiff storage order to C storage order
			mySumPSF = sumcpu(h_psf_bp, totalSizePSF);
			multivaluegpu(d_StackE, d_StackA, (float)(1 / mySumPSF), PSFx, PSFy, PSFz); // normalize PSF sum to 1
		}
		cudaMemset(d_StackA, 0, totalSizeMax2*sizeof(float));
		padPSF(d_StackA, d_StackE, FFTx, FFTy, FFTz, PSFx, PSFy, PSFz, PSFox, PSFoy, PSFoz);
		cufftExecR2C(fftPlanFwd, (cufftReal *)d_StackA, (cufftComplex *)d_StackESpectrum);
		cudaMemcpy(h_FlippedPSFSpectrum, d_StackESpectrum, totalSizeSpectrum*sizeof(fComplex), cudaMemcpyDeviceToHost);

		// ******processing in batch*************
		for (imgNum = imgNumStart; imgNum <= imgNumEnd; imgNum += imgNumInterval){
			printf("\n***Image time point number: %d \n", imgNum);
			f1 = fopen(fileLog, "a");
			fprintf(f1, "\n***Image time point number: %d \n", imgNum);
			fclose(f1);
			sprintf(imgNumStr, "%d", imgNum);
			char *fileStack, *fileDecon;
			for (int iColor = 0; iColor < subFolderCount; iColor++){
				time1 = clock();
				if (flagMultiColor){
					printf("\n Processing time point %d color %d: %s \n", imgNum, iColor + 1, &subFolderNames[iColor][0]);
					f1 = fopen(fileLog, "a");
					fprintf(f1, "\n Processing time point %d color %d: %s \n", imgNum, iColor + 1, &subFolderNames[iColor][0]);
					fclose(f1);
					outFolder = concat(3, outMainFolder, &subFolderNames[iColor][0], "/");
					inFolder = concat(3, mainFolder, &subFolderNames[iColor][0], "/");
					deconFolder = concat(2, outFolder, "Decon/");
				}
				fileStack = concat(4, inFolder, fileNamePrefix, imgNumStr, ".tif");
				fileDecon = concat(4, deconFolder, "Decon_", imgNumStr, ".tif");

				// ****************Interpolation before registration**************** //////
				// ## check files
				if (!fexists(fileStack)){
					printf("***File does not exist: %s", fileStack);
					return 0;
				}
				else
					readtifstack(h_img, fileStack, &imSize[0]);
				cudaMemcpy(d_StackA, h_img, totalSize* sizeof(float), cudaMemcpyHostToDevice);
				//eliminate 0 in stacks
				maxvalue3Dgpu(d_StackA, d_StackA, (float)(SMALLVALUE), imx, imy, imz);
				changestorageordergpu(d_StackE, d_StackA, imx, imy, imz, 1); //1: change tiff storage order to C storage order
				padStack(d_StackA, d_StackE, FFTx, FFTy, FFTz, imx, imy, imz, imox, imoy, imoz);//

				// initialize estimation
				cudaMemcpy(h_StackA, d_StackA, totalSizeFFT * sizeof(float), cudaMemcpyDeviceToHost);
				cudaMemcpy(h_StackE, d_StackA, totalSizeFFT * sizeof(float), cudaMemcpyDeviceToHost);

				d_PSFSpectrum = (fComplex *)d_StackA; // share the same physic memory
				d_FlippedPSFSpectrum = (fComplex *)d_StackA; // share the same physic memory
				cudaCheckErrors("image preparing fail");
				//printf("...Initializing deconvolution iteration...\n");
				time2 = clock();
				for (int itNum = 1; itNum <= itNumForDecon; itNum++){
					// ### iterate with StackA and PSFA///////////////////
					// convolve StackE with PSFA
					//printf("...Processing iteration %d\n", it);
					cufftExecR2C(fftPlanFwd, (cufftReal *)d_StackA, (cufftComplex *)d_StackESpectrum);
					cudaMemcpy(d_PSFSpectrum, h_PSFSpectrum, totalSizeSpectrum*sizeof(fComplex), cudaMemcpyHostToDevice);
					multicomplex3Dgpu(d_StackESpectrum, d_StackESpectrum, d_PSFSpectrum, FFTx, FFTy, (FFTz / 2 + 1));
					cufftExecC2R(fftPlanInv, (cufftComplex *)d_StackESpectrum, (cufftReal *)d_StackA);

					// divid StackA by StackTemp
					cudaMemcpy(d_StackE, h_StackA, totalSizeFFT* sizeof(float), cudaMemcpyHostToDevice);
					div3Dgpu(d_StackA, d_StackE, d_StackA, FFTx, FFTy, FFTz);   //// div3Dgpu does not work
					// convolve StackTemp with FlippedPSFA
					cufftExecR2C(fftPlanFwd, (cufftReal *)d_StackA, (cufftComplex *)d_StackESpectrum);
					cudaMemcpy(d_FlippedPSFSpectrum, h_FlippedPSFSpectrum, totalSizeSpectrum*sizeof(fComplex), cudaMemcpyHostToDevice);
					multicomplex3Dgpu(d_StackESpectrum, d_StackESpectrum, d_FlippedPSFSpectrum, FFTx, FFTy, (FFTz / 2 + 1));
					cufftExecC2R(fftPlanInv, (cufftComplex *)d_StackESpectrum, (cufftReal *)d_StackA);//test
					// multiply StackE and StackTemp
					cudaMemcpy(d_StackE, h_StackE, totalSizeFFT* sizeof(float), cudaMemcpyHostToDevice);
					multi3Dgpu(d_StackA, d_StackE, d_StackA, FFTx, FFTy, FFTz);//
					cudaMemcpy(h_StackE, d_StackA, totalSizeFFT* sizeof(float), cudaMemcpyDeviceToHost);
				}
				time3 = clock();
				cropStack(d_StackA, d_StackE, FFTx, FFTy, FFTz, imx, imy, imz, imox, imoy, imoz);//
				//printf("...Deconvolution completed ! ! !\n");
				cudaThreadSynchronize();
				//## Write stack to tiff image
				changestorageordergpu(d_StackA, d_StackE, imx, imy, imz, -1); //-1: change C storage order to tiff storage order
				cudaMemcpy(h_decon, d_StackA, totalSize* sizeof(float), cudaMemcpyDeviceToHost);
				writetifstack(fileDecon, h_decon, &imSize[0], bitPerSample);//set bitPerSample as input images

				time4 = clock();
				cudaMemGetInfo(&freeMem, &totalMem);
				printf("...GPU free memory (during processing) is %.0f MBites\n", (float)freeMem / 1048576.0f);
				records[4] = (float)freeMem / 1048576.0f;

				gpuMemMode = int(records[0]);
				switch (gpuMemMode){
				case 1:
					printf("	...Sufficient GPU memory, running in efficient mode !!!\n");
					break;
				case 2:
					printf("	...GPU memory optimized, running in memory saved mode !!!\n");
					break;

				default:
					printf("	...Not enough GPU memory, no deconvolution performed !!!\n");
				}
				printf("	...GPU free memory during deconvolution is %.0f MBites\n", records[4]);
				printf("	...all iteration time cost: %2.3f s\n", (float)(time3 - time2) / CLOCKS_PER_SEC);
				printf("	...time cost for current image: %2.3f s\n", (float)(time4 - time1) / CLOCKS_PER_SEC);

				f1 = fopen(fileLog, "a");
				switch (gpuMemMode){
				case 1:
					fprintf(f1, "	...Sufficient GPU memory, running in efficient mode !!!\n");
					break;
				case 2:
					fprintf(f1, "	...GPU memory optimized, running in memory saved mode !!!\n");
					break;
				default:
					fprintf(f1, "	...Not enough GPU memory, no deconvolution performed !!!\n");
				}
				fprintf(f1, "	...GPU free memory (during processing) is %.0f MBites\n", records[4]);
				fprintf(f1, "	... time cost for current image: %2.3f s\n", (float)(time4 - time1) / CLOCKS_PER_SEC);
				fclose(f1);
			}
		}
		// release CPU memory
		free(h_StackA);  free(h_StackE); free(h_PSFSpectrum); free(h_FlippedPSFSpectrum); free(h_StackESpectrum);
		break;
	default:
		printf("\n****Wrong gpuMemMode setup, no deconvolution performed !!! ****\n");
		f1 = fopen(fileLog, "a");
		fprintf(f1, "\n****Wrong gpuMemMode setup, no deconvolution performed !!! ****\n");
		fclose(f1);
		return -1;
	}
	free(h_img);  free(h_psf); free(h_psf_bp); free(h_decon);
	// release GPU memory
	cudaFree(d_StackA);
	cudaFree(d_StackE);
	// destroy plans
	cufftDestroy(fftPlanFwd);
	cufftDestroy(fftPlanInv);
	end = clock();
	cudaMemGetInfo(&freeMem, &totalMem);
	records[5] = (float)freeMem / 1048576.0f;
	records[6] = (float)(time3 - time2) / CLOCKS_PER_SEC;
	records[7] = (float)(time4 - time1) / CLOCKS_PER_SEC;
	records[8] = (float)(time3 - time2) / CLOCKS_PER_SEC;
	records[9] = (float)(end - start) / CLOCKS_PER_SEC;

	printf("\nGPU free memory after whole processing is %.0f MBites\n", records[5]);
	endWhole = clock();
	printf("Total time cost for whole processing is %2.3f s\n", records[9]);
	f1 = fopen(fileLog, "a");
	fprintf(f1, "\nGPU free memory after whole processing is %.0f MBites\n", records[5]);
	fprintf(f1, "Total time cost for whole processing is %2.3f s\n", records[9]);
	fclose(f1);
	return 0;
}

int fusion_dualview_batch(char *outMainFolder, char *folder1, char *folder2, char *fileNamePrefix1, char *fileNamePrefix2, int imgNumStart, int imgNumEnd, int imgNumInterval, int imgNumTest,
	float *pixelSize1, float *pixelSize2, int regMode, int imRotation, int flagInitialTmx, float *iTmx, float FTOL, int itLimit, char *filePSF1, char *filePSF2,
	int itNumForDecon, int deviceNum, int *flagSaveInterFiles, int bitPerSample, float *records, bool flagUnmatch, char *filePSF_bp1, char *filePSF_bp2){ // Next version: variable arguments input
	// regMode--> 0: no registration; 1: one image only
	//			2: dependently, based on the results of last time point; 3: independently
	// flagInitialTmx --> 0: default matrix; 1: input matrix; 2: do 3D translation registation based on phase registration.; 3: 2D registration
	// flagSaveInterFiles: 8 elements --> 1: save files; 0: not; [0] is currently not used.
	//					[0]: Intermediate outputs; [1]: reg A; [2]: reg B;
	//					[3]- [5]: Decon max projections Z, Y, X;
	//					[6], [7]: Decon 3D max projections: Y, X;
	
	char *outFolder, *inFolder1, *inFolder2;
	// ***** check if multitple color processing
	char mainFolder[MAX_PATH];
	bool flagMultiColor = false;
	int multiColor = atoi(folder1);
	int subFolderCount = 1;
	char subFolderNames[20][MAX_PATH];
	if (multiColor == 1){ // trigger multiple color
#ifdef _WIN32 
		strcpy(mainFolder, folder2);
		flagMultiColor = true;
#else
		fprintf(stderr, "*** Multi-color processing is currently not supported on Linux\n");
#endif
	}
	if (flagMultiColor){
#ifdef _WIN32 
		subFolderCount = findSubFolders(&subFolderNames[0][0], mainFolder);

		if (subFolderCount > 20)
			fprintf(stderr, "*** Number of subfolders: %d; two many subfolders\n", subFolderCount);
		else{
			printf("Procecing multicolor data: %d colors\n", subFolderCount);
			for (int j = 0; j < subFolderCount; j++)
				printf("...Subfolders %d: %s\n", j + 1, &subFolderNames[j][0]);
		}
		inFolder1 = concat(3, mainFolder, &subFolderNames[0][0], "/SPIMA/");
		inFolder2 = concat(3, mainFolder, &subFolderNames[0][0], "/SPIMB/");
#endif
	}
	else{
		inFolder1 = folder1;
		inFolder2 = folder2;
	}

	// ************get basic input images and PSFs information ******************
	unsigned int  imSizeIn1[3], imSizeIn2[3], psfSize[3], imSizeTemp[3];
	int imgNum = imgNumStart;
	if (regMode == 3)
		imgNum = imgNumTest;
	char imgNumStr[20];
	sprintf(imgNumStr, "%d", imgNum);
	char *fileStack1 = concat(4, inFolder1, fileNamePrefix1, imgNumStr, ".tif"); // TIFF file to get image information
	char *fileStack2 = concat(4, inFolder2, fileNamePrefix2, imgNumStr, ".tif");
	// **** check image files and image size ***
	unsigned short bitPerSample_input;
	if (!fexists(fileStack1)){
		fprintf(stderr, "***File does not exist: %s\n", fileStack1);
		fprintf(stderr, "*** FAILED - ABORTING\n");
		exit(1);
	}
	if (!fexists(fileStack2)){
		fprintf(stderr, "***File does not exist: %s\n", fileStack2);
		fprintf(stderr, "*** FAILED - ABORTING\n");
		exit(1);
	}
	if (!fexists(filePSF1)){
		fprintf(stderr, "***File does not exist: %s\n", filePSF1);
		fprintf(stderr, "*** FAILED - ABORTING\n");
		exit(1);
	}
	if (!fexists(filePSF2)){
		fprintf(stderr, "***File does not exist: %s\n", filePSF2);
		fprintf(stderr, "*** FAILED - ABORTING\n");
		exit(1);
	}
	if (flagUnmatch){// use unmatched back projectors
		if (!fexists(filePSF_bp1)){
			fprintf(stderr, "***File does not exist: %s\n", filePSF_bp1);
			fprintf(stderr, "*** FAILED - ABORTING\n");
			exit(1);
		}
		if (!fexists(filePSF_bp2)){
			fprintf(stderr, "***File does not exist: %s\n", filePSF_bp2);
			fprintf(stderr, "*** FAILED - ABORTING\n");
			exit(1);
		}
	}

	bitPerSample_input = gettifinfo(fileStack1, &imSizeIn1[0]);
	if (bitPerSample_input != 16 && bitPerSample_input != 32){
		fprintf(stderr, "***Input images are not supported, please use 16-bit or 32-bit image !!!\n");
		fprintf(stderr, "*** FAILED - ABORTING\n");
		exit(1);
	}
	(void)gettifinfo(fileStack2, &imSizeIn2[0]);
	(void)gettifinfo(filePSF1, &psfSize[0]);
	(void)gettifinfo(filePSF2, &imSizeTemp[0]);
	if ((psfSize[0] != imSizeTemp[0]) || (psfSize[1] != imSizeTemp[1]) || (psfSize[2] != imSizeTemp[2])){
		fprintf(stderr, "***PSF image size are not consistent to each other !!!\n");
		fprintf(stderr, "*** FAILED - ABORTING\n");
		exit(1);
	}
	if (flagUnmatch){
		(void)gettifinfo(filePSF_bp1, &imSizeTemp[0]);
		if ((psfSize[0] != imSizeTemp[0]) || (psfSize[1] != imSizeTemp[1]) || (psfSize[2] != imSizeTemp[2])){
			fprintf(stderr, "***PSF image size are not consistent to each other !!!\n");
			fprintf(stderr, "*** FAILED - ABORTING\n");
			exit(1);
		}
		(void)gettifinfo(filePSF_bp2, &imSizeTemp[0]);
		if ((psfSize[0] != imSizeTemp[0]) || (psfSize[1] != imSizeTemp[1]) || (psfSize[2] != imSizeTemp[2])){
			fprintf(stderr, "***PSF image size are not consistent to each other !!!\n");
			fprintf(stderr, "*** FAILED - ABORTING\n");
			exit(1);
		}
	}
	// ****************** Create output folders***************** //
	char *deconFolder, *tmxFolder, *regFolder1, *regFolder2,
		*deconFolderMP_XY, *deconFolderMP_YZ, *deconFolderMP_ZX, *deconFolderMP_3D_X, *deconFolderMP_3D_Y;
	// flagSaveInterFiles: 8 elements --> 1: save files; 0: not
	//					[0]: Intermediate outputs; [1]: reg A; [2]: reg B;
	//					[3]- [5]: Decon max projections Z, Y, X;
	//					[6], [7]: Decon 3D max projections: Y, X;
	if (flagMultiColor){
#ifdef _WIN32 
		CreateDirectory(outMainFolder, NULL);
		for (int j = 0; j < subFolderCount; j++){
			outFolder = concat(3, outMainFolder, &subFolderNames[j][0], "/");
			inFolder1 = concat(3, mainFolder, &subFolderNames[j][0], "/SPIMA/");
			inFolder2 = concat(3, mainFolder, &subFolderNames[j][0], "/SPIMB/");
			deconFolder = concat(2, outFolder, "Decon/");
			tmxFolder = concat(2, outFolder, "TMX/");
			regFolder1 = concat(2, outFolder, "RegA/");
			regFolder2 = concat(2, outFolder, "RegB/");
			deconFolderMP_XY = concat(2, deconFolder, "MP_ZProj/");
			deconFolderMP_YZ = concat(2, deconFolder, "MP_XProj/");
			deconFolderMP_ZX = concat(2, deconFolder, "MP_YProj/");
			deconFolderMP_3D_X = concat(2, deconFolder, "MP_3D_Xaxis/");
			deconFolderMP_3D_Y = concat(2, deconFolder, "MP_3D_Yaxis/");
			CreateDirectory(outFolder, NULL);
			CreateDirectory(deconFolder, NULL);
			CreateDirectory(tmxFolder, NULL); // currentlly the TMX file is always saved
			if (flagSaveInterFiles[1] == 1) CreateDirectory(regFolder1, NULL);
			if (flagSaveInterFiles[2] == 1) CreateDirectory(regFolder2, NULL);
			if (flagSaveInterFiles[3] == 1) CreateDirectory(deconFolderMP_XY, NULL);
			if (flagSaveInterFiles[4] == 1) CreateDirectory(deconFolderMP_YZ, NULL);
			if (flagSaveInterFiles[5] == 1) CreateDirectory(deconFolderMP_ZX, NULL);
			if (flagSaveInterFiles[6] == 1) CreateDirectory(deconFolderMP_3D_X, NULL);
			if (flagSaveInterFiles[7] == 1) CreateDirectory(deconFolderMP_3D_Y, NULL);
			free(outFolder); free(inFolder1); free(inFolder2); free(deconFolder); free(tmxFolder); free(regFolder1); //
			free(regFolder2);  free(deconFolderMP_XY); free(deconFolderMP_YZ); free(deconFolderMP_ZX);
			free(deconFolderMP_3D_X); free(deconFolderMP_3D_Y);
		}
#endif
	}
	else{
		outFolder = outMainFolder;
		inFolder1 = folder1;
		inFolder2 = folder2;
		deconFolder = concat(2, outFolder, "Decon/");
		tmxFolder = concat(2, outFolder, "TMX/");
		regFolder1 = concat(2, outFolder, "RegA/");
		regFolder2 = concat(2, outFolder, "RegB/");
		deconFolderMP_XY = concat(2, deconFolder, "MP_ZProj/");
		deconFolderMP_YZ = concat(2, deconFolder, "MP_XProj/");
		deconFolderMP_ZX = concat(2, deconFolder, "MP_YProj/");
		deconFolderMP_3D_X = concat(2, deconFolder, "MP_3D_Xaxis/");
		deconFolderMP_3D_Y = concat(2, deconFolder, "MP_3D_Yaxis/");

#ifdef _WIN32 
		CreateDirectory(outFolder, NULL);
		CreateDirectory(deconFolder, NULL);
		CreateDirectory(tmxFolder, NULL); // currentlly the TMX file is always saved
		if (flagSaveInterFiles[1] == 1) CreateDirectory(regFolder1, NULL);
		if (flagSaveInterFiles[2] == 1) CreateDirectory(regFolder2, NULL);
		if (flagSaveInterFiles[3] == 1) CreateDirectory(deconFolderMP_XY, NULL);
		if (flagSaveInterFiles[4] == 1) CreateDirectory(deconFolderMP_YZ, NULL);
		if (flagSaveInterFiles[5] == 1) CreateDirectory(deconFolderMP_ZX, NULL);
		if (flagSaveInterFiles[6] == 1) CreateDirectory(deconFolderMP_3D_X, NULL);
		if (flagSaveInterFiles[7] == 1) CreateDirectory(deconFolderMP_3D_Y, NULL);
#else
		mkdir(outFolder, 0755);
		mkdir(deconFolder, 0755);
		mkdir(tmxFolder, 0755);
		if (flagSaveInterFiles[1] == 1) mkdir(regFolder1, 0755);
		if (flagSaveInterFiles[2] == 1) mkdir(regFolder2, 0755);
		if (flagSaveInterFiles[3] == 1) mkdir(deconFolderMP_XY, 0755);
		if (flagSaveInterFiles[4] == 1) mkdir(deconFolderMP_YZ, 0755);
		if (flagSaveInterFiles[5] == 1) mkdir(deconFolderMP_ZX, 0755);
		if (flagSaveInterFiles[6] == 1) mkdir(deconFolderMP_3D_X, 0755);
		if (flagSaveInterFiles[7] == 1) mkdir(deconFolderMP_3D_Y, 0755);
#endif
	}

	// ****************** calculate images' size ************************* //
	int imx, imy, imz;
	unsigned int imSize[3], imSize1[3], imSize2[3];
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
	int
		PSFx, PSFy, PSFz;
	PSFx = psfSize[0], PSFy = psfSize[1], PSFz = psfSize[2];

	//FFT size
	int
		FFTx, FFTy, FFTz;

	FFTx = snapTransformSize(imx);// snapTransformSize(imx + PSFx - 1);
	FFTy = snapTransformSize(imy);// snapTransformSize(imy + PSFy - 1);
	FFTz = snapTransformSize(imz);// snapTransformSize(imz + PSFz - 1);

	// total pixel count for each images
	int totalSizeIn1 = imSizeIn1[0] * imSizeIn1[1] * imSizeIn1[2]; // in floating format
	int totalSizeIn2 = imSizeIn2[0] * imSizeIn2[1] * imSizeIn2[2]; // in floating format
	int totalSize1 = imx1*imy1*imz1; // in floating format
	int totalSize2 = imx2*imy2*imz2; // in floating format
	int totalSize12 = (totalSize1 > totalSize2) ? totalSize1 : totalSize2;
	int totalSize = totalSize1; // in floating format
	int totalSizePSF = PSFx * PSFy * PSFz;
	int totalSizeFFT = FFTx*FFTy*(FFTz / 2 + 1); // in complex floating format
	int totalSizeMax = (totalSize1 > totalSizeFFT * 2) ? totalSize1 : totalSizeFFT * 2; // in floating format

	// print GPU devices information
	cudaSetDevice(deviceNum);

	// Log file
	FILE *f1 = NULL, *f2 = NULL, *f3 = NULL;
	char *fileLog = concat(2, outMainFolder, "ProcessingLog.txt");

	// print images information
	printf("Image information:\n");
	printf("...Image A size %d x %d x %d\n     ...pixel size %2.4f x %2.4f x %2.4f um\n", imSizeIn1[0], imSizeIn1[1], imSizeIn1[2], pixelSize1[0], pixelSize1[1], pixelSize1[2]);
	printf("...Image B size %d x %d x %d\n     ...pixel size %2.4f x %2.4f x %2.4f um\n", imSizeTemp[0], imSizeTemp[1], imSizeTemp[2], pixelSizeTemp[0], pixelSizeTemp[1], pixelSizeTemp[2]);
	printf("...PSF size %d x %d x %d\n", psfSize[0], psfSize[1], psfSize[2]);
	printf("...FFT size: %d x %d x %d\n", FFTx, FFTy, FFTz);
	printf("...Output Image size %d x %d x %d \n    ......pixel size %2.4f x %2.4f x %2.4f um\n\n", imSize[0], imSize[1], imSize[2], pixelSize[0], pixelSize[1], pixelSize[2]);
	printf("...Image number from %d to %d with step %d\n", imgNumStart, imgNumEnd, imgNumInterval);
	switch (regMode){
	case 0:
		printf("...No registration\n"); break;
	case 1:
		printf("...One registration for all images, test image number: %d\n", imgNumTest); break;
	case 2:
		printf("...Perform registration for all images dependently\n"); break;
	case 3:
		printf("...Perform registration for all images independently\n"); break;
	default:
		printf("...regMode incorrect !!!\n"); 
		return 0;
	}

	switch (imRotation){
	case 0:
		printf("...No rotation on image B\n"); break;
	case 1:
		printf("...Rotate image B by 90 degree along Y axis\n"); break;
	case -1:
		printf("...Rotate image B by -90 degree along Y axis\n"); break;
	}

	switch (flagInitialTmx){
	case 1:
		printf("...Initial transformation matrix: based on input matrix\n"); break;
	case 2:
		printf("...Initial transformation matrix: by phase translation\n"); break;
	case 3:
		printf("...Initial transformation matrix: by 2D registration\n"); break;
	default:
		printf("...Initial transformation matrix: Default\n");
	}
	if (flagUnmatch) printf("\n...Unmatched back projectors for joint deconvolution: yes\n");
	else printf("\n...Unmatched back projectors for joint deconvolution: no\n");
	printf("\n...Iteration number for joint deconvolution:%d\n", itNumForDecon);
	printf("\n...GPU Device %d is used...\n\n", deviceNum);

	time_t now;
	time(&now);
	//****Write information to log file***
	f1 = fopen(fileLog, "w");
	// print images information
	fprintf(f1, "diSPIMFusion: %s\n", ctime(&now));
	if (flagMultiColor){
		fprintf(f1, "Multicolor data: %d colors\n", subFolderCount);
		fprintf(f1, "...Input directory: %s\n", folder2);
		for (int j = 0; j < subFolderCount; j++)
			fprintf(f1, "     ...Subfolders %d: %s\n", j + 1, &subFolderNames[j][0]);
		fprintf(f1, "...Output directory: %s\n", outMainFolder);
	}
	else{
		fprintf(f1, "Single color data:\n");
		fprintf(f1, "...SPIMA input directory: %s\n", folder1);
		fprintf(f1, "...SPIMB input directory: %s\n", folder2);
		fprintf(f1, "...Output directory: %s\n", outMainFolder);
	}

	fprintf(f1, "\nImage information:\n");
	fprintf(f1, "...Image A size %d x %d x %d\n     ...pixel size %2.4f x %2.4f x %2.4f um\n", imSizeIn1[0], imSizeIn1[1], imSizeIn1[2], pixelSize1[0], pixelSize1[1], pixelSize1[2]);
	fprintf(f1, "...Image B size %d x %d x %d\n     ...pixel size %2.4f x %2.4f x %2.4f um\n", imSizeTemp[0], imSizeTemp[1], imSizeTemp[2], pixelSizeTemp[0], pixelSizeTemp[1], pixelSizeTemp[2]);
	fprintf(f1, "...PSF size %d x %d x %d\n", psfSize[0], psfSize[1], psfSize[2]);
	fprintf(f1, "...FFT size: %d x %d x %d\n", FFTx, FFTy, FFTz);
	fprintf(f1, "...Output Image size %d x %d x %d \n    ......pixel size %2.4f x %2.4f x %2.4f um\n\n", imSize[0], imSize[1], imSize[2], pixelSize[0], pixelSize[1], pixelSize[2]);
	fprintf(f1, "...Image number from %d to %d with step %d\n", imgNumStart, imgNumEnd, imgNumInterval);
	switch (regMode){
	case 0:
		fprintf(f1, "...No registration\n"); break;
	case 1:
		fprintf(f1, "...One registration for all images, test image number: %d\n", imgNumTest); break;
	case 2:
		fprintf(f1, "...Perform registration for all images dependently\n"); break;
	case 3:
		fprintf(f1, "...Perform registration for all images independently\n"); break;
	default:
		fprintf(f1, "...regMode incorrect !!!\n");
		return 0;
	}

	switch (imRotation){
	case 0:
		fprintf(f1, "...No rotation on image B\n"); break;
	case 1:
		fprintf(f1, "...Rotate image B by 90 degree along Y axis\n"); break;
	case -1:
		fprintf(f1, "...Rotate image B by -90 degree along Y axis\n"); break;
	}

	switch (flagInitialTmx){
	case 1:
		fprintf(f1, "...Initial transformation matrix: based on input matrix\n"); break;
	case 2:
		fprintf(f1, "...Initial transformation matrix: by phase translation\n"); break;
	case 3:
		fprintf(f1, "...Initial transformation matrix: by 2D registration\n"); break;
	default:
		fprintf(f1, "...Initial transformation matrix: Default\n");
	}

	fprintf(f1, "...Registration convergence threshold:%f\n", FTOL);
	fprintf(f1, "...Registration maximum sub-iteration number:%d\n", itLimit);
	if (flagUnmatch) fprintf(f1, "\n...Unmatched back projectors for joint deconvolution: yes\n");
	else fprintf(f1, "\n...Unmatched back projectors for joint deconvolution: no\n");
	fprintf(f1, "...Iteration number for joint deconvolution:%d\n", itNumForDecon);
	fprintf(f1, "\n...GPU Device %d is used...\n\n", deviceNum);
	fclose(f1);
	// ****************** Processing Starts***************** //
	// variables for memory and time cost records
	clock_t startWhole, endWhole, start, end;
	size_t totalMem = 0;
	size_t freeMem = 0;
	printf("\nStart processing...\n");
	cudaMemGetInfo(&freeMem, &totalMem);
	printf("...GPU free memory at beginning is %.0f MBites\n", (float)freeMem / 1048576.0f);
	f1 = fopen(fileLog, "a");
	fprintf(f1, "\nStart processing...\n");
	fprintf(f1, "...GPU free memory at beginning is %.0f MBites\n", (float)freeMem / 1048576.0f);
	fclose(f1);
	startWhole = clock();
	// ***** Set GPU memory use mode based on images size and available GPU memory ****
	int gpuMemMode = 0;
	// gpuMemMode--> -1: not enough gpu memory; 0-3: enough GPU memory; 
	if (freeMem < 4 * totalSizeMax * sizeof(float)){ // 4 variables + 2 FFT calculation space + a few additional space
		gpuMemMode = -1;
		f1 = fopen(fileLog, "a");
		fprintf(f1, "***Available GPU memory is insufficient, processing terminated !!!\n ****Total memory required: %.0f MBites\n", 4 * totalSizeMax * sizeof(float) / 1048576.0f);
		fclose(f1);
		fprintf(stderr, "***Available GPU memory is insufficient, processing terminated !!!\n ****Total memory required: %.0f MBites\n", 4 * totalSizeMax * sizeof(float) / 1048576.0f);
		fprintf(stderr, "*** FAILED - ABORTING\n");
		exit(1);
	}

	// variables
	float
		*h_img1,
		*h_img2,
		*h_StackA,
		*h_StackB,
		*h_psf1,
		*h_psf2,
		*h_psf_bp1,
		*h_psf_bp2,
		*h_reg,
		*h_decon;
	float
		*d_imgE;
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
	cudaArray *d_Array1, *d_Array2;
	float *h_aff12 = (float *)malloc((NDIM)* sizeof(float));
	h_img1 = (float *)malloc(totalSizeIn1 * sizeof(float));
	h_img2 = (float *)malloc(totalSizeIn2 * sizeof(float));
	h_StackA = (float *)malloc(totalSize12 * sizeof(float));
	h_StackB = (float *)malloc(totalSize12 * sizeof(float));
	h_psf1 = (float *)malloc(totalSizePSF * sizeof(float));
	h_psf2 = (float *)malloc(totalSizePSF * sizeof(float));
	h_psf_bp1 = (float *)malloc(totalSizePSF * sizeof(float));
	h_psf_bp2 = (float *)malloc(totalSizePSF * sizeof(float));
	h_reg = (float *)malloc(totalSize * sizeof(float));
	h_decon = (float *)malloc(totalSize * sizeof(float));

	// regMode--> 0: no registration; 1: one image only
	//			3: dependently, based on the results of last time point; 4: independently
	float *h_affInitial = (float *)malloc((NDIM)* sizeof(float));
	float *h_affWeighted = (float *)malloc((NDIM)* sizeof(float));
	bool regMode3OffTrigger = false; // registration trigger for regMode 3
	float shiftX, shiftY, shiftZ;
	int inputTmx = 0;
	int regMethod = 7;
	int subBgTrigger = 1;
	float *regRecords = (float *)malloc(11 * sizeof(float));
	float *deconRecords = (float *)malloc(10 * sizeof(float));
	int regStatus;
	bool mStatus;
	// ***read PSFs ***
	readtifstack(h_psf1, filePSF1, &psfSize[0]);
	readtifstack(h_psf2, filePSF2, &psfSize[0]);
	if (flagUnmatch){
		readtifstack(h_psf_bp1, filePSF_bp1, &psfSize[0]);
		readtifstack(h_psf_bp2, filePSF_bp2, &psfSize[0]);
	}

	// ** variables for max projections **
	float *h_MP; unsigned int imSizeMP[6]; int projectNum = 36;
	if ((flagSaveInterFiles[6] == 1) || (flagSaveInterFiles[7] == 1)){ // 3D max projections
		int imRotationx = round(sqrt(imx*imx + imz*imz));
		int imRotationy = round(sqrt(imy*imy + imz*imz));
		int totalSizeMP3D = imx * imRotationy * projectNum + imRotationx * imy * projectNum;
		h_MP = (float *)malloc(totalSizeMP3D * sizeof(float));
	}
	else if ((flagSaveInterFiles[3] == 1) || (flagSaveInterFiles[4] == 1) || (flagSaveInterFiles[5] == 1)){ // 2D max projections
		int totalSizeMP2D = imx*imy + imy*imz + imz*imx;
		h_MP = (float *)malloc(totalSizeMP2D  * sizeof(float));
	}
	// ******processing in batch*************
	for (imgNum = imgNumStart; imgNum <= imgNumEnd; imgNum += imgNumInterval){
		if (regMode == 0){ // no registration
			regMethod = 0;
		}
		else if (regMode == 1){//in regMode 1, use Test number for registratiion
			imgNum = imgNumTest;
		}
		printf("\n***Image time point number: %d \n", imgNum);
		f1 = fopen(fileLog, "a");
		fprintf(f1, "\n***Image time point number: %d \n", imgNum);
		fclose(f1);
		sprintf(imgNumStr, "%d", imgNum);
		char *fileStackA, *fileStackB, *fileRegA, *fileRegB, *fileTmx,
			*fileDecon, *fileDeconMP_XY, *fileDeconMP_YZ, *fileDeconMP_ZX, *fileDeconMP_3D_X, *fileDeconMP_3D_Y;
		for (int iColor = 0; iColor < subFolderCount; iColor++){
			start = clock();
		if (flagMultiColor){
			printf("\n Processing time point %d color %d: %s \n", imgNum, iColor + 1, &subFolderNames[iColor][0]);
			f1 = fopen(fileLog, "a");
			fprintf(f1, "\n Processing time point %d color %d: %s \n", imgNum, iColor + 1, &subFolderNames[iColor][0]);
			fclose(f1);
			outFolder = concat(3, outMainFolder, &subFolderNames[iColor][0], "/");
			inFolder1 = concat(3, mainFolder, &subFolderNames[iColor][0], "/SPIMA/");
			inFolder2 = concat(3, mainFolder, &subFolderNames[iColor][0], "/SPIMB/");
			deconFolder = concat(2, outFolder, "Decon/");
			tmxFolder = concat(2, outFolder, "TMX/");
			regFolder1 = concat(2, outFolder, "RegA/");
			regFolder2 = concat(2, outFolder, "RegB/");
			deconFolderMP_XY = concat(2, deconFolder, "MP_ZProj/");
			deconFolderMP_YZ = concat(2, deconFolder, "MP_XProj/");
			deconFolderMP_ZX = concat(2, deconFolder, "MP_YProj/");
			deconFolderMP_3D_X = concat(2, deconFolder, "MP_3D_Xaxis/");
			deconFolderMP_3D_Y = concat(2, deconFolder, "MP_3D_Yaxis/");
		}
		fileStackA = concat(4, inFolder1, fileNamePrefix1, imgNumStr, ".tif");
		fileStackB = concat(4, inFolder2, fileNamePrefix2, imgNumStr, ".tif");
		fileRegA = concat(5, regFolder1, fileNamePrefix1, "reg_", imgNumStr, ".tif");
		fileRegB = concat(5, regFolder2, fileNamePrefix2, "reg_", imgNumStr, ".tif");
		fileTmx = concat(4, tmxFolder, "Matrix_", imgNumStr, ".tmx");
		///
		fileDecon = concat(4, deconFolder, "Decon_", imgNumStr, ".tif");
		fileDeconMP_XY = concat(4, deconFolderMP_XY, "MP_XY_", imgNumStr, ".tif");
		fileDeconMP_YZ = concat(4, deconFolderMP_YZ, "MP_YZ_", imgNumStr, ".tif");
		fileDeconMP_ZX = concat(4, deconFolderMP_ZX, "MP_ZX_", imgNumStr, ".tif");
		fileDeconMP_3D_X = concat(4, deconFolderMP_3D_X, "MP_3D_Xaxis_", imgNumStr, ".tif");
		fileDeconMP_3D_Y = concat(4, deconFolderMP_3D_Y, "MP_3D_Yaxis_", imgNumStr, ".tif");

		printf("...Registration...\n");
		printf("	...Initializing (rotation, interpolation, initial matrix)...\n");
		f1 = fopen(fileLog, "a");
		fprintf(f1, "...Registration...\n");
		fprintf(f1, "	...Initializing (rotation, interpolation, initial matrix)...\n");
		fclose(f1);
		// ****************Interpolation before registration**************** //////
		// ## check files
		if (!fexists(fileStackA)){
			printf("***File does not exist: %s", fileStackA);
			return 0;
		}
		if (!fexists(fileStackB)){
			printf("***File does not exist: %s", fileStackB);
			return 0;
		}
		cudaMalloc((void **)&d_img3D, totalSize12 *sizeof(float));
		if ((imRotation == 1) || (imRotation == -1))
			cudaMalloc((void **)&d_imgE, totalSizeIn1 * sizeof(float));
		if (flagInterp1)
			cudaMalloc3DArray(&d_Array1, &channelDesc, make_cudaExtent(imSizeIn1[0], imSizeIn1[1], imSizeIn1[2]));
		if (flagInterp2)
			cudaMalloc3DArray(&d_Array2, &channelDesc, make_cudaExtent(imSizeIn2[0], imSizeIn2[1], imSizeIn2[2]));

		///##image 1 or Stack A
		readtifstack(h_img1, fileStackA, &imSizeIn1[0]);
		if (flagInterp1){
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
		cudaDeviceSynchronize();

		//## image 2 or Stack B
		readtifstack(h_img2, fileStackB, &imSizeTemp[0]); //something wrong here at the 6th reading
		// rotation
		if ((imRotation == 1) || (imRotation == -1)){
			cudaMemcpy(d_imgE, h_img2, totalSizeIn2 * sizeof(float), cudaMemcpyHostToDevice);
			rotbyyaxis(d_img3D, d_imgE, imSizeIn2[0], imSizeIn2[1], imSizeIn2[2], imRotation);
			cudaMemcpy(h_StackB, d_img3D, totalSizeIn2 * sizeof(float), cudaMemcpyDeviceToHost);
			cudaFree(d_imgE);
		}
		if (flagInterp2){
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
		cudaDeviceSynchronize();

		// ***** Do 2D registration or alignment
		// only if 3D registration is set and flagInitialTmx = 2
		if ((regMode > 0) && (flagInitialTmx == 3)){
			float *tmx1 = (float *)malloc(6 * sizeof(float));
			float *tmx2 = (float *)malloc(6 * sizeof(float));
			int img2DSizeMax1 = ((imx1 * imy1) > (imz1 * imx1)) ? (imx1 * imy1) : (imz1 * imx1);
			int img2DSizeMax2 = ((imx2 * imy2) > (imz2 * imx2)) ? (imx2 * imy2) : (imz2 * imx2);
			int img2DSizeMax = (img2DSizeMax1 > img2DSizeMax2) ? img2DSizeMax1 : img2DSizeMax2;
			float *h_img2D1 = (float *)malloc(img2DSizeMax1 * sizeof(float));
			float *h_img2D2 = (float *)malloc(img2DSizeMax2 * sizeof(float));
			float *h_img2Dreg = (float *)malloc(img2DSizeMax1 * sizeof(float));
			float *regRecords2D = (float *)malloc(11 * sizeof(float));
			float *d_img2DMax = NULL;
			cudaMalloc((void **)&d_img2DMax, img2DSizeMax*sizeof(float));
			shiftX = (imx2 - imx1) / 2, shiftY = (imy2 - imy1) / 2, shiftZ = (imz2 - imz1) / 2;
			int flag2Dreg = 1;
			switch (flag2Dreg){
			case 1:
				tmx1[0] = 1; tmx1[1] = 0; tmx1[2] = shiftX;
				tmx1[3] = 0; tmx1[4] = 1; tmx1[5] = shiftY;
				cudaMemcpy(d_img3D, h_StackA, totalSize1 * sizeof(float), cudaMemcpyHostToDevice);
				maxprojection(d_img2DMax, d_img3D, imx1, imy1, imz1, 1);
				cudaMemcpy(h_img2D1, d_img2DMax, imx1 * imy1 * sizeof(float), cudaMemcpyDeviceToHost);
				cudaMemcpy(d_img3D, h_StackB, totalSize2 * sizeof(float), cudaMemcpyHostToDevice);
				maxprojection(d_img2DMax, d_img3D, imx2, imy2, imz2, 1);
				cudaMemcpy(h_img2D2, d_img2DMax, imx2 * imy2 * sizeof(float), cudaMemcpyDeviceToHost);
				(void)reg_2dshiftaligngpu(h_img2Dreg, tmx1, h_img2D1, h_img2D2, imx1, imy1, imx2, imy2,
					0, 0.3, 15, deviceNum, regRecords2D);
				shiftX = tmx1[2];
				shiftY = tmx1[5];

				tmx2[0] = 1; tmx2[1] = 0; tmx2[2] = shiftZ;
				tmx2[3] = 0; tmx2[4] = 1; tmx2[5] = shiftX;
				cudaMemcpy(d_img3D, h_StackA, totalSize1 * sizeof(float), cudaMemcpyHostToDevice);
				maxprojection(d_img2DMax, d_img3D, imx1, imy1, imz1, 2);
				cudaMemcpy(h_img2D1, d_img2DMax, imz1 * imx1 * sizeof(float), cudaMemcpyDeviceToHost);
				cudaMemcpy(d_img3D, h_StackB, totalSize2 * sizeof(float), cudaMemcpyHostToDevice);
				maxprojection(d_img2DMax, d_img3D, imx2, imy2, imz2, 2);
				cudaMemcpy(h_img2D2, d_img2DMax, imx2 * imz2 * sizeof(float), cudaMemcpyDeviceToHost);
				(void)reg_2dshiftalignXgpu(h_img2Dreg, tmx2, h_img2D1, h_img2D2, imz1, imx1, imz2, imx2,
					1, 0.3, 15, deviceNum, regRecords2D);
				shiftZ = tmx2[2];
				break;
			default:
				break;
			}
			for (int j = 0; j < NDIM; j++) iTmx[j] = 0;
			iTmx[0] = 1;
			iTmx[5] = 1;
			iTmx[10] = 1;
			iTmx[3] = shiftX;
			iTmx[7] = shiftY;
			iTmx[11] = shiftZ;
			printf("...shift translation, X: %f; Y: %f; Z: %f\n", shiftX, shiftY, shiftZ);
			if ((regMode == 1) || (regMode == 2)) flagInitialTmx = 1; // perform 2D reg only one time
			free(tmx1); free(tmx2); free(h_img2D1); free(h_img2D2); free(h_img2Dreg); free(regRecords2D);
		}
		cudaFree(d_img3D);

		/// ****** initialize matrix *******
		// ***** Do 3D registration ******
		// regMode--> 0: no registration; 1: one image only
		//			2: dependently, based on the results of last time point; 3: independently
		cudaMemGetInfo(&freeMem, &totalMem);
		printf("	...GPU free memory before registration is %.0f MBites\n", (float)freeMem / 1048576.0f);
		f1 = fopen(fileLog, "a");
		fprintf(f1, "	...GPU free memory before registration is %.0f MBites\n", (float)freeMem / 1048576.0f);
		fclose(f1);
		if (flagInitialTmx == 3)
			inputTmx = 1;
		else
			inputTmx = flagInitialTmx;
		if (inputTmx) memcpy(h_affInitial, iTmx, NDIM*sizeof(float));
		switch (regMode){
		case 0:
			regMethod = 0;
			regStatus = reg_3dgpu(h_reg, iTmx, h_StackA, h_StackB, imSize, imSize2, regMethod,
				inputTmx, FTOL, itLimit, subBgTrigger, deviceNum, regRecords);
			break;
		case 1:
			regStatus = reg_3dgpu(h_reg, iTmx, h_StackA, h_StackB, imSize, imSize2, regMethod,
				inputTmx, FTOL, itLimit, subBgTrigger, deviceNum, regRecords);
			mStatus = checkmatrix(iTmx);//if registration is good
			if (!mStatus){ // repeat with different initial matrix 
				regStatus = reg_3dgpu(h_reg, iTmx, h_StackA, h_StackB, imSize, imSize2, regMode,
					2, FTOL, itLimit, subBgTrigger, deviceNum, regRecords); // registration with phase translation 
			}
			imgNum = imgNumStart;
			regMode = 0; // Don't do more registraion for other time points
			flagInitialTmx = 1; // Apply matrix to all other time points
			continue;
			break;
		case 2:
			if ((imgNum != imgNumStart) || (iColor>0)){
				inputTmx = 1; // use previous matrix as input
				memcpy(iTmx, h_affWeighted, NDIM*sizeof(float));
			}
			regStatus = reg_3dgpu(h_reg, iTmx, h_StackA, h_StackB, imSize, imSize2, regMethod,
				inputTmx, FTOL, itLimit, subBgTrigger, deviceNum, regRecords);
			mStatus = checkmatrix(iTmx);//if registration is good
			if (!mStatus){ // repeat with different initial matrix 
				regStatus = reg_3dgpu(h_reg, iTmx, h_StackA, h_StackB, imSize, imSize2, regMode,
					2, FTOL, itLimit, subBgTrigger, deviceNum, regRecords); // registration with phase translation 
				mStatus = checkmatrix(iTmx);//if registration is good
				if (!mStatus){ // apply previous matrix
					memcpy(iTmx, h_affInitial, NDIM*sizeof(float)); // use input or previous matrix
					regStatus = reg_3dgpu(h_reg, iTmx, h_StackA, h_StackB, imSize, imSize2, 0,
						1, FTOL, itLimit, subBgTrigger, deviceNum, regRecords);
				}
			}
			if ((imgNum == imgNumStart) && (iColor==0)){
				memcpy(h_affWeighted, iTmx, NDIM*sizeof(float));
			}
			else{
				for (int j = 0; j < NDIM; j++){
					h_affWeighted[j] = 0.8*h_affWeighted[j] + 0.2*iTmx[j]; // weighted matrix for next time point
				}
			}
			break;
		case 3:
			if (inputTmx) memcpy(iTmx, h_affInitial, NDIM*sizeof(float));
			regStatus = reg_3dgpu(h_reg, iTmx, h_StackA, h_StackB, imSize, imSize2, 7,
				inputTmx, FTOL, itLimit, subBgTrigger, deviceNum, regRecords);
			mStatus = checkmatrix(iTmx);//check if registration is good
			if (!mStatus){ // apply previous matrix
				memcpy(iTmx, h_affInitial, NDIM*sizeof(float)); // use input or previous matrix
				regStatus = reg_3dgpu(h_reg, iTmx, h_StackA, h_StackB, imSize, imSize2, 0,
					inputTmx, FTOL, itLimit, subBgTrigger, deviceNum, regRecords);
			}
			break;
		default:
			;
		}
		//regRecords
		//[0] -[3]: initial GPU memory, after variables allocated, after processing, after variables released ( all in MB)
		//[4] -[6]: initial cost function value, minimized cost function value, intermediate cost function value
		//[7] -[10]: registration time (in s), whole time (in s), single sub iteration time (in ms), total sub iterations
		printf("	...initial cost function value: %f\n", regRecords[4]);
		printf("	...minimized cost function value: %f\n", regRecords[5]);
		printf("	...total sub-iteration number: %d\n", int(regRecords[10]));
		printf("	...each sub-iteration time cost: %2.3f ms\n", regRecords[9]);
		printf("	...all iteration time cost: %2.3f s\n", regRecords[7]);
		printf("	...registration time cost: %2.3f s\n", regRecords[7]);
		f1 = fopen(fileLog, "a");
		fprintf(f1, "	...initial cost function value: %f\n", regRecords[4]);
		fprintf(f1, "	...minimized cost function value: %f\n", regRecords[5]);
		//fprintf(f1, "	...total sub-iteration number: %d\n", int(regRecords[10]));
		//fprintf(f1, "	...each sub-iteration time cost: %2.3f ms\n", regRecords[9]);
		//fprintf(f1, "	...all iteration time cost: %2.3f s\n", regRecords[7]);
		fprintf(f1, "	...registration time cost: %2.3f s\n", regRecords[7]);
		fclose(f1);
		
		// always save transformation matrix
		f2 = fopen(fileTmx, "w");
		for (int j = 0; j < NDIM; j++)
		{
			fprintf(f2, "%f\t", iTmx[j]);
			if ((j + 1) % 4 == 0)
				fprintf(f2, "\n");
		}
		fprintf(f2, "%f\t%f\t%f\t%f\n", 0.0, 0.0, 0.0, 1.0);
		fclose(f2);
		memcpy(records, regRecords, 11 * sizeof(float));
		if (flagSaveInterFiles[1] == 1) writetifstack(fileRegA, h_StackA, &imSize[0], bitPerSample_input);//set bitPerSample as input images
		if (flagSaveInterFiles[2] == 1) writetifstack(fileRegB, h_reg, &imSize[0], bitPerSample_input);//set bitPerSample as input images

		// *************** Joint deconvolution ***********
		cudaMemGetInfo(&freeMem, &totalMem);
		printf("...Deconvolution ...\n");
		printf("	...GPU free memory before deconvolution is %.0f MBites\n", (float)freeMem / 1048576.0f);
		f1 = fopen(fileLog, "a");
		fprintf(f1, "...Deconvolution ...\n");
		fprintf(f1, "	...GPU free memory before deconvolution is %.0f MBites\n", (float)freeMem / 1048576.0f);
		fclose(f1);
		int deconStatus = decon_dualview(h_decon, h_StackA, h_reg, imSize, h_psf1, h_psf2,
			psfSize, itNumForDecon, deviceNum, gpuMemMode, deconRecords, flagUnmatch, h_psf_bp1, h_psf_bp2);
		memcpy(&records[11], deconRecords, 10 * sizeof(float));
		writetifstack(fileDecon, h_decon, &imSize[0], bitPerSample);//set bitPerSample as input images
		//deconRecords: 10 elements
		//[0]:  the actual memory mode used;
		//[1] -[5]: initial GPU memory, after variables partially allocated, during processing, after processing, after variables released ( all in MB);
		//[6] -[9]: initializing time, prepocessing time, decon time, total time;
		gpuMemMode = int(deconRecords[0]);
		switch (gpuMemMode){
			case 1:
				printf("	...Sufficient GPU memory, running in efficient mode !!!\n");
				break;
			case 2:
				printf("	...GPU memory optimized, running in memory saved mode !!!\n");
				break;
			case 3:
				printf("	...GPU memory further optimized, running in memory saved mode !!!\n");
				break;
			default:
				printf("	...Not enough GPU memory, no deconvolution performed !!!\n");
		}
		printf("	...GPU free memory during deconvolution is %.0f MBites\n", deconRecords[3]);
		printf("	...all iteration time cost: %2.3f s\n", deconRecords[8]);
		printf("	...deconvolution time cost: %2.3f s\n", deconRecords[9]);
		cudaMemGetInfo(&freeMem, &totalMem);
		printf("...GPU free memory after deconvolution is %.0f MBites\n", (float)freeMem / 1048576.0f);
		f1 = fopen(fileLog, "a");
		switch (gpuMemMode){
		case 1:
			fprintf(f1, "	...Sufficient GPU memory, running in efficient mode !!!\n");
			break;
		case 2:
			fprintf(f1, "	...GPU memory optimized, running in memory saved mode !!!\n");
			break;
		case 3:
			fprintf(f1, "	...GPU memory further optimized, running in memory saved mode !!!\n");
			break;
		default:
			fprintf(f1, "	...Not enough GPU memory, no deconvolution performed !!!\n");
		}
		//fprintf(f1, "	...all iteration time cost: %2.3f s\n", deconRecords[8]);
		fprintf(f1, "	...deconvolution time cost: %2.3f s\n", deconRecords[9]);
		fprintf(f1, "	...GPU free memory before deconvolution is %.0f MBites\n", (float)freeMem / 1048576.0f);
		fclose(f1);
		///********* save max projections
		if ((flagSaveInterFiles[3] == 1) || (flagSaveInterFiles[4] == 1) || (flagSaveInterFiles[5] == 1)){
			// 2D MP max projections
			(void)mp2Dgpu(h_MP, &imSizeMP[0], h_decon, &imSize[0], (bool)flagSaveInterFiles[3], (bool)flagSaveInterFiles[4], (bool)flagSaveInterFiles[5]);
			imSizeTemp[2] = 1;
			if (flagSaveInterFiles[3] == 1) {
				imSizeTemp[0] = imSizeMP[0]; imSizeTemp[1] = imSizeMP[1];
				writetifstack(fileDeconMP_XY, h_MP, &imSizeTemp[0], bitPerSample);
			}
			if (flagSaveInterFiles[4] == 1) {
				imSizeTemp[0] = imSizeMP[2]; imSizeTemp[1] = imSizeMP[3];
				writetifstack(fileDeconMP_YZ, &h_MP[imx*imy], &imSizeTemp[0], bitPerSample);
			}
			if (flagSaveInterFiles[5] == 1) {
				imSizeTemp[0] = imSizeMP[4]; imSizeTemp[1] = imSizeMP[5];
				writetifstack(fileDeconMP_ZX, &h_MP[imx*imy+imy*imz], &imSizeTemp[0], bitPerSample);
			}
		}
		if ((flagSaveInterFiles[6] == 1) || (flagSaveInterFiles[7] == 1)){ // 3D max projections
			(void)mp3Dgpu(h_MP, &imSizeMP[0], h_decon, &imSize[0], (bool)flagSaveInterFiles[6], (bool)flagSaveInterFiles[7], projectNum);
			if (flagSaveInterFiles[6] == 1) writetifstack(fileDeconMP_3D_X, h_MP, &imSizeMP[0], bitPerSample);
			if (flagSaveInterFiles[7] == 1) writetifstack(fileDeconMP_3D_Y, &h_MP[imSizeMP[0]*imSizeMP[1]*imSizeMP[2]], &imSizeMP[3], bitPerSample);
		}

		end = clock();
		records[21] = (float)(end - start) / CLOCKS_PER_SEC;

		// release file names
		if (flagMultiColor){
			free(outFolder); free(inFolder1); free(inFolder2); free(deconFolder); free(tmxFolder); free(regFolder1); //
			free(regFolder2);  free(deconFolderMP_XY); free(deconFolderMP_YZ); free(deconFolderMP_ZX);
			free(deconFolderMP_3D_X); free(deconFolderMP_3D_Y);
		}
		free(fileStackA); free(fileStackB); free(fileRegA); free(fileRegB); free(fileDecon); //
		free(fileTmx);  free(fileDeconMP_XY); free(fileDeconMP_YZ); free(fileDeconMP_ZX);
		free(fileDeconMP_3D_X); free(fileDeconMP_3D_Y);
		printf("...Time cost for current image is %2.3f s\n", records[21]);
		f1 = fopen(fileLog, "a");
		fprintf(f1, "...Time cost for current image is %2.3f s\n", records[21]);
		fclose(f1);
		}
	}

	////release CPU memory 
	free(h_affInitial);
	free(h_affWeighted);
	free(regRecords);
	free(deconRecords);
	free(h_aff12);
	free(h_StackA);
	free(h_StackB);
	free(h_psf1);
	free(h_psf2);
	free(h_psf_bp1);
	free(h_psf_bp2);
	free(h_reg);
	free(h_decon);
	if ((flagSaveInterFiles[3] == 1) || (flagSaveInterFiles[4] == 1) || (flagSaveInterFiles[5] == 1) || (flagSaveInterFiles[6] == 1) || (flagSaveInterFiles[7] == 1))
		free(h_MP);
	cudaMemGetInfo(&freeMem, &totalMem);
	printf("\nGPU free memory after whole processing is %.0f MBites\n", (float)freeMem / 1048576.0f);
	endWhole = clock();
	printf("Total time cost for whole processing is %2.3f s\n", (float)(endWhole - startWhole) / CLOCKS_PER_SEC);
	f1 = fopen(fileLog, "a");
	fprintf(f1, "\nGPU free memory after whole processing is %.0f MBites\n", (float)freeMem / 1048576.0f);
	fprintf(f1, "Total time cost for whole processing is %2.3f s\n", (float)(endWhole - startWhole) / CLOCKS_PER_SEC);
	fclose(f1);
	return 0;
}
#undef NRANSI
