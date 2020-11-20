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
#else
#include <sys/stat.h>
#endif

// Includes CUDA
#include <cuda_runtime.h>
#include <memory>
#include "device_launch_parameters.h"

#include "tiff.h"
#include "tiffio.h"
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



cudaError_t __err;
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
	if (!fexists(tifdir)) {
		fprintf(stderr, "*** File does not exist: %s\n", tifdir);
		exit(1);
	}
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

// Read and write tiff image

void readtifstack(float *h_Image, char tifdir[], unsigned int *imsize){
	// check if file exists
	if (!fexists(tifdir)) {
		fprintf(stderr, "*** Failed to read image!!! File does not exist: %s\n", tifdir);
		exit(1);
	}

	// get TIFF image information
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
	// check if file exists
	if (!fexists(tifdir)) {
		fprintf(stderr, "*** Failed to read image!!! File does not exist: %s\n", tifdir);
		exit(1);
	}

	// get TIFF image information
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
void writetifstack(char tifdir[], float *h_Image, unsigned int *imsize, unsigned short bitPerSample) {
	int imTotalSize = imsize[0] * imsize[1] * imsize[2];
	uint32 imxy = imsize[0] * imsize[1];
	uint32 nByte = (uint32)(bitPerSample / 8);

	TIFF *tif;
	if (!(tif = TIFFOpen(tifdir, "w"))) { // check file opening
		fprintf(stderr, "*** Failed to create file!!! Please check the directory: %s\n", tifdir);
		exit(1);
	}
	if (bitPerSample == 16) {
		uint16 *buf = (uint16 *)_TIFFmalloc(imTotalSize * sizeof(uint16));
		for (int i = 0; i < imTotalSize; i++) {
			buf[i] = (uint16)h_Image[i];
		}

		
		
		for (uint32 n = 0; n < imsize[2]; n++) {
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
		_TIFFfree(buf);
	}
	else if (bitPerSample == 32) {
		for (uint32 n = 0; n < imsize[2]; n++) {

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
	}
	else
		printf("Image bit per sample is not supported, please set bitPerPample to 16 or 32 !!!\n\n");

	(void)TIFFClose(tif);
}


void writetifstack_16to16(char tifdir[], unsigned short *h_Image, unsigned int *imsize) {
	int imTotalSize = imsize[0] * imsize[1] * imsize[2];
	uint32 imxy = imsize[0] * imsize[1];
	uint32 nByte = (uint32)(16 / 8);
	TIFF *tif;
	if (!(tif = TIFFOpen(tifdir, "w"))) { // check file opening
		fprintf(stderr, "*** Failed to create file!!! Please check the directory: %s\n", tifdir);
		exit(1);
	}
	for (uint32 n = 0; n < imsize[2]; n++) {
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
	//printf("Reseting GPU devices....\n");
	//cudaDeviceReset();
	//printf("...Reseting Done!!!\n");
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

//// 3D image operations
int alignsize3d(float *h_odata, float *h_idata, long long int sx, long long int sy, long long int sz, long long int sx2, long long int sy2, long long int sz2, int gpuMemMode) {
	int runStatus = 0;
	float *d_img1=NULL, *d_img2 = NULL;
	switch (gpuMemMode) {
	case 0:
		alignsize3Dcpu(h_odata, h_idata, sx, sy, sz, sx2, sy2, sz2);
		break;
	case 1:
		cudaMalloc((void **)&d_img1, sx*sy*sz * sizeof(float));
		cudaMalloc((void **)&d_img2, sx2*sy2*sz2 * sizeof(float));
		cudaMemcpy(d_img2, h_idata, sx2*sy2*sz2 * sizeof(float), cudaMemcpyHostToDevice);
		alignsize3Dgpu(d_img1, d_img2, sx, sy, sz, sx2, sy2, sz2);
		cudaMemcpy(h_odata, d_img1, sx*sy*sz * sizeof(float), cudaMemcpyDeviceToHost);
		cudaFree(d_img1);
		cudaFree(d_img2);
		break;
	case 2:
		cudaMalloc((void **)&d_img1, sx*sy*sz * sizeof(float));
		cudaMalloc((void **)&d_img2, sx2*sy2*sz2 * sizeof(float));
		cudaMemcpy(d_img2, h_idata, sx2*sy2*sz2 * sizeof(float), cudaMemcpyHostToDevice);
		alignsize3Dgpu(d_img1, d_img2, sx, sy, sz, sx2, sy2, sz2);
		cudaMemcpy(h_odata, d_img1, sx*sy*sz * sizeof(float), cudaMemcpyDeviceToHost);
		cudaFree(d_img1);
		cudaFree(d_img2);
		break;
	default:
		printf("\n****Wrong gpuMemMode setup, processing stopped !!! ****\n");
		return 1;
	}
	return runStatus;
}

int imresize3d(float *h_odata, float *h_idata, long long int sx, long long int sy, long long int sz, 
	long long int sx2, long long int sy2, long long int sz2, int deviceNum) {

	float *iTmx = (float *)malloc(12 * sizeof(float));
	for (int j = 0; j < 12; j++)
	{
		iTmx[j] = 0;
	}
	iTmx[0] = float(sx2) / float(sx);
	iTmx[5] = float(sy2) / float(sy);
	iTmx[10] = float(sz2) / float(sz);
	unsigned int imSize1[3], imSize2[3];
	imSize1[0] = sx, imSize1[1] = sy, imSize1[2] = sz;
	imSize2[0] = sx2, imSize2[1] = sy2, imSize2[2] = sz2;
	(void)atrans3dgpu(h_odata, iTmx, h_idata, &imSize1[0], &imSize2[0], deviceNum);
	free(iTmx);
	return 0;
}

int imoperation3D(float *h_odata, unsigned int *sizeOut, float *h_idata, unsigned int *sizeIn, int opChoice, int deviceNum) {
	//rot direction
		// 1: rotate 90 deg around Y axis
		//-1: rotate -90 deg around Y axis
	long long int sx = sizeIn[0], sy = sizeIn[1], sz = sizeIn[2];
	long long int totalSize = sx * sy * sz;
	float *d_img1 = NULL, *d_img2 = NULL;
	switch (opChoice) {
	case 0:
		break;
	case 1: // // rotate 90 deg around Y axis
		cudaMalloc((void **)&d_img1, totalSize * sizeof(float));
		cudaMalloc((void **)&d_img2, totalSize * sizeof(float));
		cudaMemcpy(d_img2, h_idata, totalSize * sizeof(float), cudaMemcpyHostToDevice);
		rotbyyaxis(d_img1, d_img2, sx, sy, sz, 1);
		cudaMemcpy(h_odata, d_img1, totalSize * sizeof(float), cudaMemcpyDeviceToHost);
		cudaFree(d_img1);
		cudaFree(d_img2);
		sizeOut[0] = sizeIn[2], sizeOut[1] = sizeIn[1], sizeOut[2] = sizeIn[0];
		break;
	case 2: //-1: rotate -90 deg around Y axis
		cudaMalloc((void **)&d_img1, totalSize * sizeof(float));
		cudaMalloc((void **)&d_img2, totalSize * sizeof(float));
		cudaMemcpy(d_img2, h_idata, totalSize * sizeof(float), cudaMemcpyHostToDevice);
		rotbyyaxis(d_img1, d_img2, sx, sy, sz, -1);
		cudaMemcpy(h_odata, d_img1, totalSize * sizeof(float), cudaMemcpyDeviceToHost);
		cudaFree(d_img1);
		cudaFree(d_img2);
		sizeOut[0] = sizeIn[2], sizeOut[1] = sizeIn[1], sizeOut[2] = sizeIn[0];
		break;
	default:
		printf("\n*** Wrong operation choice !!! **** \n");
		return 1;
	}	
	return 0;
}

int mp2dgpu(float *h_MP, unsigned int *sizeMP, float *h_img, unsigned int *sizeImg, bool flagZProj, bool flagXProj, bool flagYProj){
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

int mp3dgpu(float *h_MP, unsigned int *sizeMP, float *h_img, unsigned int *sizeImg, bool flagXaxis, bool flagYaxis, int projectNum){
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
			cudaThreadSynchronize();
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
			cudaThreadSynchronize();
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

int mip3dgpu(float *h_MP, unsigned int *sizeMP, float *h_img, unsigned int *sizeImg, int rAxis, long long int projectNum) {
	// bool flagXaxis, bool flagYaxis
	// if rAxis == 1: X axis; if rAxis==2: Y axis
	//sizeMP: sx, imRotationy, projectNum, imRotationx, sy, projectNum
	long long int sx = sizeImg[0], sy = sizeImg[1], sz = sizeImg[2];
	long long int sr = 1, imRotation = 1, totalSizeProject = 1;
	if (rAxis == 1) {
		sr = sx;
		imRotation = round(sqrt(sy*sy + sz*sz));	
	}
	else if (rAxis == 2) {
		sr = sy;
		imRotation = round(sqrt(sx*sx + sz*sz));
	}
	else
		return -1;
	totalSizeProject = sr * imRotation;
	long long int totalSizeRotation = sr * imRotation * imRotation;
	//long long int totalSizeProjectStack = sr * imRotation * (long long)projectNum;
	
	float projectAng = 0;
	float projectStep = 3.14159 * 2 / (float)projectNum;
	float *h_affRot = (float *)malloc(NDIM * sizeof(float));
	float *d_StackProject, *d_StackRotation;
	
	cudaMalloc((void **)&d_StackRotation, totalSizeRotation * sizeof(float));
	cudaMalloc((void **)&d_StackProject, totalSizeProject * sizeof(float));
	cudaChannelFormatDesc channelDescT = cudaCreateChannelDesc<float>();
	cudaArray *d_Array;
	cudaMalloc3DArray(&d_Array, &channelDescT, make_cudaExtent(sx, sy, sz));
	cudacopyhosttoarray(d_Array, channelDescT, h_img, sx, sy, sz);
	BindTexture(d_Array, channelDescT);
	cudaCheckErrors("Texture create fail");
	if (rAxis == 1) {// 3D projection by X axis
		for (int iProj = 0; iProj < projectNum; iProj++) {
			projectAng = projectStep * iProj;
			rot2matrix(h_affRot, projectAng, sx, sy, sz, 1);
			//rot3Dbyyaxis(h_aff_temp, projectAng, imx, imz, imRotationx, imRotationx);
			CopyTranMatrix(h_affRot, NDIM * sizeof(float));
			affineTransform(d_StackRotation, sr, imRotation, imRotation, sx, sy, sz);
			maxprojection(d_StackProject, d_StackRotation, sr, imRotation, imRotation, 1);
			cudaMemcpy(&h_MP[totalSizeProject*iProj], d_StackProject, totalSizeProject * sizeof(float), cudaMemcpyDeviceToHost);
			cudaThreadSynchronize();
		}
		sizeMP[0] = sr; sizeMP[1] = imRotation; sizeMP[2] = projectNum;
	}

	else if (rAxis == 2) {// 3D projection by Y axis
		// 3D projection by Y axis
		for (int iProj = 0; iProj < projectNum; iProj++) {
			projectAng = projectStep * iProj;
			rot2matrix(h_affRot, projectAng, sx, sy, sz, 2);
			//rot3Dbyyaxis(h_aff_temp, projectAng, imx, imz, imRotationx, imRotationx);
			CopyTranMatrix(h_affRot, NDIM * sizeof(float));
			affineTransform(d_StackRotation, imRotation, sr, imRotation, sx, sy, sz);
			maxprojection(d_StackProject, d_StackRotation, imRotation, sr, imRotation, 1);
			cudaMemcpy(&h_MP[totalSizeProject*iProj], d_StackProject, totalSizeProject * sizeof(float), cudaMemcpyDeviceToHost);
			cudaThreadSynchronize();
		}
		sizeMP[0] = imRotation; sizeMP[1] = sr; sizeMP[2] = projectNum;
	}
	UnbindTexture();
	free(h_affRot);
	cudaFree(d_StackRotation);
	cudaFree(d_StackProject);
	cudaFreeArray(d_Array);

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
