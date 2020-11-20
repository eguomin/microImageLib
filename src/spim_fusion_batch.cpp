#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>       
#include <iostream>
#include <fstream>
#include <string.h>
#include <math.h>
#include <ctime>
#include <time.h>
#include <math.h>

#ifdef _WIN32
#include <windows.h>
#include <tchar.h>
#include <strsafe.h>
#else
#define MAX_PATH 4096
#include <sys/stat.h>
#endif

extern "C" {
#include "libapi.h"
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
	if (hFile != INVALID_HANDLE_VALUE) {
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
#else
int findSubFolders(char *subFolderNames, char *pathIn)
{
	// modify here
	fprintf(stderr, "*** Multi-color processing is currently not supported on Linux\n");

	return -1;
}
#endif

#define NDIM 12

void helpmessage(char *appName, bool flagHelp) {
	printf("\n%s: Dual-view fusion (registration and joint deconvolution) for diSPIM images in batch mode\n", appName);
	printf("\nUsage:\t%s [OPTIONS: 34 or 36 manatary arguments]\n", appName);
	if (!flagHelp) {
		printf("\nUse command for more details:\n\t%s -help or %s -h\n", appName, appName);
		return;
	}
	printf("\tOnly 16-bit or 32-bit standard TIFF images are currently supported.\n");
	printf("\n= = = [Mandatory arguments exactly ordered as following] = = = = = = = = = = = = = = = = = = = = = = = = = = = =\n");
	printf("\t 1: <path>\t\tOutput directory\n");
	printf("\t 2: <path>\t\tInput image 1 (SPIM A) directory\n");
	printf("\t 3: <path>\t\tInput image 2 (SPIM B) directory\n");
	printf("\t 4: <string>\t\tInput image 1 base name\n");
	printf("\t 5: <string>\t\tInput image 2 base name\n");
	printf("\t 6: <int>\t\tInput image index - start\n");
	printf("\t 7: <int>\t\tInput image index - end\n");
	printf("\t 8: <int>\t\tInput image index - interval\n");
	printf("\t 9: <int>\t\tInput image index - test (used when argument 16 is set as 1)\n");
	printf("\t10: <float>\t\tPixel Size X 1 (um)\n");
	printf("\t11: <float>\t\tPixel Size Y 1 (um)\n");
	printf("\t12: <float>\t\tPixel Size Z 1 (um)\n");
	printf("\t13: <float>\t\tPixel Size X 2 (um)\n");
	printf("\t14: <float>\t\tPixel Size Y 2 (um)\n");
	printf("\t15: <float>\t\tPixel Size Z 2 (um)\n");
	printf("\t16: <int>\t\tRegistration mode\n");
	printf("\t\t\t\t0: no registration, but transform image 2 based on input matrix\n");
	printf("\t\t\t\t1: one image only, use test image to generate registration for all other images\n");
	printf("\t\t\t\t2: dependently, based on the results of last image\n");
	printf("\t\t\t\t3: independently, do registration for each image independently\n");
	printf("\t17: <int>\t\tImage 2 (SPIM B) rotation:\n");
	printf("\t\t\t\t0, no rotation; 1: 90 deg by Y-axis; -1: -90 deg by Y-axis \n");
	printf("\t18: <int>\t\tSet initial matrix\n");
	printf("\t\t\t\t0: default matrix (identity matrix)\n");
	printf("\t\t\t\t1: specify input matrix file\n");
	printf("\t\t\t\t2: based on 3D phase translation\n");
	printf("\t\t\t\t3: based on 2D max projection image registration\n");
	printf("\t19: <filename>\t\tInput matrix file name, use any string if arguement 18 is not set as 1\n");
	printf("\t20: <float>\t\tTolerance or threshold of the stop point (typically 0.001~0.00001)\n");
	printf("\t21: <int>\t\tMaximum iteration number for registration(typically 2000~5000)\n");
	printf("\t22: <int>\t\tSave registered image 1 (0: no; 1: yes)\n");
	printf("\t23: <int>\t\tSave registered image 2 (0: no; 1: yes)\n");
	printf("\t24: <filename>\t\tPSF1 (forward projector 1) image filename\n");
	printf("\t25: <filename>\t\tPSF2 (forward projector 2) image filename\n");
	printf("\t26: <int>\t\tIteration number for deconvolution (typically 10~20)\n");
	printf("\t27: <int>\t\tSave max projection of decon image: X projection (0: no; 1: yes)\n");
	printf("\t28: <int>\t\tSave max projection of decon image: Y Projection (0: no; 1: yes)\n");
	printf("\t29: <int>\t\tSave max projection of decon image: Z Projection (0: no; 1: yes)\n");
	printf("\t30: <int>\t\tSave 3D max projection of decon image: X-axis (0: no; 1: yes)\n");
	printf("\t31: <int>\t\tSave 3D max projection of decon image: Y-axis (0: no; 1: yes)\n");
	printf("\t32: <int>\t\tBit of output images (16 or 32)\n");
	printf("\t33: <int>\t\tQuery GPU information before processing (0: no; 1: yes)\n");
	printf("\t34: <int>\t\tSpecify the GPU device if multiple GPUs (1st GPU indexed as 0)\n");
	printf("\t35: <filename>\t(optional)Backward projector 1 filename (if not set, use flip of PSF1)\n");
	printf("\t36: <filename>\t(optional)Backward projector 2 filename (if not set, use flip of PSF2)\n");
	printf("\nArguments are set same with the ImageJ diSPIMFusion plugin.\n");
	return;
}

int main(int argc, char* argv[])
{
	if (argc == 1) {
		helpmessage(argv[0], false);
		return EXIT_SUCCESS;
	}
	else if (argc == 2) {
		if (strcmp(argv[1], "-help") == 0 || strcmp(argv[1], "-h") == 0)
		{
			helpmessage(argv[0], true);
			return EXIT_SUCCESS;
		}

		else {
			helpmessage(argv[0], false);
			return EXIT_SUCCESS;
		}
	}
	else if ((argc != 35) && (argc != 37)) {
		printf("Arguments do NOT match! Please input exactly 34 or 36 arguments...\n");
		printf("For more information, use option -help or -h.\n");
		return 0;
	}
	char *outMainFolder = argv[1]; // Output folder
	char *folder1 = argv[2];
	char *folder2 = argv[3];
	char *fileNamePrefix1 = argv[4];
	char *fileNamePrefix2 = argv[5];
	int imgNumStart = atoi(argv[6]);
	int imgNumEnd = atoi(argv[7]);
	int imgNumInterval = atoi(argv[8]);
	int imgNumTest = atoi(argv[9]);
	float pixelSizex1 = (float)atof(argv[10]);
	float pixelSizey1 = (float)atof(argv[11]);
	float pixelSizez1 = (float)atof(argv[12]);
	float pixelSizex2 = (float)atof(argv[13]);
	float pixelSizey2 = (float)atof(argv[14]);
	float pixelSizez2 = (float)atof(argv[15]);

	//registration
	int regMode = atoi(argv[16]); // regMode--> 0: no registration; 1: one image only
								  //			2: dependently, based on the results of last time point; 3: independently
	int imRotation = atoi(argv[17]); //0: no rotation; 1: 90deg rotation ; -1: -90deg rotation ;
	int flagInitialTmx = atoi(argv[18]); //initial matrix --> 0: default matrix; 1: input matrix; 2: based on 2D registration
	char *fileiTmx = argv[19]; // input matrix file
	float FTOL = (float)atof(argv[20]); //1.0e-3, threshold for convergence of registration
	int itLimit = atoi(argv[21]);
	//bool savePre1 = (bool)(atoi(argv[22])); //0: do NOT save pre reg images; 1: save pre reg images
	//bool savePre2 = (bool)(atoi(argv[23])); //0: do NOT save pre reg images; 1: save pre reg images
	bool saveReg1 = (bool)(atoi(argv[22])); //0: do NOT save reg images; 1: save reg images
	bool saveReg2 = (bool)(atoi(argv[23])); //0: do NOT save reg images; 1: save reg images

											// deconvolution
	char *filePSF1 = argv[24];
	char *filePSF2 = argv[25];
	int itNumForDecon = atoi(argv[26]); //total iteration number for deconvolution
	bool saveXProj = (bool)(atoi(argv[27])); //
	bool saveYProj = (bool)(atoi(argv[28])); //
	bool saveZProj = (bool)(atoi(argv[29])); //
	bool saveXaxisProj = (bool)(atoi(argv[30])); //
	bool saveYaxisProj = (bool)(atoi(argv[31])); //

	unsigned short bitPerSample = (unsigned short)(atoi(argv[32]));
	//GPU Options
	bool dQuery = (bool)(atoi(argv[33]));
	int deviceNum = atoi(argv[34]); // set which gpu if there are multiple GPUS on the computer
	//input for unmatched backprojector
	bool flagUnmatch = false;
	char *filePSF1_bp = "Balabala", *filePSF2_bp = "Balabala";
	if (argc == 37) {
		flagUnmatch = true;
		filePSF1_bp = argv[35];
		filePSF2_bp = argv[36];
	}
	int flagSaveInterFiles[8];
	for (int i = 0; i < 8; i++) flagSaveInterFiles[i] = 0;
	if (saveReg1) flagSaveInterFiles[1] = 1;
	if (saveReg2) flagSaveInterFiles[2] = 1;
	if (saveZProj) flagSaveInterFiles[3] = 1;
	if (saveXProj) flagSaveInterFiles[4] = 1;
	if (saveYProj) flagSaveInterFiles[5] = 1;
	if (saveXaxisProj) flagSaveInterFiles[6] = 1;
	if (saveYaxisProj) flagSaveInterFiles[7] = 1;
	

	// print GPU devices information
	if (dQuery) {
		queryDevice();
	}

	printf("=====================================================\n");
	printf("=====================================================\n\n");
	printf("=== diSPIM Fusion settings ...\n");
	char *outFolder, *inFolder1, *inFolder2;
	// ***** check if multitple color processing
	char mainFolder[MAX_PATH];
	bool flagMultiColor = false;
	int multiColor = atoi(folder1);
	int subFolderCount = 1;
	char subFolderNames[20][MAX_PATH];
	if (multiColor == 1) { // trigger multiple color
#ifdef _WIN32 
		strcpy(mainFolder, folder2);
		flagMultiColor = true;
#else
		fprintf(stderr, "*** Multi-color processing is currently not supported on Linux\n");
#endif
	}
	if (flagMultiColor) {
		subFolderCount = findSubFolders(&subFolderNames[0][0], mainFolder);

		if (subFolderCount > 20)
			fprintf(stderr, "*** Number of subfolders: %d; two many subfolders\n", subFolderCount);
		else {
			printf("Procecing multicolor data: %d colors\n", subFolderCount);
			for (int j = 0; j < subFolderCount; j++)
				printf("...Subfolders %d: %s\n", j + 1, &subFolderNames[j][0]);
		}
		inFolder1 = concat(3, mainFolder, &subFolderNames[0][0], "/SPIMA/");
		inFolder2 = concat(3, mainFolder, &subFolderNames[0][0], "/SPIMB/");
	}
	else {
		inFolder1 = folder1;
		inFolder2 = folder2;
	}
	
	// ****************** Create output folders*****************//
	char *deconFolder, *tmxFolder, *regFolder1, *regFolder2,
		*deconFolderMP_XY, *deconFolderMP_YZ, *deconFolderMP_ZX, *deconFolderMP_3D_X, *deconFolderMP_3D_Y;
	// flagSaveInterFiles: 8 elements --> 1: save files; 0: not
	//					[0]: Intermediate outputs; [1]: reg A; [2]: reg B;
	//					[3]- [5]: Decon max projections Z, Y, X;
	//					[6], [7]: Decon 3D max projections: Y, X;
	if (flagMultiColor) {
#ifdef _WIN32 
		CreateDirectory(outMainFolder, NULL);
		for (int j = 0; j < subFolderCount; j++) {
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
	else {
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

	// Log file
	FILE *f1 = NULL, *f2 = NULL, *f3 = NULL;
	char *fileLog = concat(2, outMainFolder, "ProcessingLog.txt");

	// variables for memory and time cost records
	time_t startWhole, endWhole, now, time0, time1, time2, time3, time4;
	startWhole = clock();

	// ************ get basic input images and PSFs information ******************
	unsigned int imSize1In[3], imSize2In[3], imSize1[3], imSize2[3], psfSize[3], imSize[3], tempSize[3];
	long long int sx, sy, sz, sx1, sy1, sz1, sx2, sy2, sz2, sxpsf, sypsf, szpsf, sxtemp,sytemp,sztemp;
	int imgNum = imgNumStart;
	if (regMode == 1)
		imgNum = imgNumTest;
	char imgNumStr[20];
	sprintf(imgNumStr, "%d", imgNum);
	char *fileStack1 = concat(4, inFolder1, fileNamePrefix1, imgNumStr, ".tif"); // TIFF file to get image information
	char *fileStack2 = concat(4, inFolder2, fileNamePrefix2, imgNumStr, ".tif");
	// input image information and size
	unsigned int bitPerSampleImg = gettifinfo(fileStack1, &imSize1In[0]);
	bitPerSampleImg = gettifinfo(fileStack2, &imSize2In[0]);
	unsigned int bitPerSamplePSF = gettifinfo(filePSF1, &psfSize[0]);
	bitPerSamplePSF = gettifinfo(filePSF2, &tempSize[0]);
	if (bitPerSampleImg != 16 && bitPerSampleImg != 32) {
		fprintf(stderr, "***Input images are not supported, please use 16-bit or 32-bit image !!!\n");
		fprintf(stderr, "*** FAILED - ABORTING\n");
		exit(1);
	}
	// check PSF size
	if ((psfSize[0] != tempSize[0]) || (psfSize[1] != tempSize[1]) || (psfSize[2] != tempSize[2])) {
		printf("\tThe two forward projectors don't have the same image size, processing stopped !!!\n");
		return 1;
	}
	if (flagUnmatch) {
		bitPerSamplePSF = gettifinfo(filePSF1_bp, &tempSize[0]);
		if ((psfSize[0] != tempSize[0]) || (psfSize[1] != tempSize[1]) || (psfSize[2] != tempSize[2])) {
			printf("\tForward projector and backward projector don't have the same image size, processing stopped !!!\n");
			return 1;
		}
		bitPerSamplePSF = gettifinfo(filePSF2_bp, &tempSize[0]);
		if ((psfSize[0] != tempSize[0]) || (psfSize[1] != tempSize[1]) || (psfSize[2] != tempSize[2])) {
			printf("\tForward projector and backward projector don't have the same image size, processing stopped !!!\n");
			return 1;
		}
	}
	sxpsf = psfSize[0], sypsf = psfSize[1], szpsf = psfSize[2];
	// calculate image size
	sx = sx1 = imSize[0] = imSize1[0] = imSize1In[0];
	sy = sy1 = imSize[1] = imSize1[1] = round(float(imSize1In[1]) * pixelSizey1 / pixelSizex1);
	sz = sz1 = imSize[2] = imSize1[2] = round(float(imSize1In[2]) * pixelSizez1 / pixelSizex1);
	tempSize[0] = round(float(imSize2In[0]) * pixelSizex2 / pixelSizex1);
	tempSize[1] = round(float(imSize2In[1]) * pixelSizey2 / pixelSizex1);
	tempSize[2] = round(float(imSize2In[2]) * pixelSizez2 / pixelSizex1);

	// check image 2 rotations and adjust image size
	float pixelSizeTemp = 0;
	int opChoice = 0;
	switch (imRotation) {
	case 1:
		opChoice = 1; // 90 deg by Y-axis
		sx2 = imSize2[0] = tempSize[2];
		sy2 = imSize2[1] = tempSize[1];
		sz2 = imSize2[2] = tempSize[0];
		break;
	case -1:
		opChoice = 2; // -90 deg by Y-axis
		sx2 = imSize2[0] = tempSize[2];
		sy2 = imSize2[1] = tempSize[1];
		sz2 = imSize2[2] = tempSize[0];
		break;
	default:
		opChoice = 0; // no rotation
	}
	// image data size
	long long int totalSize = sx * sy * sz; // output size
	long long int totalSizeIn1 = (long long)imSize1In[0] * (long long)imSize1In[1] * (long long)imSize1In[2]; // input size
	long long int totalSizeIn2 = (long long)imSize2In[0] * (long long)imSize2In[1] * (long long)imSize2In[2];
	long long int totalSize1 = sx1 * sy1 * sz1; // after interpolation
	long long int totalSize2 = sx2 * sy2 * sz2;
	long long int totalSizePSF = sxpsf * sypsf * szpsf;

	// print images information
	printf("Image information:\n");
	printf("\tInput image 1 size %d x %d x %d\n  ", imSize1In[0], imSize1In[1], imSize1In[2]);
	printf("\t\t pixel size %.4f um x %.4f um x %.4f um\n", pixelSizex1, pixelSizey1, pixelSizez1);
	printf("\tInput image 2 size %d x %d x %d\n  ", imSize2In[0], imSize2In[1], imSize2In[2]);
	printf("\t\t pixel size %.4f um x %.4f um x %.4f um\n", pixelSizex2, pixelSizey2, pixelSizez2);
	printf("\tPSF image size %d x %d x %d\n  ", psfSize[0], psfSize[1], psfSize[2]);
	printf("\t\t pixel size %.4f um x %.4f um x %.4f um\n", pixelSizex1, pixelSizex1, pixelSizex1);
	printf("\tOutput image size %d x %d x %d\n  ", imSize[0], imSize[1], imSize[2]);
	printf("\t\t pixel size %.4f um x %.4f um x %.4f um\n", pixelSizex1, pixelSizex1, pixelSizex1);
	printf("\tImage number from %d to %d with step %d\n", imgNumStart, imgNumEnd, imgNumInterval);

	switch (regMode) {
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

	switch (imRotation) {
	case 0:
		printf("...No rotation on image B\n"); break;
	case 1:
		printf("...Rotate image B by 90 degree along Y axis\n"); break;
	case -1:
		printf("...Rotate image B by -90 degree along Y axis\n"); break;
	}

	switch (flagInitialTmx) {
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

	// ****Write information to log file***
	time(&now);
	f1 = fopen(fileLog, "w");
	// print images information
	fprintf(f1, "diSPIMFusion: %s\n", ctime(&now));
	if (flagMultiColor) {
		fprintf(f1, "Multicolor data: %d colors\n", subFolderCount);
		fprintf(f1, "...Input directory: %s\n", folder2);
		for (int j = 0; j < subFolderCount; j++)
			fprintf(f1, "     ...Subfolders %d: %s\n", j + 1, &subFolderNames[j][0]);
		fprintf(f1, "...Output directory: %s\n", outMainFolder);
	}
	else {
		fprintf(f1, "Single color data:\n");
		fprintf(f1, "...SPIMA input directory: %s\n", folder1);
		fprintf(f1, "...SPIMB input directory: %s\n", folder2);
		fprintf(f1, "...Output directory: %s\n", outMainFolder);
	}

	fprintf(f1, "\nImage information:\n");
	fprintf(f1, "\tInput image 1 size %d x %d x %d\n  ", imSize1In[0], imSize1In[1], imSize1In[2]);
	fprintf(f1, "\t\t pixel size %.4f um x %.4f um x %.4f um\n", pixelSizex1, pixelSizey1, pixelSizez1);
	fprintf(f1, "\tInput image 2 size %d x %d x %d\n  ", imSize2In[0], imSize2In[1], imSize2In[2]);
	fprintf(f1, "\t\t pixel size %.4f um x %.4f um x %.4f um\n", pixelSizex2, pixelSizey2, pixelSizez2);
	fprintf(f1, "\tPSF image size %d x %d x %d\n  ", psfSize[0], psfSize[1], psfSize[2]);
	fprintf(f1, "\t\t pixel size %.4f um x %.4f um x %.4f um\n", pixelSizex1, pixelSizex1, pixelSizex1);
	fprintf(f1, "\tOutput image size %d x %d x %d\n  ", imSize[0], imSize[1], imSize[2]);
	fprintf(f1, "\t\t pixel size %.4f um x %.4f um x %.4f um\n", pixelSizex1, pixelSizex1, pixelSizex1);
	fprintf(f1, "\tImage number from %d to %d with step %d\n", imgNumStart, imgNumEnd, imgNumInterval);
	switch (regMode) {
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

	switch (imRotation) {
	case 0:
		fprintf(f1, "...No rotation on image B\n"); break;
	case 1:
		fprintf(f1, "...Rotate image B by 90 degree along Y axis\n"); break;
	case -1:
		fprintf(f1, "...Rotate image B by -90 degree along Y axis\n"); break;
	}

	switch (flagInitialTmx) {
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

	// ****************** Start Processing ***************** //
	// image variables
	float *h_img1In = (float *)malloc(totalSizeIn1 * sizeof(float));
	float *h_img2In = (float *)malloc(totalSizeIn2 * sizeof(float));
	float *h_img1 = (float *)malloc(totalSize1 * sizeof(float));
	float *h_img2Temp = (float *)malloc(totalSizeIn2 * sizeof(float)); // rotation
	float *h_img2 = (float *)malloc(totalSize2 * sizeof(float));
	float *h_img2Reg = (float *)malloc(totalSize * sizeof(float));
	float *h_decon = (float *)malloc(totalSize * sizeof(float));
	float *h_psf1 = (float *)malloc(totalSizePSF * sizeof(float));
	float *h_psf2 = (float *)malloc(totalSizePSF * sizeof(float));
	float *h_psf1_bp = (float *)malloc(totalSizePSF * sizeof(float));
	float *h_psf2_bp = (float *)malloc(totalSizePSF * sizeof(float));
	float *regRecords = (float *)malloc(11 * sizeof(float));
	float *deconRecords = (float *)malloc(10 * sizeof(float));
	// ***read PSFs ***
	readtifstack(h_psf1, filePSF1, &psfSize[0]);
	readtifstack(h_psf2, filePSF2, &psfSize[0]);
	if (flagUnmatch) {
		readtifstack(h_psf1_bp, filePSF1_bp, &tempSize[0]);
		readtifstack(h_psf2_bp, filePSF2_bp, &tempSize[0]);
	}	

	// regMode--> 0: no registration; 1: one image only
	//			3: dependently, based on the results of last time point; 4: independently
	float *h_affInitial = (float *)malloc((NDIM) * sizeof(float));
	float *h_affWeighted = (float *)malloc((NDIM) * sizeof(float));
	bool regMode3OffTrigger = false; // registration trigger for regMode 3
	bool flagiTmx = true;
	int regChoice = 2;
	int affMethod = 6;
	bool mStatus;
	
	// matrix
	float *iTmx = (float *)malloc((NDIM + 4) * sizeof(float));
	switch (flagInitialTmx) {
	case 0:
		regChoice = 2;
		flagiTmx = false;
		break;
	case 1:
		regChoice = 2;
		flagiTmx = true;
		break;
	case 2:
		regChoice = 3;
		flagiTmx = false;
		break;
	case 3:
		regChoice = 4;
		flagiTmx = false;
		break;
	}

	if (flagiTmx) {// read input matrix
		FILE *f1 = fopen(fileiTmx, "r");
		for (int j = 0; j < NDIM; j++) fscanf(f1, "%f", &iTmx[j]);
		fclose(f1);
		iTmx[12] = 0, iTmx[13] = 0, iTmx[14] = 0, iTmx[15] = 1;
	}
	else { // Note: this is not the default matrix
		for (int i = 0; i < 16; i++) iTmx[i] = 0;
		iTmx[0] = 1, iTmx[5] = 1, iTmx[10] = 1, iTmx[15] = 1;
	}

	// ** variables for max projections **
	float *h_MP, *h_MP3D1, *h_MP3D2; unsigned int imSizeMP[6]; long long int projectNum = 36;
	if ((flagSaveInterFiles[3] == 1) || (flagSaveInterFiles[4] == 1) || (flagSaveInterFiles[5] == 1)) { // 2D max projections
		int totalSizeMP2D = sx*sy + sy*sz + sz*sx;
		h_MP = (float *)malloc(totalSizeMP2D * sizeof(float));
	}
	if (flagSaveInterFiles[6] == 1) { // 3D max projections: X-axis
		long long int imRotation = round(sqrt(sy*sy + sz * sz));
		h_MP3D1 = (float *)malloc(sx * imRotation * projectNum * sizeof(float));
	}
	if (flagSaveInterFiles[7] == 1) { // 3D max projections: Y-axis
		long long int imRotation = round(sqrt(sx*sx + sz * sz));
		h_MP3D2 = (float *)malloc(sy * imRotation * projectNum * sizeof(float));
	}
	FILE *fTmxIn = NULL, *fTmxOut = NULL;
	// ****** processing in batch *************
	int gpuMemMode = -1;
	bool verbose = true;
	bool flagConstInitial = false;
	for (imgNum = imgNumStart; imgNum <= imgNumEnd; imgNum += imgNumInterval) {
		if (regMode == 0) { // no registration
			regChoice = 0;
		}
		else if (regMode == 1) {//in regMode 1, use Test number for registratiion
			imgNum = imgNumTest;
		}
		printf("\n*** Image time point number: %d \n", imgNum);
		f1 = fopen(fileLog, "a");
		fprintf(f1, "\n*** Image time point number: %d \n", imgNum);
		fclose(f1);
		sprintf(imgNumStr, "%d", imgNum);
		char *fileImg1, *fileImg2, *fileReg1, *fileReg2, *fileTmx,
			*fileDecon, *fileDeconMP_XY, *fileDeconMP_YZ, *fileDeconMP_ZX, *fileDeconMP_3D_X, *fileDeconMP_3D_Y;
		for (int iColor = 0; iColor < subFolderCount; iColor++) {
			time0 = clock();
			if (flagMultiColor) {
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
			fileImg1 = concat(4, inFolder1, fileNamePrefix1, imgNumStr, ".tif");
			fileImg2 = concat(4, inFolder2, fileNamePrefix2, imgNumStr, ".tif");
			fileReg1 = concat(5, regFolder1, fileNamePrefix1, "reg_", imgNumStr, ".tif");
			fileReg2 = concat(5, regFolder2, fileNamePrefix2, "reg_", imgNumStr, ".tif");
			fileTmx = concat(4, tmxFolder, "Matrix_", imgNumStr, ".tmx");
			///
			fileDecon = concat(4, deconFolder, "Decon_", imgNumStr, ".tif");
			fileDeconMP_XY = concat(4, deconFolderMP_XY, "MP_XY_", imgNumStr, ".tif");
			fileDeconMP_YZ = concat(4, deconFolderMP_YZ, "MP_YZ_", imgNumStr, ".tif");
			fileDeconMP_ZX = concat(4, deconFolderMP_ZX, "MP_ZX_", imgNumStr, ".tif");
			fileDeconMP_3D_X = concat(4, deconFolderMP_3D_X, "MP_3D_Xaxis_", imgNumStr, ".tif");
			fileDeconMP_3D_Y = concat(4, deconFolderMP_3D_Y, "MP_3D_Yaxis_", imgNumStr, ".tif");

			printf("... Preprocessing ...\n");
			printf("\tInitializing and image reading ...\n");
			f1 = fopen(fileLog, "a");
			fprintf(f1, "...Registration...\n");
			fprintf(f1, "	...Initializing (rotation, interpolation, initial matrix)...\n");
			fclose(f1);
			readtifstack(h_img1In, fileImg1, &tempSize[0]);
			if ((imSize1In[0] != tempSize[0]) || (imSize1In[1] != tempSize[1]) || (imSize1In[2] != tempSize[2])) {
				printf("\t Input image 1 size does not match !!!\n");
				return 1;
			}
			readtifstack(h_img2In, fileImg2, &tempSize[0]);
			if ((imSize2In[0] != tempSize[0]) || (imSize2In[1] != tempSize[1]) || (imSize2In[2] != tempSize[2])) {
				printf("\t Input image 2 size does not match !!!\n");
				return 1;
			}
			// image 1: interpolation
			sxtemp = imSize1In[0], sytemp = imSize1In[1], sztemp = imSize1In[2];
			if ((sxtemp == sx1) && (sytemp == sy1) && (sztemp == sz1)) {
				memcpy(h_img1, h_img1In, totalSizeIn1 * sizeof(float));
			}
			else {
				printf("\tImage 1 interpolation ...\n");
				(void)imresize3d(h_img1, h_img1In, sx1, sy1, sz1, sxtemp, sytemp, sztemp, deviceNum);
			}

			// image 2: roation and interpolation
			if (opChoice == 0) {
				memcpy(h_img2Temp, h_img2In, totalSizeIn2 * sizeof(float));
				sxtemp = imSize2In[0];
				sytemp = imSize2In[1];
				sztemp = imSize2In[2];
			}
			else {
				printf("\tImage 2 rotation ...\n");
				(void)imoperation3D(h_img2Temp, &tempSize[0], h_img2In, &imSize2In[0], opChoice, deviceNum);
				sxtemp = tempSize[0];
				sytemp = tempSize[1];
				sztemp = tempSize[2];
			}
			//
			if ((sxtemp == sx2) && (sytemp == sy2) && (sztemp == sz2)) {
				memcpy(h_img2, h_img2Temp, totalSizeIn2 * sizeof(float));
			}
			else {
				printf("\tImage 2 interpolation ...\n");
				(void)imresize3d(h_img2, h_img2Temp, sx2, sy2, sz2, sxtemp, sytemp, sztemp, deviceNum);
			}
			time1 = clock();
			printf("\tTime cost for  preprocessing: %2.3f s\n", (float)(time1 - time0) / CLOCKS_PER_SEC);

			printf("...Registration...\n");
			memset(h_img2Reg, 0, totalSize1 * sizeof(float));
			if (flagiTmx) memcpy(h_affInitial, iTmx, NDIM * sizeof(float));
			switch (regMode) {
			case 0:
				(void)reg3d(h_img2Reg, iTmx, h_img1, h_img2, &imSize1[0], &imSize2[0], regChoice, affMethod,
					flagiTmx, FTOL, itLimit, deviceNum, gpuMemMode, verbose, regRecords);
				break;
			case 1:
				(void)reg3d(h_img2Reg, iTmx, h_img1, h_img2, &imSize1[0], &imSize2[0], regChoice, affMethod,
					flagiTmx, FTOL, itLimit, deviceNum, gpuMemMode, verbose, regRecords);
				mStatus = checkmatrix(iTmx, sx, sy, sz);//if registration is good
				if (!mStatus) { // repeat with different initial matrix 
					(void)reg3d(h_img2Reg, iTmx, h_img1, h_img2, &imSize1[0], &imSize2[0], regChoice, affMethod,
						false, FTOL, itLimit, deviceNum, gpuMemMode, verbose, regRecords);
				}
				imgNum = imgNumStart - 1; // set to start batch processing
				regMode = 0; // Don't do more registraion for other time points
				flagiTmx = 1; // Apply matrix to all other time points
				continue;
				break;
			case 2:
				if ((imgNum != imgNumStart) || (iColor > 0)) {
					flagiTmx = 1; // use previous matrix as input
					memcpy(iTmx, h_affWeighted, NDIM * sizeof(float));
				}
				(void)reg3d(h_img2Reg, iTmx, h_img1, h_img2, &imSize1[0], &imSize2[0], regChoice, affMethod,
					flagiTmx, FTOL, itLimit, deviceNum, gpuMemMode, verbose, regRecords);
				mStatus = checkmatrix(iTmx, sx, sy, sz);//if registration is good
				if (!mStatus) { // repeat with different initial matrix 
					(void)reg3d(h_img2Reg, iTmx, h_img1, h_img2, &imSize1[0], &imSize2[0], regChoice, affMethod,
						false, FTOL, itLimit, deviceNum, gpuMemMode, verbose, regRecords);
					mStatus = checkmatrix(iTmx, sx, sy, sz);//if registration is good
					if (!mStatus) { // apply previous matrix
						memcpy(iTmx, h_affInitial, NDIM * sizeof(float)); // use input or previous matrix
						(void)reg3d(h_img2Reg, iTmx, h_img1, h_img2, &imSize1[0], &imSize2[0], 0, affMethod,
							true, FTOL, itLimit, deviceNum, gpuMemMode, verbose, regRecords);
					}
				}
				if ((imgNum == imgNumStart) && (iColor == 0)) {
					memcpy(h_affWeighted, iTmx, NDIM * sizeof(float));
				}
				else {
					for (int j = 0; j < NDIM; j++) {
						h_affWeighted[j] = 0.8*h_affWeighted[j] + 0.2*iTmx[j]; // weighted matrix for next time point
					}
				}
				break;
			case 3:
				if (flagiTmx) memcpy(iTmx, h_affInitial, NDIM * sizeof(float));
				(void)reg3d(h_img2Reg, iTmx, h_img1, h_img2, &imSize1[0], &imSize2[0], regChoice, affMethod,
					flagiTmx, FTOL, itLimit, deviceNum, gpuMemMode, verbose, regRecords);
				mStatus = checkmatrix(iTmx, sx, sy, sz);//if registration is good
				if (!mStatus) { // repeat with different initial matrix 
					(void)reg3d(h_img2Reg, iTmx, h_img1, h_img2, &imSize1[0], &imSize2[0], 0, affMethod,
						false, FTOL, itLimit, deviceNum, gpuMemMode, verbose, regRecords);
				}
				break;
			default:
				;
			}
			// *** records: 11 element array
			//[0] : actual gpu memory mode
			//[1] - [3] : initial ZNCC(zero - normalized cross - correlation, negtive of the cost function), intermediate ZNCC, optimized ZNCC;
			//[4] - [7]: single sub iteration time(in ms), total number of sub iterations, iteralation time(in s), whole registration time(in s);
			//[8] - [10]: initial GPU memory, before registration, after processing(all in MB), if use gpu

			f1 = fopen(fileLog, "a");
			fprintf(f1, "\t... initial cost function value: %f\n", regRecords[1]);
			fprintf(f1, "\t... minimized cost function value: %f\n", regRecords[3]);
			//fprintf(f1, "\t... total sub-iteration number: %d\n", int(regRecords[5]));
			//fprintf(f1, "\t... each sub-iteration time cost: %2.3f ms\n", regRecords[4]);
			//fprintf(f1, "\t... all iteration time cost: %2.3f s\n", regRecords[6]);
			fprintf(f1, "\t... registration time cost: %2.3f s\n", regRecords[7]);
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
			time2 = clock();
			if (flagSaveInterFiles[1] == 1) writetifstack(fileReg1, h_img1, &imSize[0], bitPerSampleImg);//set bit as input images
			if (flagSaveInterFiles[2] == 1) writetifstack(fileReg2, h_img2Reg, &imSize[0], bitPerSampleImg);//set bit as input images
			printf("\tTime cost for  registration: %2.3f s\n", (float)(time2 - time1) / CLOCKS_PER_SEC);

			//  *************** Deconvolution: h_img1 and h_img2Reg ***********
			printf("... Deconvolution ...\n");
			f1 = fopen(fileLog, "a");
			fprintf(f1, "...Deconvolution...\n");
			fclose(f1);
			memset(h_decon, 0, totalSize * sizeof(float));
			(void)decon_dualview(h_decon, h_img1, h_img2Reg, &imSize[0], h_psf1, h_psf2, &psfSize[0], flagConstInitial,
				itNumForDecon, deviceNum, gpuMemMode, verbose, deconRecords, flagUnmatch, h_psf1_bp, h_psf2_bp);
			int gpuMemModeActual = int(deconRecords[0]);
			//printf("\tDeconvolution status: %d\n", runStatus);
			//deconRecords: 10 elements
			//[0]:  the actual GPU memory mode used;
			//[1] -[5]: initial GPU memory, after variables partially allocated, during processing, after processing, after variables released ( all in MB);
			//[6] -[9]: initializing time, prepocessing time, decon time, total time;	

			f1 = fopen(fileLog, "a");
			switch (gpuMemModeActual) {
			case 1:
				fprintf(f1, "	...Sufficient GPU memory, running in efficient mode !!!\n");
				break;
			case 2:
				fprintf(f1, "	...GPU memory partially optimized, running in memory saved mode !!!\n");
				break;
			case 3:
				fprintf(f1, "	...GPU memory fully optimized, running in memory saved mode !!!\n");
				break;
			default:
				fprintf(f1, "	...Not enough GPU memory, no deconvolution performed !!!\n");
			}
			//fprintf(f1, "	...all iteration time cost: %2.3f s\n", deconRecords[8]);
			fprintf(f1, "	...deconvolution time cost: %2.3f s\n", deconRecords[9]);
			fprintf(f1, "	...GPU free memory after deconvolution is %.0f MBites\n", deconRecords[5]);
			fclose(f1);
			writetifstack(fileDecon, h_decon, &imSize[0], bitPerSample);
			time3 = clock();
			printf("\tTime cost for  deconvolution: %2.3f s\n", (float)(time3 - time2) / CLOCKS_PER_SEC);

			///********* save max projections
			if ((flagSaveInterFiles[3] == 1) || (flagSaveInterFiles[4] == 1) || (flagSaveInterFiles[5] == 1)) {
				// 2D MP max projections
				(void)mp2dgpu(h_MP, &imSizeMP[0], h_decon, &imSize[0], (bool)flagSaveInterFiles[3], (bool)flagSaveInterFiles[4], (bool)flagSaveInterFiles[5]);
				tempSize[2] = 1;
				if (flagSaveInterFiles[3] == 1) {
					tempSize[0] = imSizeMP[0]; tempSize[1] = imSizeMP[1];
					writetifstack(fileDeconMP_XY, h_MP, &tempSize[0], bitPerSample);
				}
				if (flagSaveInterFiles[4] == 1) {
					tempSize[0] = imSizeMP[2]; tempSize[1] = imSizeMP[3];
					writetifstack(fileDeconMP_YZ, &h_MP[sx*sy], &tempSize[0], bitPerSample);
				}
				if (flagSaveInterFiles[5] == 1) {
					tempSize[0] = imSizeMP[4]; tempSize[1] = imSizeMP[5];
					writetifstack(fileDeconMP_ZX, &h_MP[sx*sy + sy * sz], &tempSize[0], bitPerSample);
				}
			}
			if (flagSaveInterFiles[6] == 1) { // 3D max projections: X-axis
				(void)mip3dgpu(h_MP3D1, &imSizeMP[0], h_decon, &imSize[0], 1, projectNum);
				writetifstack(fileDeconMP_3D_X, h_MP3D1, &imSizeMP[0], bitPerSample);
			}
			if (flagSaveInterFiles[7] == 1) { // 3D max projections: X-axis
				(void)mip3dgpu(h_MP3D2, &imSizeMP[0], h_decon, &imSize[0], 2, projectNum);
				writetifstack(fileDeconMP_3D_Y, h_MP3D2, &imSizeMP[0], bitPerSample);
			}
			time4 = clock();

			// release file names
			if (flagMultiColor) {
				free(outFolder); free(inFolder1); free(inFolder2); free(deconFolder); free(tmxFolder); free(regFolder1); //
				free(regFolder2);  free(deconFolderMP_XY); free(deconFolderMP_YZ); free(deconFolderMP_ZX);
				free(deconFolderMP_3D_X); free(deconFolderMP_3D_Y);
			}
			free(fileImg1); free(fileImg2); free(fileReg1); free(fileReg2); free(fileDecon); //
			free(fileTmx);  free(fileDeconMP_XY); free(fileDeconMP_YZ); free(fileDeconMP_ZX);
			free(fileDeconMP_3D_X); free(fileDeconMP_3D_Y);
			printf("...Time cost for current image is %2.3f s\n", (float)(time4 - time0) / CLOCKS_PER_SEC);
			f1 = fopen(fileLog, "a");
			fprintf(f1, "...Time cost for current image is %2.3f s\n", (float)(time4 - time0) / CLOCKS_PER_SEC);
			fclose(f1);

		}
	}
	////release CPU memory 
	free(h_affInitial);
	free(h_affWeighted);
	free(regRecords);
	free(deconRecords);
	free(h_img1In);
	free(h_img2In);
	free(h_img2Temp);
	free(h_img1);
	free(h_img2);
	free(h_img2Reg);
	free(h_psf1);
	free(h_psf2);
	free(h_psf1_bp);
	free(h_psf2_bp);
	free(h_decon);
	if ((flagSaveInterFiles[3] == 1) || (flagSaveInterFiles[4] == 1) || (flagSaveInterFiles[5] == 1))
		free(h_MP);
	if (flagSaveInterFiles[6] == 1)
		free(h_MP3D1);
	if (flagSaveInterFiles[7] == 1)
		free(h_MP3D2);
	endWhole = clock();
	printf("Total time cost for whole processing is %2.3f s\n", (float)(endWhole - startWhole) / CLOCKS_PER_SEC);
	f1 = fopen(fileLog, "a");
	fprintf(f1, "Total time cost for whole processing is %2.3f s\n", (float)(endWhole - startWhole) / CLOCKS_PER_SEC);
	fclose(f1);

	return 0;
}

#undef NDIM