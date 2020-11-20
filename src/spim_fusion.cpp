#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>       
#include <iostream>
#include <fstream>
#include <string.h>
#include <math.h>
#include <ctime>

extern "C" {
#include "libapi.h"
}


void helpmessage(char *appName, bool flagHelp) {
	printf("\n%s: Dual-view fusion (registration and joint deconvolution) for diSPIM images\n", appName);
	printf("\nUsage:\t%s -i1 <inputImageName1> -i2 <inputImageName2> -fp1 <psfImageName1> -fp2 <psfImageName2> -o <outputImageName> [OPTIONS]\n", appName);
	if (!flagHelp) {
		printf("\nUse command for more details:\n\t%s -help or %s -h\n", appName, appName);
		return;
	}
	printf("\tOnly 16-bit or 32-bit standard TIFF images are currently supported.\n");
	printf("\n= = [OPTIONS: mandatory] = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = \n");
	printf("\t-i1 <filename>\t\tInput image filename 1 (SPIM A) (mandatory)\n");
	printf("\t-i2 <filename>\t\tInput image filename 2 (SPIM B) (mandatory)\n");
	printf("\t-fp1 <filename>\t\tPSF1 (forward projector 1) image filename (mandatory)\n");
	printf("\t-fp2 <filename>\t\tPSF2 (forward projector 2) image filename (mandatory)\n");
	printf("\t-o <filename>\t\tOutput filename of the deconvolved image (mandatory)\n");

	printf("= = [OPTIONS: pre-processing] = = = = = = = = = = = = = \n");
	printf("\t-pxx1 <float>\t\tPixel Size X 1 (um) [0.1625]\n");
	printf("\t-pxy1 <float>\t\tPixel Size Y 1 (um) [0.1625]\n");
	printf("\t-pxz1 <float>\t\tPixel Size Z 1 (um) [1.0]\n");
	printf("\t-pxx2 <float>\t\tPixel Size X 2 (um) [0.1625]\n");
	printf("\t-pxy2 <float>\t\tPixel Size Y 2 (um) [0.1625]\n");
	printf("\t-pxz2 <float>\t\tPixel Size Z 2 (um) [1.0]\n");
	//printf("\t-bg1 <float>\t\tBackground value 1: [0, no background subtraction]\n");
	//printf("\t-bg2 <float>\t\tBackground value 2: [0, no background subtraction]\n");
	printf("\t-imgrot <int>\t\tImage 2 (SPIM B) rotation: [-1]\n");
	printf("\t\t\t\t0, no rotation; 1: 90 deg by Y-axis; -1: -90 deg by Y-axis \n");

	printf("= = [OPTIONS: registration] = = = = = = = = = = = = = = \n");
	printf("\t-oreg1 <filename>\tOutput filename of the registrered image 1 (optional) [no output]\n");
	printf("\t-oreg2 <filename>\tOutput filename of the registrered image 2 (optional) [no output]\n");
	printf("\t-itmx <filename>\tInput tranformation matrix filename [identity matrix]\n");
	printf("\t-otmx <filename>\tOutput tranformation matrix filename (optional) [no output]\n");
	printf("\t-regc <int>\t\tOptions for registration choice [2]\n");
	printf("\t\t\t\t0: no registration, but transform image 2 based on input matrix\n");
	printf("\t\t\t\t1: phasor registraion (pixel-level translation, input matrix disabled)\n");
	printf("\t\t\t\t2: affine registration (with or without input matrix)\n");
	printf("\t\t\t\t3: phasor registration --> affine registration (input matrix disabled)\n");
	printf("\t\t\t\t4: 2D MIP registration --> affine registration (input matrix disabled)\n");
	printf("\t-affm <int>\t\tOptions for affine method [7]\n");
	printf("\t\t\t\t0: no affine, but transform image 2 based on input matrix\n");
	printf("\t\t\t\t1: translation only (3 DOF)\n");
	printf("\t\t\t\t2: rigid-body (6 DOF)\n");
	printf("\t\t\t\t3: 7 DOF (translation, rotation, scaling equally in 3 dimensions)\n");
	printf("\t\t\t\t4: 9 DOF (translation, rotation, scaling)\n");
	printf("\t\t\t\t5: directly 12 DOF\n");
	printf("\t\t\t\t6: rigid body (6 DOF) --> 12 DOF\n");
	printf("\t\t\t\t7: 3 DOF --> 6 DOF--> 9 DOF--> 12 DOF\n");
	printf("\t-ftol <float>\t\tTolerance or threshold of the stop point [0.0001]\n");
	printf("\t-itreg <int>\t\tMaximum iteration number for registration[3000]\n");

	printf("= = [OPTIONS: deconvolution] = = = = = = = = = = = = = \n");
	printf("\t-bp1 <filename>\t\tBackward projector 1 filename [flip of PSF1]\n");
	printf("\t-bp2 <filename>\t\tBackward projector 2 filename [flip of PSF2]\n");
	printf("\t-it <int>\t\tIteration number of the deconvolution [10]\n");
	printf("\t-cON or -cOFF\t\tON: constant as initialization; OFF: input image as initialization [OFF]\n");

	printf("= = [OPTIONS: others] = = = = = = = = = = = = = \n");
	printf("\t-gm <int>\t\tChoose CPU or GPU processing [-1]\n");
	printf("\t\t\t\t-1: automatically choose\n");
	printf("\t\t\t\t0: all in CPU (currently does not work)\n");
	printf("\t\t\t\t1: efficient GPU mode if enough GPU memory\n");
	printf("\t\t\t\t2: memory-saved GPU mode if insufficient GPU memroy\n");
	printf("\t-dev <int>\t\tSpecify the GPU device if multiple GPUs on board [0]\n");
	printf("\t-bit <int>\t\tSpecify output image bit: 16 or 32 [same as input image]\n");
	printf("\t-verbON or -verbOFF\tTurn on/off verbose information [ON]\n");
	printf("\t-log <filename>\t\tLog filename [no log file] (currently does not work)\n");
	return;
}

int main(int argc, char* argv[])
{
	if (argc == 1) {
		helpmessage(argv[0], false);
		return EXIT_SUCCESS;
	}

	// * * * * * Variables and default values * * * * * * * //
	char *fileImg1 = "../Data/SPIMA_0_crop.tif";
	char *fileImg2 = "../Data/SPIMB_0_crop.tif";
	char *filePSF1 = "../Data/PSFA.tif";
	char *filePSF2 = "../Data/PSFA.tif";
	char *fileDecon = "../Data/Decon_0.tif";
	// preprocessing parameters
	float pixelSizex1 = 0.1625, pixelSizey1 = 0.1625, pixelSizez1 = 1.0;
	float pixelSizex2 = 0.1625, pixelSizey2 = 0.1625, pixelSizez2 = 1.0;
	unsigned int imSize1In[3], imSize2In[3], imSize1[3], imSize2[3], psfSize[3], imSize[3], tempSize[3];
	long long int sx, sy, sz, sx1, sy1, sz1, sx2, sy2, sz2, sxpsf, sypsf, szpsf, sxtemp, sytemp, sztemp;
	bool flagBg1 = false, flagBg2 = false;
	float bgValue1 = 0, bgValue2 = 0;
	bool flagImg2Rot = true;
	int imRotation = -1;
	// registration parameters
	bool flagSaveReg1 = false, flagSaveReg2 = false;
	char *fileReg1 = "../Data/SPIMA_reg_0.tif";
	char *fileReg2 = "../Data/SPIMB_reg_0.tif";
	bool flagiTmx = false;
	float *iTmx = (float *)malloc(12 * sizeof(float));
	memset(iTmx, 0, 12 * sizeof(float));
	iTmx[0] = iTmx[5] = iTmx[10] = 1;
	char *fileiTmx = "../Data/tmxIn.tmx";
	bool flagoTmx = false;
	char *fileoTmx = "../Data/tmxOut.tmx";
	int regChoice = 2;
	int affMethod = 6;
	float FTOL = 0.0001;
	int itLimit = 3000;

	// decon parameters
	bool flagUnmatch = false;
	char *filePSF1_bp = "../Data/PSFA_bp.tif";
	char *filePSF2_bp = "../Data/PSFB_bp.tif";
	int itNumForDecon = 10; // decon it number
	bool flagConstInitial = false;
	// other parameters
	int deviceNum = 0;
	int gpuMemMode = -1;
	bool verbose = true;
	bool flagBitInput = true;
	unsigned int bitPerSample = 16;
	bool flagLog = false;
	char *fileLog = NULL;
	// ****************** Processing Starts***************** //
	// *** variables for memory and time cost records
	clock_t time0, time1, time2, time3;
	time0 = clock();
	// *** get arguments
	for (int i = 1; i < argc; i++)
	{
		if (strcmp(argv[i], "-help") == 0 || strcmp(argv[i], "-h") == 0)
		{
			helpmessage(argv[0], true);
			return EXIT_SUCCESS;
		}
		else if (strcmp(argv[i], "-i1") == 0)
		{
			fileImg1 = argv[++i];
		}
		else if (strcmp(argv[i], "-i2") == 0)
		{
			fileImg2 = argv[++i];
		}
		else if (strcmp(argv[i], "-fp1") == 0)
		{
			filePSF1 = argv[++i];
		}
		else if (strcmp(argv[i], "-fp2") == 0)
		{
			filePSF2 = argv[++i];
		}
		else if (strcmp(argv[i], "-o") == 0)
		{
			fileDecon = argv[++i];
		}

		// preprocessing parameters
		else if (strcmp(argv[i], "-pxx1") == 0)
		{
			pixelSizex1 = (float)atof(argv[++i]);
		}
		else if (strcmp(argv[i], "-pxy1") == 0)
		{
			pixelSizey1 = (float)atof(argv[++i]);
		}
		else if (strcmp(argv[i], "-pxz1") == 0)
		{
			pixelSizez1 = (float)atof(argv[++i]);
		}
		else if (strcmp(argv[i], "-pxx2") == 0)
		{
			pixelSizex2 = (float)atof(argv[++i]);
		}
		else if (strcmp(argv[i], "-pxy2") == 0)
		{
			pixelSizey2 = (float)atof(argv[++i]);
		}
		else if (strcmp(argv[i], "-pxz2") == 0)
		{
			pixelSizez2 = (float)atof(argv[++i]);
		}
		else if (strcmp(argv[i], "-bg1") == 0)
		{
			bgValue1 = (float)atof(argv[++i]);
			flagBg1 = true;
		}
		else if (strcmp(argv[i], "-bg2") == 0)
		{
			bgValue2 = (float)atof(argv[++i]);
			flagBg2 = true;
		}
		else if (strcmp(argv[i], "-imgrot") == 0)
		{
			imRotation = atoi(argv[++i]);
		}
		// registration parameters
		else if (strcmp(argv[i], "-oreg1") == 0)
		{
			fileReg1 = argv[++i];
			flagSaveReg1 = true;
		}
		else if (strcmp(argv[i], "-oreg2") == 0)
		{
			fileReg2 = argv[++i];
			flagSaveReg2 = true;
		}
		else if (strcmp(argv[i], "-itmx") == 0)
		{
			fileiTmx = argv[++i];
			flagiTmx = true;
		}
		else if (strcmp(argv[i], "-otmx") == 0)
		{
			fileoTmx = argv[++i];
			flagoTmx = true;
		}
		else if (strcmp(argv[i], "-regc") == 0)
		{
			regChoice = atoi(argv[++i]);
		}
		else if (strcmp(argv[i], "-affm") == 0)
		{
			affMethod = atoi(argv[++i]);
		}
		else if (strcmp(argv[i], "-ftol") == 0)
		{
			FTOL = (float)atof(argv[++i]);
		}
		else if (strcmp(argv[i], "-itreg") == 0)
		{
			itLimit = atoi(argv[++i]);
		}
		// decon parameters
		else if (strcmp(argv[i], "-bp1") == 0)
		{
			filePSF1_bp = argv[++i];
			flagUnmatch = true;
		}
		else if (strcmp(argv[i], "-bp2") == 0)
		{
			filePSF2_bp = argv[++i];
			flagUnmatch = true;
		}
		else if (strcmp(argv[i], "-it") == 0)
		{
			itNumForDecon = atoi(argv[++i]);
		}
		else if (strcmp(argv[i], "-cON") == 0)
		{
			flagConstInitial = true;
		}
		else if (strcmp(argv[i], "-cOFF") == 0)
		{
			flagConstInitial = false;
		}
		// other parameters
		else if (strcmp(argv[i], "-gm") == 0)
		{
			gpuMemMode = atoi(argv[++i]);
		}
		else if (strcmp(argv[i], "-dev") == 0)
		{
			deviceNum = atoi(argv[++i]);
		}
		else if (strcmp(argv[i], "-verbON") == 0)
		{
			verbose = true;
		}
		else if (strcmp(argv[i], "-verbOFF") == 0)
		{
			verbose = false;
		}
		else if (strcmp(argv[i], "-bit") == 0)
		{
			bitPerSample = atoi(argv[++i]);
			flagBitInput = false;
		}
		else if (strcmp(argv[i], "-log") == 0)
		{
			flagLog = false; // modify to true and set log path;
		}
	}

	printf("=====================================================\n");
	printf("=== diSPIM Fusion settings ...\n");
	printf("... Image information: \n");
	printf("\tInput image 1: %s\n", fileImg1);
	printf("\tInput image 2: %s\n", fileImg2);
	printf("\tPSF 1 (forward projector) image: %s\n", filePSF1);
	printf("\tPSF 2 (forward projector) image: %s\n", filePSF2);
	if (flagUnmatch) {
		printf("\tBackward projector 1 image path: %s\n", filePSF1_bp);
		printf("\tBackward projector 2 image path: %s\n", filePSF2_bp);
	}
	printf("\tOutput image: %s\n", fileDecon);
	// input image information and size
	unsigned int bitPerSampleImg = gettifinfo(fileImg1, &imSize1In[0]);
	bitPerSampleImg = gettifinfo(fileImg2, &imSize2In[0]);
	unsigned int bitPerSamplePSF = gettifinfo(filePSF1, &psfSize[0]);
	bitPerSamplePSF = gettifinfo(filePSF2, &tempSize[0]);
	if (bitPerSampleImg != 16 && bitPerSampleImg != 32) {
		fprintf(stderr, "***Input images are not supported, please use 16-bit or 32-bit image !!!\n");
		fprintf(stderr, "*** FAILED - ABORTING\n");
		exit(1);
	}
	if (flagBitInput) bitPerSample = bitPerSampleImg;
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

	printf("\tInput image 1 size %d x %d x %d\n  ", imSize1In[0], imSize1In[1], imSize1In[2]);
	printf("\t\t pixel size %.4f um x %.4f um x %.4f um\n", pixelSizex1, pixelSizey1, pixelSizez1);
	printf("\tInput image 2 size %d x %d x %d\n  ", imSize2In[0], imSize2In[1], imSize2In[2]);
	printf("\t\t pixel size %.4f um x %.4f um x %.4f um\n", pixelSizex2, pixelSizey2, pixelSizez2);
	printf("\tPSF image size %d x %d x %d\n  ", psfSize[0], psfSize[1], psfSize[2]);
	printf("\t\t pixel size %.4f um x %.4f um x %.4f um\n", pixelSizex1, pixelSizex1, pixelSizex1);
	printf("\tOutput image size %d x %d x %d\n  ", imSize[0], imSize[1], imSize[2]);
	printf("\t\t pixel size %.4f um x %.4f um x %.4f um\n", pixelSizex1, pixelSizex1, pixelSizex1);
	// Show preprocessing parameters
	printf("... Preprocessing Paremeters:\n");
	if (flagBg1) {
		printf("\tImage 1 background subtraction: %.1f\n", bgValue1);
	}
	else {
		printf("\tImage 1 background subtraction: no\n");
	}
	if (flagBg2) {
		printf("\tImage 2 background subtraction: %.1f\n", bgValue2);
	}
	else {
		printf("\tImage 2 background subtraction: no\n");
	}
	switch (imRotation) {
	case 0:
		printf("\tImage 2 rotation: no rotation\n"); break;
	case 1:
		printf("\tImage 2 rotation: 90 deg by Y-axis\n"); break;
	case -1:
		printf("\tImage 2 rotation: -90 deg by Y-axis\n"); break;
	}
	// Show registration parameters
	printf("... Registration Paremeters:\n");
	if (flagSaveReg1) {
		printf("\tSave registered image 1: %s\n", fileReg1);
	}
	else {
		printf("\tSave registered image 1: no\n");
	}
	if (flagSaveReg2) {
		printf("\tSave registered image 2: %s\n", fileReg2);
	}
	else {
		printf("\tSave registered image 2: no\n");
	}
	if (flagiTmx) {
		printf("\tInitial transformation matrix: %s\n", fileiTmx);
	}
	else {
		printf("\tInitial transformation matrix: Default\n");
	}
	if (flagoTmx) {
		printf("\tSave output transformation matrix: %s\n", fileoTmx);
	}
	else {
		printf("\tSave output transformation matrix: no \n");
	}
	switch (regChoice) {
	case 0:
		printf("\tRegistration choice: no registration\n");
		break;
	case 1:
		printf("\tRegistration choice: phasor registration\n");
		break;
	case 2:
		printf("\tRegistration choice: affine registration\n");
		break;
	case 3:
		printf("\tRegistration choice: pahse registration --> affine registration\n");
		break;
	case 4:
		printf("\tRegistration choice: 2D registration --> affine registration\n");
		break;
	default:
		printf("\tWrong registration choice, processing stopped !!!\n");
		return 1;
	}
	if (regChoice >= 2) {
		switch (affMethod) {
		case 0:
			printf("\tAfine registration method: no registration\n");
			break;
		case 1:
			printf("\tAfine registration method: tanslation only\n");
			break;
		case 2:
			printf("\tAfine registration method: rigid body\n");
			break;
		case 3:
			printf("\tAfine registration method: 7 DOF\n");
			break;
		case 4:
			printf("\tAfine registration method: 9 DOF\n");
			break;
		case 5:
			printf("\tAfine registration method: 12 DOF\n");
			break;
		case 6:
			printf("\tAfine registration method: rigid body --> 12 DOF\n");
			break;
		case 7:
			printf("\tAfine registration method: 3 DOF --> 6 DOF --> 9 DOF --> 12 DOF\n");
			break;
		default:
			printf("\tWrong affine registration method, processing stopped !!!\n");
			return 1;
		}
	}
	printf("\tTolerance or threshold: %f\n", FTOL);
	printf("\tMaximum iteration number: %d\n", itLimit);
	// Show deconvolution parameters
	printf("... Deconvolution Paremeters:\n");
	if (flagUnmatch) {
		printf("\tUse traditional backward projector: no\n");
	}
	else {
		printf("\tUse traditional backward projector: yes\n");
	}
	printf("\tIteration number of the deconvolution: %d\n", itNumForDecon);
	if (flagConstInitial) {
		printf("\tInitialization of the deconvolution: constant mean of the input image\n");
	}
	else {
		printf("\tInitialization of the deconvolution: the input image\n");
	}
	
	// Show other parameters
	switch (gpuMemMode) {
	case -1:
		printf("\tCPU or GPU processing: automatically setting\n");
		printf("\tPotential GPU device number: %d\n", deviceNum);
		break;
	case 0:
		printf("\tCPU or GPU processing: CPU\n");
		break;
	case 1:
		printf("\tCPU or GPU processing: efficient GPU\n");
		printf("\tGPU device number: %d\n", deviceNum);
		break;
	case 2:
		printf("\tCPU or GPU processing: memory-saved GPU\n");
		printf("\tGPU device number: %d\n", deviceNum);
		break;
	default:
		printf("\tWrong GPU mode setting, processing stopped !!!\n");
		return 1;
	}
	if (flagBitInput) {
		printf("\tOutput image bit: %d bit, same as input image\n", bitPerSample);
	}
	else {
		printf("\tOutput image bit: %d bit\n", bitPerSample);
	}
	if (verbose) {
		printf("\tverbose information: true\n");
	}
	else {
		printf("\tverbose information: false\n");
	}
	printf("=====================================================\n\n");

	// ************** Preprocessing **********************
	printf("... Preprocessing ...\n");
	printf("\tInitializing and image reading ...\n");
	// image variables
	float 
		*h_img1In,
		*h_img2In,
		*h_img1,
		*h_img2Temp, // rotation
		*h_img2,
		*h_img2Reg,
		*h_decon,
		*h_psf1,
		*h_psf2,
		*h_psf1_bp,
		*h_psf2_bp;
	float *regRecords = (float *)malloc(11 * sizeof(float));
	float *deconRecords = (float *)malloc(10 * sizeof(float));

	h_img1In = (float *)malloc(totalSizeIn1 * sizeof(float));
	h_img2In = (float *)malloc(totalSizeIn2 * sizeof(float));
	h_img1 = (float *)malloc(totalSize1 * sizeof(float));
	h_img2Temp = (float *)malloc(totalSizeIn2 * sizeof(float)); // rotation
	h_img2 = (float *)malloc(totalSize2 * sizeof(float));

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
	
	free(h_img1In); // release CPU memory
	free(h_img2In);
	free(h_img2Temp);
	time1 = clock();
	printf("\tTime cost for  preprocessing: %2.3f s\n", (float)(time1 - time0) / CLOCKS_PER_SEC);


	//  *************** Registration: h_img1 and h_img2 ***********
	printf("... Registration ...\n");
	h_img2Reg = (float *)malloc(totalSize1 * sizeof(float));
	memset(h_img2Reg, 0, totalSize1 * sizeof(float));
	FILE *fTmxIn = NULL, *fTmxOut = NULL;
	if (flagiTmx) {
		if (fexists(fileiTmx)) {
			fTmxIn = fopen(fileiTmx, "r");
			for (int j = 0; j < 12; j++)
			{
				fscanf(fTmxIn, "%f", &iTmx[j]);
			}
			fclose(fTmxIn);
		}
		else {
			printf("***** Iput transformation matrix file does not exist: %s\n", fileiTmx);
			return 1;
		}
	}

	int runStatus = -1;
	runStatus = reg3d(h_img2Reg, iTmx, h_img1, h_img2, &imSize1[0], &imSize2[0], regChoice, affMethod,
		flagiTmx, FTOL, itLimit, deviceNum, gpuMemMode, verbose, regRecords);
	// printf("\tRegistration status: %d\n", runStatus);

	// save matrix file and reg images
	if (flagoTmx) {
		fTmxOut = fopen(fileoTmx, "w");
		for (int j = 0; j < 12; j++)
		{
			fprintf(fTmxOut, "%f\t", iTmx[j]);
			if ((j + 1) % 4 == 0)
				fprintf(fTmxOut, "\n");
		}
		fprintf(fTmxOut, "%f\t%f\t%f\t%f\n", 0.0, 0.0, 0.0, 1.0);
		fclose(fTmxOut);
	}
	if(flagSaveReg1) writetifstack(fileReg1, h_img1, &imSize1[0], bitPerSampleImg);//set bit as input images
	if (flagSaveReg2) writetifstack(fileReg2, h_img2Reg, &imSize1[0], bitPerSampleImg);//set bit as input images
	
	free(h_img2); // release CPU memory
	free(regRecords);
	time2 = clock();
	printf("\tTime cost for  registration: %2.3f s\n", (float)(time2 - time1) / CLOCKS_PER_SEC);

	//  *************** Deconvolution: h_img1 and h_img2Reg ***********
	printf("... Deconvolution ...\n");
	h_decon = (float *)malloc(totalSize * sizeof(float));
	h_psf1 = (float *)malloc(totalSizePSF * sizeof(float));
	h_psf2 = (float *)malloc(totalSizePSF * sizeof(float));
	h_psf1_bp = (float *)malloc(totalSizePSF * sizeof(float));
	h_psf2_bp = (float *)malloc(totalSizePSF * sizeof(float));
	runStatus = -1;
	memset(h_decon, 0, totalSize * sizeof(float));
	readtifstack(h_psf1, filePSF1, &psfSize[0]);
	readtifstack(h_psf2, filePSF2, &psfSize[0]);
	if (flagUnmatch) {
		readtifstack(h_psf1_bp, filePSF1_bp, &tempSize[0]);
		readtifstack(h_psf2_bp, filePSF2_bp, &tempSize[0]);
	}
	runStatus = decon_dualview(h_decon, h_img1, h_img2Reg, &imSize[0], h_psf1, h_psf2, &psfSize[0], flagConstInitial,
		itNumForDecon, deviceNum, gpuMemMode, verbose, deconRecords, flagUnmatch, h_psf1_bp, h_psf2_bp);
	int gpuModeActual = int(deconRecords[0]);
	// printf("\tDeconvolution status: %d\n", runStatus);
	//printf("\tGPU running mode: %d\n", gpuModeActual);
	writetifstack(fileDecon, h_decon, &imSize[0], bitPerSample);
	//free CPU memory
	free(h_decon);
	free(h_img1);
	free(h_img2Reg);
	free(h_psf1);
	free(h_psf2);
	free(h_psf1_bp);
	free(h_psf2_bp);
	free(deconRecords);
	time3 = clock();
	printf("\tTime cost for  deconvolution: %2.3f s\n", (float)(time3 - time2) / CLOCKS_PER_SEC);

	printf("\n=== Processing completed, time cost for  whole processing: %2.3f s\n", (float)(time3 - time0) / CLOCKS_PER_SEC);

	return 0;
}

