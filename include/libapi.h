//// *********** API functions ********************** ////
#ifdef PROJECT_EXPORTS
#	ifdef _WIN32 // or _MSC_VER
#		define PROJECT_API extern "C" __declspec(dllexport) // Win export API
#	else
#		define PROJECT_API extern "C" // Linux export API
#	endif
#else
#  define PROJECT_API		// import API
#endif
//// file I/O
PROJECT_API char* concat(int count, ...);
PROJECT_API bool fexists(const char * filename);
PROJECT_API unsigned short gettifinfo(char tifdir[], unsigned int *tifSize);
PROJECT_API void readtifstack(float *h_Image, char *tifdir, unsigned int *imsize);
PROJECT_API void writetifstack(char *tifdir, float *h_Image, unsigned int *imsize, unsigned short bitPerSample);
PROJECT_API void readtifstack_16to16(unsigned short *h_Image, char *tifdir, unsigned int *imsize);
PROJECT_API void writetifstack_16to16(char *tifdir, unsigned short *h_Image, unsigned int *imsize);

// Query GPU device
PROJECT_API void queryDevice();

//// 2D registration
PROJECT_API int reg2d(float *h_reg, float *iTmx, float *h_img1, float *h_img2, unsigned int *imSize1, unsigned int *imSize2, int regChoice,
	bool flagTmx, float FTOL, int itLimit, int deviceNum, int gpuMemMode, bool verbose, float *records);

//// 3D affine transformation
PROJECT_API bool checkmatrix(float *iTmx, long long int sx, long long int sy, long long int sz);

PROJECT_API int atrans3dgpu(float *h_reg, float *iTmx, float *h_img2, unsigned int *imSize1, unsigned int *imSize2, int deviceNum);

PROJECT_API int atrans3dgpu_16bit(unsigned short *h_reg, float *iTmx, unsigned short *h_img2, unsigned int *imSize1, unsigned int *imSize2, int deviceNum);

//// 3D registration
PROJECT_API int reg3d(float *h_reg, float *iTmx, float *h_img1, float *h_img2, unsigned int *imSize1, unsigned int *imSize2, int regChoice, int regMethod,
	bool inputTmx, float FTOL, int itLimit, int deviceNum, int gpuMemMode, bool verbose, float *records);

PROJECT_API int reg_3dgpu(float *h_reg, float *iTmx, float *h_img1, float *h_img2, unsigned int *imSize1, unsigned int *imSize2, int regMethod,
int inputTmx, float FTOL, int itLimit, int subBgTrigger, int deviceNum, float *regRecords);

//// 3D deonvolution
PROJECT_API int decon_singleview(float *h_decon, float *h_img, unsigned int *imSize, float *h_psf, unsigned int *psfSize, bool initialFlag,
int itNumForDecon, int deviceNum, int gpuMemMode, bool verbose, float *deconRecords, bool flagUnmatch, float *h_psf_bp);

PROJECT_API int decon_dualview(float *h_decon, float *h_img1, float *h_img2, unsigned int *imSize, float *h_psf1, float *h_psf2, unsigned int *psfSize,
bool initialFlag, int itNumForDecon, int deviceNum, int gpuMemMode, bool verbose, float *deconRecords, bool flagUnmatch, float *h_psf_bp1, float *h_psf_bp2);

//// 3D fusion: registration and deconvolution
PROJECT_API int fusion_dualview(float *h_decon, float *h_reg, float *h_prereg1, float *h_prereg2, float *iTmx, float *h_img1, float *h_img2, unsigned int *imSizeIn1, unsigned int *imSizeIn2,
	float *pixelSize1, float *pixelSize2, int imRotation, bool flagTmx, int regChoice, float FTOL, int itLimit, float *h_psf1, float *h_psf2,
	unsigned int *psfSizeIn, int itNumForDecon, int deviceNum, int gpuMemMode, bool verbose, float *fusionRecords, bool flagUnmatch, float *h_psf_bp1, float *h_psf_bp2);
	
//// maximum intensity projectoions:
PROJECT_API int mp2dgpu(float *h_MP, unsigned int *sizeMP, float *h_img, unsigned int *sizeImg, bool flagZProj, bool flagXProj, bool flagYProj);

PROJECT_API int mp3dgpu(float *h_MP, unsigned int *sizeMP, float *h_img, unsigned int *sizeImg, bool flagXaxis, bool flagYaxis, int projectNum);

PROJECT_API
int mip3dgpu(float *h_MP, unsigned int *sizeMP, float *h_img, unsigned int *sizeImg, int rAxis, long long int projectNum);

PROJECT_API
int alignsize3d(float *h_odata, float *h_idata, long long int sx, long long int sy, long long int sz, long long int sx2, long long int sy2, long long int sz2, int gpuMemMode);

PROJECT_API
int imresize3d(float *h_odata, float *h_idata, long long int sx1, long long int sy1, long long int sz1, long long int sx2, long long int sy2, long long int sz2, int deviceNum);

PROJECT_API
int imoperation3D(float *h_odata, unsigned int *sizeOut, float *h_idata, unsigned int *sizeIn, int opChoice, int deviceNum);

#undef PROJECT_API