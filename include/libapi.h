#ifdef __CUDACC__
typedef double2 dComplex;
#else
typedef struct{
	double x;
	double y;
} dComplex;
#endif

#ifdef __CUDACC__
typedef float2 fComplex;
#else
typedef struct{
	float x;
	float y;
} fComplex;

#endif


#define MAX_PATH 256

////***********API functions
//// file I/O
extern "C"
#ifdef _MSC_VER  
__declspec(dllexport) 
#endif
char* concat(int count, ...);

extern "C"
#ifdef _MSC_VER  
__declspec(dllexport) 
#endif
bool fexists(const char * filename);


extern "C"
#ifdef _MSC_VER  
__declspec(dllexport) 
#endif
unsigned short gettifinfo(char tifdir[], unsigned int *tifSize);

extern "C"
#ifdef _MSC_VER  
__declspec(dllexport) 
#endif
void readtifstack(float *h_Image, char tifdir[], unsigned int *imsize);

extern "C"
#ifdef _MSC_VER  
__declspec(dllexport) 
#endif
void writetifstack(char tifdir[], float *h_Image, unsigned int *imsize, unsigned short bitPerSample);

extern "C"
#ifdef _MSC_VER  
__declspec(dllexport) 
#endif
void readtifstack_16to16(unsigned short *h_Image, char tifdir[], unsigned int *imsize);

extern "C"
#ifdef _MSC_VER  
__declspec(dllexport) 
#endif
void writetifstack_16to16(char tifdir[], unsigned short *h_Image, unsigned int *imsize);

//// 2D registration
extern "C"
#ifdef _MSC_VER  
__declspec(dllexport) 
#endif 
int reg_2dgpu(float *h_reg, float *iTmx, float *h_img1, float *h_img2, int imSizex1, int imSizey1, int imSizex2, int imSizey2,
	int inputTmx, float FTOL, int itLimit, int deviceNum, float *regRecords);

extern "C"
#ifdef _MSC_VER  
__declspec(dllexport) 
#endif
int reg_2dshiftaligngpu(float *h_reg, float *iTmx, float *h_img1, float *h_img2, int imSizex1, int imSizey1, int imSizex2, int imSizey2,
	int inputTmx, float shiftRegion, int totalStep, int deviceNum, float *regRecords);

extern "C"
#ifdef _MSC_VER  
__declspec(dllexport) 
#endif
int reg_2dshiftalignXgpu(float *h_reg, float *iTmx, float *h_img1, float *h_img2, int imSizex1, int imSizey1, int imSizex2, int imSizey2,
	int inputTmx, float shiftRegion, int totalStep, int deviceNum, float *regRecords);

//// 3D registration
extern "C"
#ifdef _MSC_VER  
__declspec(dllexport) 
#endif
int reg_3dcpu(float *h_reg, float *iTmx, float *h_img1, float *h_img2, unsigned int *imSize1, unsigned int *imSize2, int regMethod,
int inputTmx, float FTOL, int itLimit, int subBgTrigger, int deviceNum, float *regRecords);

extern "C"
#ifdef _MSC_VER  
__declspec(dllexport) 
#endif
int reg_3dgpu(float *h_reg, float *iTmx, float *h_img1, float *h_img2, unsigned int *imSize1, unsigned int *imSize2, int regMethod,
int inputTmx, float FTOL, int itLimit, int subBgTrigger, int deviceNum, float *regRecords);

extern "C"
#ifdef _MSC_VER  
__declspec(dllexport) 
#endif
int reg_3dphasetransgpu(int *shiftXYZ, float *h_img1, float *h_img2, unsigned int *imSize1, unsigned int *imSize2, int downSample, int deviceNum, float *regRecords);

extern "C"
#ifdef _MSC_VER  
__declspec(dllexport) 
#endif
int affinetrans_3dgpu(float *h_reg, float *iTmx, float *h_img2, unsigned int *imSize1, unsigned int *imSize2, int deviceNum);

extern "C"
#ifdef _MSC_VER  
__declspec(dllexport) 
#endif
int affinetrans_3dgpu_16to16(unsigned short *h_reg, float *iTmx, unsigned short *h_img2, unsigned int *imSize1, unsigned int *imSize2, int deviceNum);

/// 3D deonvolution
extern "C"
#ifdef _MSC_VER  
__declspec(dllexport) 
#endif
int decon_singleview(float *h_decon, float *h_img, unsigned int *imSize, float *h_psf, float *h_psf_bp, unsigned int *psfSize,
int itNumForDecon, int deviceNum, int gpuMemMode, float *deconRecords);

extern "C"
#ifdef _MSC_VER  
__declspec(dllexport) 
#endif
int decon_dualview(float *h_decon, float *h_img1, float *h_img2, unsigned int *imSize, float *h_psf1, float *h_psf2,
unsigned int *psfSize, int itNumForDecon, int deviceNum, int gpuMemMode, float *deconRecords, bool flagUnmatch, float *h_psf_bp1, float *h_psf_bp2);

//// 3D fusion: registration and deconvolution
extern "C"
#ifdef _MSC_VER  
__declspec(dllexport) 
#endif
int fusion_dualview(float *h_decon, float *h_reg, float *iTmx, float *h_img1, float *h_img2, unsigned int *imSizeIn1, unsigned int *imSizeIn2,
	float *pixelSize1, float *pixelSize2, int imRotation, int regMethod, int flagInitialTmx, float FTOL, int itLimit, float *h_psf1, float *h_psf2,
	unsigned int *psfSizeIn, int itNumForDecon, int deviceNum, int gpuMemMode, float *fusionRecords, bool flagUnmatch, float *h_psf_bp1, float *h_psf_bp2);

/// maximum intensity projectoions:
extern "C"
#ifdef _MSC_VER  
__declspec(dllexport) 
#endif
int mp2Dgpu(float *h_MP, unsigned int *sizeMP, float *h_img, unsigned int *sizeImg, bool flagZProj, bool flagXProj, bool flagYProj);

extern "C"
#ifdef _MSC_VER  
__declspec(dllexport) 
#endif
int mp3Dgpu(float *h_MP, unsigned int *sizeMP, float *h_img, unsigned int *sizeImg, bool flagXaxis, bool flagYaxis, int projectNum);

//// 3D Decon and fusion: batch processing
extern "C"
#ifdef _MSC_VER  
__declspec(dllexport) 
#endif
int reg_3dgpu_batch(char *outMainFolder, char *folder1, char *folder2, char *fileNamePrefix1, char *fileNamePrefix2, int imgNumStart, int imgNumEnd, int imgNumInterval, int imgNumTest,
	float *pixelSize1, float *pixelSize2, int regMode, int imRotation, int flagInitialTmx, float *iTmx, float FTOL, int itLimit, int deviceNum, int *flagSaveInterFiles, float *records);

extern "C"
#ifdef _MSC_VER  
__declspec(dllexport) 
#endif
int fusion_dualview_batch(char *outFolder, char *inFolder1, char *inFolder2, char *fileNamePrefix1, char *fileNamePrefix2, int imgNumStart, int imgNumEnd, int imgNumInterval, int imgNumTest,
	float *pixelSize1, float *pixelSize2, int regMode, int imRotation, int flagInitialTmx, float *iTmx, float FTOL, int itLimit, char *filePSF1, char *filePSF2,
	int itNumForDecon, int deviceNum, int *flagSaveInterFiles, int bitPerSample, float *records, bool flagUnmatch, char *filePSF_bp1, char *filePSF_bp2);

// Query GPU device
extern "C"
#ifdef _MSC_VER  
__declspec(dllexport) 
#endif
void queryDevice();

// sum cpu
template <class T>
double sumcpu(T *h_idata, int totalSize);

// Align FFT size to 2^n or 64*n to boosting computation
int snapTransformSize(int dataSize);

// CPU functions
template <class T>
void addcpu(T *h_odata, T *h_idata1, T *h_idata2, int totalSize);

template <class T>
void addvaluecpu(T *h_odata, T *h_idata1, T h_idata2, int totalSize);

template <class T>
void subcpu(T *h_odata, T *h_idata1, T *h_idata2, int totalSize);

template <class T>
void multicpu(T *h_odata, T *h_idata1, T *h_idata2, int totalSize);

template <class T>
void divcpu(T *h_odata, T *h_idata1, T *h_idata2, int totalSize);

template <class T>
void multivaluecpu(T *h_odata, T *h_idata1, T h_idata2, int totalSize);

extern "C"
void multicomplexcpu(fComplex *h_odata, fComplex *h_idata1, fComplex *h_idata2, int totalSize);

template <class T>
void maxvaluecpu(T *h_odata, T *h_idata1, T h_idata2, int totalSize);

// Declare cuda functions
template <class T>
void add3Dgpu(T *d_odata, T *d_idata1, T *d_idata2, int sx, int sy, int sz);

template <class T>
void addvaluegpu(T *d_odata, T *d_idata1, T d_idata2, int sx, int sy, int sz);

template <class T>
void sub3Dgpu(T *d_odata, T *d_idata1, T *d_idata2, int sx, int sy, int sz);

template <class T>
void multi3Dgpu(T *d_odata, T *d_idata1, T *d_idata2, int sx, int sy, int sz);

template <class T>
void multivaluegpu(T *d_odata, T *d_idata1, T d_idata2, int sx, int sy, int sz);

extern "C"
void multicomplex3Dgpu(fComplex *d_odata, fComplex *d_idata1, fComplex *d_idata2, int sx, int sy, int sz);

extern "C"
void multidcomplex3Dgpu(dComplex *d_odata, dComplex *d_idata1, dComplex *d_idata2, int sx, int sy, int sz);

template <class T>
void div3Dgpu(T *d_odata, T *d_idata1, T *d_idata2, int sx, int sy, int sz);

extern "C"
void conj3Dgpu(fComplex *d_odata, fComplex *d_idata, int sx, int sy, int sz);

template <class T>
T sumgpu(T *d_idata, T *d_temp, T *h_temp, int totalSize);

template <class T>
double sum3Dgpu(T *d_idata, double *d_temp, double *h_temp, int sx, int sy, int sz);

template <class T>
T sumgpu1D(T *d_idata, T *d_temp, T *h_temp, int totalSize);

template <class T>
T max3Dgpu(int *corXYZ, T *d_idata, int sx, int sy, int sz);

template <class T>
T max3Dcpu(int *corXYZ, T *h_idata, int sx, int sy, int sz);

template <class T>
void maxvalue3Dgpu(T *d_odata, T *d_idata1, T d_idata2, int sx, int sy, int sz);

template <class T>
void maxprojection(T *d_odata, T *d_idata, int sx, int sy, int sz, int pDirection);

template <class T>
void changestorageordergpu(T *d_odata, T *d_idata, int sx, int sy, int sz, int orderMode);
//orderMode
// 1: change tiff storage order to C storage order
//-1: change C storage order to tiff storage order

template <class T>
void rotbyyaxis(T *d_odata, T *d_idata, int sx, int sy, int sz, int rotDirection);
//rot direction
// 1: rotate 90 deg around Y axis
//-1: rotate -90 deg around Y axis

void p2matrix(float *m, float *x);
void matrix2p(float *m, float *x);
extern "C" void matrixmultiply(float * m, float *m1, float *m2);

//void rot3Dbyyaxis(float *d_odata, float theta, int sx, int sz, int sx2, int sz2);
extern "C" void rot2matrix(float * p_out, float theta, int sx, int sy, int sz, int rotAxis);
// DOF to affine matrix
extern "C" void dof9tomatrix(float * p_out, float *p_dof, int dofNum);


template <class T>
void flipPSF(T *d_odata, T *d_idata, int sx, int sy, int sz);

template <class T>
void padPSF(
	T *d_PaddedPSF,
	T *d_PSF,
	int FFTx,
	int FFTy,
	int FFTz,
	int PSFx,
	int PSFy,
	int PSFz,
	int PSFox,
	int PSFoy,
	int PSFoz
	);

template <class T>
void padStack(
	T *d_PaddedStack,
	T *d_Stack,
	int FFTx,
	int FFTy,
	int FFTz,
	int sx,
	int sy,
	int sz,
	int imox,
	int imoy,
	int imoz
	);

template <class T>
void cropStack(
	T *d_PaddedStack,
	T *d_Stack,
	int FFTx,
	int FFTy,
	int FFTz,
	int sx,
	int sy,
	int sz,
	int imox,
	int imoy,
	int imoz
	);

extern "C" void CopyTranMatrix(
	float *p,
	int dataSize
	);

//extern "C"
//void cudacopyhosttoarray(cudaArray *d_Array, cudaChannelFormatDesc channelDesc, float *h_idata, int sx, int sy, int sz);

template <class T>
void cudacopyhosttoarray(cudaArray *d_Array, cudaChannelFormatDesc channelDesc, T *h_idata, int sx, int sy, int sz);

extern "C"
void cudacopydevicetoarray(cudaArray *d_Array, cudaChannelFormatDesc channelDesc, float *h_idata, int sx, int sy, int sz);

extern "C"
void BindTexture(
cudaArray *d_Stack,
cudaChannelFormatDesc channelDesc
);

extern "C"
void BindTexture16(
cudaArray *d_Stack,
cudaChannelFormatDesc channelDesc
);

extern "C"
void UnbindTexture();

extern "C"
void UnbindTexture16();

extern "C"
void AccessTexture(float x,
	float y,
	float z
);


template <class T>
void affineTransform(T *d_t,
	int sx,
	int sy,
	int sz,
	int sx2,
	int sy2,
	int sz2
);

double corrfunc(float *d_s, // source stack
	float *d_sSqr,
	float *d_corr,
	float *aff,
	double *d_temp,
	double *h_temp,
	int sx,
	int sy,
	int sz,
	int sx2,
	int sy2,
	int sz2
	);

// optimized correlation function
double corrfunc2(float *d_s, // source stack
	float *aff,
	double *d_temp1,
	double *d_temp2,
	double *h_temp,
	int sx,
	int sy,
	int sz,
	int sx2,
	int sy2,
	int sz2
	);
double corrfunc3(float *d_s, // source stack
	float *aff,
	double *d_temp1,
	double *d_temp2,
	double *d_temp3,
	double *h_temp,
	int sx,
	int sy,
	int sz,
	int sx2,
	int sy2,
	int sz2
	);

double corrfunccpu(float *h_s, 
	float *h_t,// source stack
	float *aff,
	int sx,
	int sy,
	int sz,
	int sx2,
	int sy2,
	int sz2
	);

double corrfunccpu2(float *h_s,
	float *h_t,// source stack
	float *aff,
	int sx,
	int sy,
	int sz,
	int sx2,
	int sy2,
	int sz2
	);


// 2D registration
extern "C"
void affineTransform2D(float *d_t, int sx, int sy, int sx2, int sy2);

double corrfunc2D(float *d_s, // source stack
	float *d_sSqr,
	float *d_corr,
	float *aff,
	float *h_temp,
	int sx,
	int sy,
	int sx2,
	int sy2
	);

extern "C"
void cudacopyhosttoarray2D(cudaArray *d_Array, cudaChannelFormatDesc channelDesc, float *h_idata, int totalSize);

extern "C" bool checkmatrix(float *m);

extern "C" void BindTexture2D(cudaArray *d_Array, cudaChannelFormatDesc channelDesc );

extern "C" void UnbindTexture2D();

float ilerp(float x, float x1, float x2, float q00, float q01);

float ibilerp(float x, float y, float x1, float x2, float y1, float y2, float q11, float q12, float q21, float q22);

float itrilerp(float x, float y, float z, float x1, float x2, float y1, float y2, float z1, float z2,
float q111, float q112, float q121, float q122, float q211, float q212, float q221, float q222);

void affinetransformcpu(float *h_s,
	float *h_t,// source stack
	float *aff,
	int sx,
	int sy,
	int sz,
	int sx2,
	int sy2,
	int sz2
	);
