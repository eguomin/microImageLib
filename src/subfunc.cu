#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
// Includes CUDA
//#include <cuda.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <memory>
#include "device_launch_parameters.h"

#include "cukernel.cuh"

extern "C"
bool isPow2(unsigned int x)
{
	return ((x&(x - 1)) == 0);
};

//Round a / b to nearest higher integer value
inline int iDivUp(int a, int b)
{
	return (a % b != 0) ? (a / b + 1) : (a / b);
}

//Align a to nearest higher multiple of b
inline int iAlignUp(int a, int b)
{
	return (a % b != 0) ? (a - a % b + b) : a;
}


int snapTransformSize(int dataSize)//
{
	int hiBit;
	unsigned int lowPOT, hiPOT;

	dataSize = iAlignUp(dataSize, 16);

	for (hiBit = 31; hiBit >= 0; hiBit--)
		if (dataSize & (1U << hiBit))
		{
			break;
		}

	lowPOT = 1U << hiBit;

	if (lowPOT == (unsigned int)dataSize)
	{
		return dataSize;
	}

	hiPOT = 1U << (hiBit + 1);

	if (hiPOT <= 128)//把256的阈值改成了128
	{
		return hiPOT;
	}
	else
	{
		return iAlignUp(dataSize, 64);
	}
}

//////////////// Basic math functions  /////////////////
// CPU functions
template <class T>
void addcpu(T *h_odata, T *h_idata1, T *h_idata2, int totalSize){
	for (int i = 0; i < totalSize; i++)
		h_odata[i] = h_idata1[i] + h_idata2[i];
}
template void addcpu<int>(int *h_odata, int *h_idata1, int *h_idata2, int totalSize);
template void addcpu<float>(float *h_odata, float *h_idata1, float *h_idata2, int totalSize);
template void addcpu<double>(double *h_odata, double *h_idata1, double *h_idata2, int totalSize);

template <class T>
void addvaluecpu(T *h_odata, T *h_idata1, T h_idata2, int totalSize){
	const T b = h_idata2;
	for (int i = 0; i < totalSize; i++)
		h_odata[i] = h_idata1[i] + b;
}
template void addvaluecpu<int>(int *h_odata, int *h_idata1, int h_idata2, int totalSize);
template void addvaluecpu<float>(float *h_odata, float *h_idata1, float h_idata2, int totalSize);
template void addvaluecpu<double>(double *h_odata, double *h_idata1, double h_idata2, int totalSize);

template <class T>
void subcpu(T *h_odata, T *h_idata1, T *h_idata2, int totalSize){
	for (int i = 0; i < totalSize; i++)
		h_odata[i] = h_idata1[i] - h_idata2[i];
}
template void subcpu<int>(int *h_odata, int *h_idata1, int *h_idata2, int totalSize);
template void subcpu<float>(float *h_odata, float *h_idata1, float *h_idata2, int totalSize);
template void subcpu<double>(double *h_odata, double *h_idata1, double *h_idata2, int totalSize);

template <class T>
void multicpu(T *h_odata, T *h_idata1, T *h_idata2, int totalSize){
	for (int i = 0; i < totalSize; i++)
		h_odata[i] = h_idata1[i] * h_idata2[i];
}
template void multicpu<int>(int *h_odata, int *h_idata1, int *h_idata2, int totalSize);
template void multicpu<float>(float *h_odata, float *h_idata1, float *h_idata2, int totalSize);
template void multicpu<double>(double *h_odata, double *h_idata1, double *h_idata2, int totalSize);

template <class T>
void divcpu(T *h_odata, T *h_idata1, T *h_idata2, int totalSize){
	for (int i = 0; i < totalSize; i++)
		h_odata[i] = h_idata1[i] / h_idata2[i];
}
template void divcpu<int>(int *h_odata, int *h_idata1, int *h_idata2, int totalSize);
template void divcpu<float>(float *h_odata, float *h_idata1, float *h_idata2, int totalSize);
template void divcpu<double>(double *h_odata, double *h_idata1, double *h_idata2, int totalSize);

template <class T>
void multivaluecpu(T *h_odata, T *h_idata1, T h_idata2, int totalSize){
	for (int i = 0; i < totalSize; i++)
		h_odata[i] = h_idata1[i] * h_idata2;
}
template void multivaluecpu<int>(int *h_odata, int *h_idata1, int h_idata2, int totalSize);
template void multivaluecpu<float>(float *h_odata, float *h_idata1, float h_idata2, int totalSize);
template void multivaluecpu<double>(double *h_odata, double *h_idata1, double h_idata2, int totalSize);

extern "C"
void multicomplexcpu(fComplex *h_odata, fComplex *h_idata1, fComplex *h_idata2, int totalSize){
	fComplex a;
	fComplex b;
	for (int i = 0; i < totalSize; i++){
		a = h_idata1[i];
		b = h_idata2[i];
		h_odata[i].x = a.x*b.x - a.y*b.y;
		h_odata[i].y = a.x*b.y + a.y*b.x;
	}		
}


///// GPU functions
//add
template <class T>
void add3Dgpu(T *d_odata, T *d_idata1, T *d_idata2, int sx, int sy, int sz){
	dim3 threads(blockSize3Dx, blockSize3Dy, blockSize3Dz);
	dim3 grids(iDivUp(sx, blockSize3Dx), iDivUp(sy, blockSize3Dy), iDivUp(sz, blockSize3Dz));
	add3Dkernel<T> << <grids, threads >> >(d_odata, d_idata1, d_idata2, sx, sy, sz);
	cudaThreadSynchronize();
}
template void add3Dgpu<int>(int *d_odata, int *d_idata1, int *d_idata2, int sx, int sy, int sz);
template void add3Dgpu<float>(float *d_odata, float *d_idata1, float *d_idata2, int sx, int sy, int sz);
template void add3Dgpu<double>(double *d_odata, double *d_idata1, double *d_idata2, int sx, int sy, int sz);

// add with a single value
template <class T>
void addvaluegpu(T *d_odata, T *d_idata1, T d_idata2, int sx, int sy, int sz){
	dim3 threads(blockSize3Dx, blockSize3Dy, blockSize3Dz);
	dim3 grids(iDivUp(sx, blockSize3Dx), iDivUp(sy, blockSize3Dy), iDivUp(sz, blockSize3Dz));
	addvaluekernel<T> <<<grids, threads >>>(d_odata, d_idata1, d_idata2, sx, sy, sz);
	cudaThreadSynchronize();
}
template void addvaluegpu<int>(int *d_odata, int *d_idata1, int d_idata2, int sx, int sy, int sz);
template void addvaluegpu<float>(float *d_odata, float *d_idata1, float d_idata2, int sx, int sy, int sz);
template void addvaluegpu<double>(double *d_odata, double *d_idata1, double d_idata2, int sx, int sy, int sz);

//subtract
template <class T>
void sub3Dgpu(T *d_odata, T *d_idata1, T *d_idata2, int sx, int sy, int sz){
	dim3 threads(blockSize3Dx, blockSize3Dy, blockSize3Dz);
	dim3 grids(iDivUp(sx, blockSize3Dx), iDivUp(sy, blockSize3Dy), iDivUp(sz, blockSize3Dz));
	sub3Dkernel<T> <<<grids, threads >>>(d_odata, d_idata1, d_idata2, sx, sy, sz);
	cudaThreadSynchronize();
}
template void sub3Dgpu<int>(int *d_odata, int *d_idata1, int *d_idata2, int sx, int sy, int sz);
template void sub3Dgpu<float>(float *d_odata, float *d_idata1, float *d_idata2, int sx, int sy, int sz);
template void sub3Dgpu<double>(double *d_odata, double *d_idata1, double *d_idata2, int sx, int sy, int sz);


//multiply
template <class T>
void multi3Dgpu(T *d_odata, T *d_idata1, T *d_idata2, int sx, int sy, int sz){
	dim3 threads(blockSize3Dx, blockSize3Dy, blockSize3Dz);
	dim3 grids(iDivUp(sx, blockSize3Dx), iDivUp(sy, blockSize3Dy), iDivUp(sz, blockSize3Dz));
	multi3Dkernel<T> << <grids, threads >> >(d_odata, d_idata1, d_idata2, sx, sy, sz);
	cudaThreadSynchronize();
}
template void multi3Dgpu<int>(int *d_odata, int *d_idata1, int *d_idata2, int sx, int sy, int sz);
template void multi3Dgpu<float>(float *d_odata, float *d_idata1, float *d_idata2, int sx, int sy, int sz);
template void multi3Dgpu<double>(double *d_odata, double *d_idata1, double *d_idata2, int sx, int sy, int sz);

// multiply with a single value
template <class T>
void multivaluegpu(T *d_odata, T *d_idata1, T d_idata2, int sx, int sy, int sz){
	dim3 threads(blockSize3Dx, blockSize3Dy, blockSize3Dz);
	dim3 grids(iDivUp(sx, blockSize3Dx), iDivUp(sy, blockSize3Dy), iDivUp(sz, blockSize3Dz));
	multivaluekernel<T> <<<grids, threads >>>(d_odata, d_idata1, d_idata2, sx, sy, sz);
	cudaThreadSynchronize();
}
template void multivaluegpu<int>(int *d_odata, int *d_idata1, int d_idata2, int sx, int sy, int sz);
template void multivaluegpu<float>(float *d_odata, float *d_idata1, float d_idata2, int sx, int sy, int sz);
template void multivaluegpu<double>(double *d_odata, double *d_idata1, double d_idata2, int sx, int sy, int sz);

//multiply float complex
extern "C"
void multicomplex3Dgpu(fComplex *d_odata, fComplex *d_idata1, fComplex *d_idata2, int sx, int sy, int sz){
	dim3 threads(blockSize3Dx, blockSize3Dy, blockSize3Dz);
	dim3 grids(iDivUp(sx, blockSize3Dx), iDivUp(sy, blockSize3Dy), iDivUp(sz, blockSize3Dz));
	multicomplex3Dkernel<<<grids, threads >>>(d_odata, d_idata1, d_idata2, sx, sy, sz);
	cudaThreadSynchronize();
}

//multiply double complex
extern "C"
void multidcomplex3Dgpu(dComplex *d_odata, dComplex *d_idata1, dComplex *d_idata2, int sx, int sy, int sz){
	dim3 threads(blockSize3Dx, blockSize3Dy, blockSize3Dz);
	dim3 grids(iDivUp(sx, blockSize3Dx), iDivUp(sy, blockSize3Dy), iDivUp(sz, blockSize3Dz));
	multidcomplex3Dkernel<<<grids, threads >> >(d_odata, d_idata1, d_idata2, sx, sy, sz);
	cudaThreadSynchronize();
}

//divide
template <class T>
void div3Dgpu(T *d_odata, T *d_idata1, T *d_idata2, int sx, int sy, int sz){
	dim3 threads(blockSize3Dx, blockSize3Dy, blockSize3Dz);
	dim3 grids(iDivUp(sx, blockSize3Dx), iDivUp(sy, blockSize3Dy), iDivUp(sz, blockSize3Dz));
	div3Dkernel<T> <<<grids, threads >>>(d_odata, d_idata1, d_idata2, sx, sy, sz);
	cudaThreadSynchronize();
}
template void div3Dgpu<int>(int *d_odata, int *d_idata1, int *d_idata2, int sx, int sy, int sz);
template void div3Dgpu<float>(float *d_odata, float *d_idata1, float *d_idata2, int sx, int sy, int sz);
template void div3Dgpu<double>(double *d_odata, double *d_idata1, double *d_idata2, int sx, int sy, int sz);

//conjugation of complex
extern "C"
void conj3Dgpu(fComplex *d_odata, fComplex *d_idata, int sx, int sy, int sz){
	dim3 threads(blockSize3Dx, blockSize3Dy, blockSize3Dz);
	dim3 grids(iDivUp(sx, blockSize3Dx), iDivUp(sy, blockSize3Dy), iDivUp(sz, blockSize3Dz));
	conj3Dkernel <<<grids, threads >>>(d_odata, d_idata, sx, sy, sz);
	cudaThreadSynchronize();
}

// sumarization
// sumcpu
template <class T>
double sumcpu(T *h_idata, int totalSize){
	double mySumCPU = 0;
	for (int i = 0; i < totalSize; i++){
		mySumCPU += (double)h_idata[i];
	}
	return mySumCPU;
}
template double sumcpu<int>(int *h_idata, int totalSize);
template double sumcpu<float>(float *h_idata, int totalSize);
template double sumcpu<double>(double *h_idata, int totalSize);

// sumgpu 1
template <class T>
T sumgpu(T *d_idata, T *d_temp, T *h_temp, int totalSize){
	int gridSize = iDivUp(totalSize, blockSize);
	bool nIsPow2 = isPow2(totalSize);
	int smemSize = (blockSize <= 32) ? 2 * blockSize * sizeof(T) : blockSize * sizeof(T);
	sumgpukernel<T><<<gridSize, blockSize, smemSize >>>(
		d_idata,
		d_temp,
		totalSize,
		nIsPow2
		);
	cudaThreadSynchronize();
	cudaMemcpy(h_temp, d_temp, gridSize * sizeof(T), cudaMemcpyDeviceToHost);
	T mySumGPU = 0;
	for (int i = 0; i < gridSize; i++){
		mySumGPU += h_temp[i];
	}
	return mySumGPU;
}

template int sumgpu<int>(int *d_idata, int *d_temp, int *h_temp, int totalSize);
template float sumgpu<float>(float *d_idata, float *d_temp, float *h_temp, int totalSize);
template double sumgpu<double>(double *d_idata, double *d_temp, double *h_temp, int totalSize);

// sumgpu 2
template <class T>
double sum3Dgpu(T *d_idata, double *d_temp, double *h_temp, int sx, int sy, int sz){
	
	dim3 threads(blockSize2Dx, blockSize2Dy, 1);
	dim3 grids(iDivUp(sx, threads.x), iDivUp(sy, threads.y));
	reduceZ<T> <<<grids, threads >>>(d_idata, d_temp, sx, sy, sz); 
	cudaThreadSynchronize();
	int sxy = sx * sy;
	cudaMemcpy(h_temp, d_temp, sxy * sizeof(double), cudaMemcpyDeviceToHost); 
	double mySumGPU = 0; 
	for (int i = 0; i < sxy; i++)
		mySumGPU += h_temp[i];
	return mySumGPU;
}

template double sum3Dgpu<int>(int *d_idata, double *d_temp, double *h_temp,  int sx, int sy, int sz);
template double sum3Dgpu<float>(float *d_idata, double *d_temp, double *h_temp, int sx, int sy, int sz);
template double sum3Dgpu<double>(double *d_idata, double *d_temp, double *h_temp, int sx, int sy, int sz);

// sumgpu 3
template <class T>
T sumgpu1D(T *d_idata, T *d_temp, T *h_temp, int totalSize){
	sumgpu1Dkernel<T> <<<5, blockSize >>>(
		d_idata,
		d_temp,
		totalSize
		);
	cudaThreadSynchronize();
	int tempSize = 5 * blockSize;
	cudaMemcpy(h_temp, d_temp, tempSize * sizeof(T), cudaMemcpyDeviceToHost);
	T mySumGPU = 0;
	for (int i = 0; i < tempSize; i++){
		mySumGPU += h_temp[i];
	}
	return mySumGPU;
}

template int sumgpu1D<int>(int *d_idata, int *d_temp, int *h_temp, int totalSize);
template float sumgpu1D<float>(float *d_idata, float *d_temp, float *h_temp, int totalSize);
template double sumgpu1D<double>(double *d_idata, double *d_temp, double *h_temp, int totalSize);

// max3Dgpu: find max value and coordinates
template <class T>
T max3Dgpu(int *corXYZ, T *d_idata, int sx, int sy, int sz){
	int sx0 = 10, sy0 = 0, sz0 = 0;
	T *d_temp1 = NULL, *h_temp1 = NULL;
	int *d_temp2 = NULL, *h_temp2 = NULL;
	cudaMalloc((void **)&d_temp1, sx*sy *sizeof(T));
	cudaMalloc((void **)&d_temp2, sx*sy *sizeof(int));
	h_temp1 = (T *)malloc(sx*sy * sizeof(T));
	h_temp2 = (int *)malloc(sx*sy * sizeof(int));
	dim3 threads(blockSize2Dx, blockSize2Dy, 1);
	dim3 grids(iDivUp(sx, threads.x), iDivUp(sy, threads.y));
	maxZkernel<T> <<<grids, threads >>>(d_idata, d_temp1, d_temp2, sx, sy, sz);
	cudaThreadSynchronize();
	cudaMemcpy(h_temp1, d_temp1, sx*sy * sizeof(T), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_temp2, d_temp2, sx*sy * sizeof(int), cudaMemcpyDeviceToHost);
	cudaThreadSynchronize();

	T peakValue = -1; // h_temp1[0];
	T t;
	for (int i = 0; i < sx; i++){
		for (int j = 0; j < sy; j++){
			t = h_temp1[i + j * sx];
			if (peakValue < t){
				peakValue = t;
				sx0 = i; 
				sy0 = j;
				sz0 = h_temp2[i + j * sx];
			}
		}
	}
		
			
	corXYZ[0] = sx0; corXYZ[1] = sy0; corXYZ[2] = sz0;
	free(h_temp1); free(h_temp2);
	cudaFree(d_temp1); cudaFree(d_temp2);
	return peakValue;
}
template int max3Dgpu<int>(int *corXYZ, int *d_idata, int sx, int sy, int sz);
template float max3Dgpu<float>(int *corXYZ, float *d_idata, int sx, int sy, int sz);
template double max3Dgpu<double>(int *corXYZ, double *d_idata, int sx, int sy, int sz);

// max3Dcpu: find max value and coordinates
template <class T>
T max3Dcpu(int *corXYZ, T *h_idata, int sx, int sy, int sz){
	T peakValue = h_idata[0];
	T t;
	int sx0 = 10, sy0 = 0, sz0 = 0;
	for (int i = 0; i < sx; i++){
		for (int j = 0; j < sy; j++){
			for (int k = 0; k < sz; k++){
				t = h_idata[i + j * sx + k * sx * sy];
				if (peakValue < t){
					peakValue = t;
					sx0 = i; 
					sy0 = j;
					sz0 = k;
				}
			}
		}
	}
			
	corXYZ[0] = sx0; corXYZ[1] = sy0; corXYZ[2] = sz0;
	return peakValue;
}
template int max3Dcpu<int>(int *corXYZ, int *h_idata, int sx, int sy, int sz);
template float max3Dcpu<float>(int *corXYZ, float *h_idata, int sx, int sy, int sz);
template double max3Dcpu<double>(int *corXYZ, double *h_idata, int sx, int sy, int sz);

// max with a single value
template <class T>
void maxvalue3Dgpu(T *d_odata, T *d_idata1, T d_idata2, int sx, int sy, int sz){
	dim3 threads(blockSize3Dx, blockSize3Dy, blockSize3Dz);
	dim3 grids(iDivUp(sx, blockSize3Dx), iDivUp(sy, blockSize3Dy), iDivUp(sz, blockSize3Dz));
	maxvalue3Dgpukernel<T><<<grids, threads >>>(d_odata, d_idata1, d_idata2, sx, sy, sz);
	cudaThreadSynchronize();
}
template void maxvalue3Dgpu<int>(int *d_odata, int *d_idata1, int d_idata2, int sx, int sy, int sz);
template void maxvalue3Dgpu<float>(float *d_odata, float *d_idata1, float d_idata2, int sx, int sy, int sz);
template void maxvalue3Dgpu<double>(double *d_odata, double *d_idata1, double d_idata2, int sx, int sy, int sz);

// max with a single value
template <class T>
void maxvaluecpu(T *h_odata, T *h_idata1, T h_idata2, int totalSize){
	T a;
	const T b = h_idata2;
	for (int i = 0; i < totalSize; i++){
		a = h_idata1[i];
		h_odata[i] = (a > b) ? a : b;
	}	
}
template void maxvaluecpu<int>(int *d_odata, int *d_idata1, int d_idata2, int totalSize);
template void maxvaluecpu<float>(float *d_odata, float *d_idata1, float d_idata2, int totalSize);
template void maxvaluecpu<double>(double *d_odata, double *d_idata1, double d_idata2, int totalSize);


// maximum projection
template <class T>
void maxprojection(T *d_odata, T *d_idata, int sx, int sy, int sz, int pDirection){
	int psx, psy, psz;
	if (pDirection == 1){
		psx = sx; psy = sy; psz = sz;
	}
	if (pDirection == 2){
		psx = sz; psy = sx; psz = sy;
	}
	if (pDirection == 3){
		psx = sy; psy = sz; psz = sx;
	}
	dim3 threads(blockSize2Dx, blockSize2Dy, 1);
	dim3 grids(iDivUp(psx, threads.x), iDivUp(psy, threads.y));
	maxprojectionkernel<T> <<<grids, threads >>>(d_odata, d_idata, sx, sy, sz, psx, psy, psz, pDirection);
	cudaThreadSynchronize();
}

template void maxprojection<int>(int *d_odata, int *d_idata, int sx, int sy, int sz, int pDirection);
template void maxprojection<float>(float *d_odata, float *d_idata, int sx, int sy, int sz, int pDirection);
template void maxprojection<double>(double *d_odata, double *d_idata, int sx, int sy, int sz, int pDirection);
//Other functions
template <class T>
void changestorageordergpu(T *d_odata, T *d_idata, int sx, int sy, int sz, int orderMode){
	//orderMode
	// 1: change tiff storage order to C storage order
	//-1: change C storage order to tiff storage order
	assert(d_odata != d_idata);
	dim3 threads(blockSize3Dx, blockSize3Dy, blockSize3Dz);
	dim3 grids(iDivUp(sx, blockSize3Dx), iDivUp(sy, blockSize3Dy), iDivUp(sz, blockSize3Dz));
	changestorageordergpukernel<T><<<grids, threads>>>(d_odata, d_idata, sx, sy, sz, orderMode);
	cudaThreadSynchronize();
}
template void changestorageordergpu<int>(int *d_odata, int *d_idata, int sx, int sy, int sz, int orderMode);
template void changestorageordergpu<float>(float *d_odata, float *d_idata, int sx, int sy, int sz, int orderMode);
template void changestorageordergpu<double>(double *d_odata, double *d_idata, int sx, int sy, int sz, int orderMode);

// rotate 90/-90 degree by axis
template <class T>
void rotbyyaxis(T *d_odata, T *d_idata, int sx, int sy, int sz, int rotDirection){
	//rot direction
	// 1: rotate 90 deg around Y axis
	//-1: rotate -90 deg around Y axis
	dim3 threads(blockSize3Dx, blockSize3Dy, blockSize3Dz);
	dim3 grids(iDivUp(sx, blockSize3Dx), iDivUp(sy, blockSize3Dy), iDivUp(sz, blockSize3Dz));
	rotbyyaxiskernel<T> <<<grids, threads >>>(d_odata, d_idata, sx, sy, sz, rotDirection);
	cudaThreadSynchronize();
}
template void rotbyyaxis<int>(int *d_odata, int *d_idata, int sx, int sy, int sz, int rotDirection);
template void rotbyyaxis<float>(float *d_odata, float *d_idata, int sx, int sy, int sz, int rotDirection);
template void rotbyyaxis<double>(double *d_odata, double *d_idata, int sx, int sy, int sz, int rotDirection);

/*
// rotate any degree by y axis: matrix for affine transformation
void rot3Dbyyaxis(float *d_odata, float theta, int sx, int sz, int sx2, int sz2){
// Rotation matrix:translation (-sx2/2, -sz2/2) --> rotation--> translation back(sx/2,sy/2)
//	1	0	0	sx / 2			cos(theta)	0	sin(theta)	0		1	0	0	-sx2/2
//	0	1	0		0		*		0		1		0		0	*	0	1	0	0	
//	0	0	1	sz / 2			-sin(theta)	0	cos(theta)	0		0	0	1	-sz2/2
//	0	0	0		1				0		0		0		1		0	0	0	1
	d_odata[0] = cos(theta); d_odata[1] = 0; d_odata[2] = sin(theta);
	d_odata[3] = sx / 2 - sx2 / 2 * cos(theta) - sz2 / 2 * sin(theta);
	d_odata[4] = 0; d_odata[5] = 1; d_odata[6] = 0; d_odata[7] = 0;
	d_odata[8] = -sin(theta); d_odata[9] = 0; d_odata[10] = cos(theta);
	d_odata[11] = sz / 2 + sx2 / 2 * sin(theta) - sz2 / 2 * cos(theta);
}
*/

void p2matrix(float *m, float *x){

	m[0] = x[4], m[1] = x[5], m[2] = x[6], m[3] = x[1];
	m[4] = x[7], m[5] = x[8], m[6] = x[9], m[7] = x[2];
	m[8] = x[10], m[9] = x[11], m[10] = x[12], m[11] = x[3];

	/*
	m[0] = x[1], m[1] = x[2], m[2] = x[3], m[3] = x[4];
	m[4] = x[5], m[5] = x[6], m[6] = x[7], m[7] = x[8];
	m[8] = x[9], m[9] = x[10], m[10] = x[11], m[11] = x[12];
	*/
}
void matrix2p(float *m, float *x){
	x[0] = 0;

	x[1] = m[3], x[2] = m[7], x[3] = m[11], x[4] = m[0];
	x[5] = m[1], x[6] = m[2], x[7] = m[4], x[8] = m[5];
	x[9] = m[6], x[10] = m[8], x[11] = m[9], x[12] = m[10];

	/*
	x[1] = m[0], x[2] = m[1], x[3] = m[2], x[4] = m[3];
	x[5] = m[4], x[6] = m[5], x[7] = m[6], x[8] = m[7];
	x[9] = m[8], x[10] = m[9], x[11] = m[10], x[12] = m[11];
	*/
}


extern "C" void matrixmultiply(float * m, float *m1, float *m2){//for transformation matrix calcution only
	m[0] = m1[0] * m2[0] + m1[1] * m2[4] + m1[2] * m2[8];
	m[1] = m1[0] * m2[1] + m1[1] * m2[5] + m1[2] * m2[9];
	m[2] = m1[0] * m2[2] + m1[1] * m2[6] + m1[2] * m2[10];
	m[3] = m1[0] * m2[3] + m1[1] * m2[7] + m1[2] * m2[11] + m1[3];

	m[4] = m1[4] * m2[0] + m1[5] * m2[4] + m1[6] * m2[8];
	m[5] = m1[4] * m2[1] + m1[5] * m2[5] + m1[6] * m2[9];
	m[6] = m1[4] * m2[2] + m1[5] * m2[6] + m1[6] * m2[10];
	m[7] = m1[4] * m2[3] + m1[5] * m2[7] + m1[6] * m2[11] + m1[7];

	m[8] = m1[8] * m2[0] + m1[9] * m2[4] + m1[10] * m2[8];
	m[9] = m1[8] * m2[1] + m1[9] * m2[5] + m1[10] * m2[9];
	m[10] = m1[8] * m2[2] + m1[9] * m2[6] + m1[10] * m2[10];
	m[11] = m1[8] * m2[3] + m1[9] * m2[7] + m1[10] * m2[11] + m1[11];
	//**** 12 13 14 15 never change ****
	//no need to calculate m[12,13,14,15]:0 0 0 1

	/*
	m[0] = m1[0] * m2[0] + m1[1] * m2[4] + m1[2] * m2[8] + m1[3] * m2[12];
	m[1] = m1[0] * m2[1] + m1[1] * m2[5] + m1[2] * m2[9] + m1[3] * m2[13];
	m[2] = m1[0] * m2[2] + m1[1] * m2[6] + m1[2] * m2[10] + m1[3] * m2[14];
	m[3] = m1[0] * m2[3] + m1[1] * m2[7] + m1[2] * m2[11] + m1[3] * m2[15];

	m[4] = m1[4] * m2[0] + m1[5] * m2[4] + m1[6] * m2[8] + m1[7] * m2[12];
	m[5] = m1[4] * m2[1] + m1[5] * m2[5] + m1[6] * m2[9] + m1[7] * m2[13];
	m[6] = m1[4] * m2[2] + m1[5] * m2[6] + m1[6] * m2[10] + m1[7] * m2[14];
	m[7] = m1[4] * m2[3] + m1[5] * m2[7] + m1[6] * m2[11] + m1[7] * m2[15];

	m[8] = m1[8] * m2[0] + m1[9] * m2[4] + m1[10] * m2[8] + m1[11] * m2[12];
	m[9] = m1[8] * m2[1] + m1[9] * m2[5] + m1[10] * m2[9] + m1[11] * m2[13];
	m[10] = m1[8] * m2[2] + m1[9] * m2[6] + m1[10] * m2[10] + m1[11] * m2[14];
	m[11] = m1[8] * m2[3] + m1[9] * m2[7] + m1[10] * m2[11] + m1[11] * m2[15];

	m[12] = m1[12] * m2[0] + m1[13] * m2[4] + m1[14] * m2[8] + m1[15] * m2[12];
	m[13] = m1[12] * m2[1] + m1[13] * m2[5] + m1[14] * m2[9] + m1[15] * m2[13];
	m[14] = m1[12] * m2[2] + m1[13] * m2[6] + m1[14] * m2[10] + m1[15] * m2[14];
	m[15] = m1[12] * m2[3] + m1[13] * m2[7] + m1[14] * m2[11] + m1[15] * m2[15];
	*/
}


extern "C" void rot2matrix(float * p_out, float theta, int sx, int sy, int sz, int rotAxis){
	//p_out: 12 elements
	//theta: rotation angle
	//sx, sy, sz: images size
	////rotAxis
	// 1: rotate theta around X axis
	// 2: rotate theta around Y axis
	// 3: rotate theta around Z axis

	int sNew;
	float *p_temp, *p_temp1, *p_temp2, *p_temp3;
	p_temp = (float *)malloc(16 * sizeof(float));
	p_temp1 = (float *)malloc(16 * sizeof(float));
	p_temp2 = (float *)malloc(16 * sizeof(float));
	p_temp3 = (float *)malloc(16 * sizeof(float));
	for (int i = 0; i < 15; i++){
		p_temp[i] = p_temp1[i] = p_temp2[i] = p_temp3[i] = 0;
	}
	p_temp[15] = p_temp1[15] = p_temp2[15] = p_temp3[15] = 1; //**** 12 13 14 15 never change ****

	// matrix: p_temp1 * p_temp2 * p_temp3

	if (rotAxis == 1){//Rotate by x axis
		// Rotation matrix:translation (0, -sx2/2, -sz2/2) --> rotation--> translation back(0,sy/2,sz/2)
		//	1	0	0		0			1		0			0		0		1	0	0	0
		//	0	1	0	sx / 2		*	0	cos(theta)	sin(theta)	0	*	0	1	0	-sy2/2	
		//	0	0	1	sz / 2			0	-sin(theta)	cos(theta)	0		0	0	1	-sz2/2
		//	0	0	0		1			0		0			0		1		0	0	0	1
		p_temp1[0] = p_temp1[5] = p_temp1[10] = 1;
		p_temp1[7] = sy / 2; p_temp1[11] = sz / 2;

		p_temp2[0] = 1; p_temp2[1] = 0; p_temp2[2] = 0; p_temp2[3] = 0;
		p_temp2[4] = 0; p_temp2[5] = cos(theta); p_temp2[6] = sin(theta); p_temp2[7] = 0;
		p_temp2[8] = 0; p_temp2[9] = -sin(theta); p_temp2[10] = cos(theta); p_temp2[11] = 0;

		sNew = round(sqrt(sy * sy + sz*sz));
		p_temp3[0] = p_temp3[5] = p_temp3[10] = 1;
		p_temp3[7] = - sNew / 2; p_temp3[11] = - sNew / 2; 
	}

	if (rotAxis == 2){//Rotate by y axis

		// Rotation matrix:translation (-sx2/2, 0, -sz2/2) --> rotation--> translation back(sx/2,0,sz/2)
		//	1	0	0	sx / 2			cos(theta)	0	-sin(theta)	0		1	0	0	-sx2/2
		//	0	1	0		0		*		0		1		0		0	*	0	1	0	0	
		//	0	0	1	sz / 2			sin(theta)	0	cos(theta)	0		0	0	1	-sz2/2
		//	0	0	0		1				0		0		0		1		0	0	0	1

		p_temp1[0] = p_temp1[5] = p_temp1[10] = 1;
		p_temp1[3] = sx / 2; p_temp1[11] = sz / 2;

		p_temp2[0] = cos(theta); p_temp2[1] = 0; p_temp2[2] = -sin(theta); p_temp2[3] = 0;
		p_temp2[4] = 0; p_temp2[5] = 1; p_temp2[6] = 0; p_temp2[7] = 0;
		p_temp2[8] = sin(theta); p_temp2[9] = 0; p_temp2[10] = cos(theta); p_temp2[11] = 0;

		sNew = round(sqrt(sx * sx + sz*sz));
		p_temp3[0] = p_temp3[5] = p_temp3[10] = 1;
		p_temp3[3] = -sNew / 2; p_temp3[11] = -sNew / 2;
	}

	if (rotAxis == 3){//Rotate by z axis
		// Rotation matrix:translation (-sx2/2,-sy2/2, 0) --> rotation--> translation back(sx/2,sy/2,0)
		//	1	0	0	sx / 2			cos(theta)	sin(theta)	0	0		1	0	0	-sx2/2
		//	0	1	0	sy / 2		*	-sin(theta)	cos(theta)	0	0	*	0	1	0	-sy2/2	
		//	0	0	1		0				0			0		1	0		0	0	1	0
		//	0	0	0		1				0			0		0	1		0	0	0	1

		p_temp1[0] = p_temp1[5] = p_temp1[10] = 1;
		p_temp1[3] = sx / 2; p_temp1[7] = sy / 2;

		p_temp2[0] = cos(theta); p_temp2[1] = sin(theta); p_temp2[2] = 0; p_temp2[3] = 0;
		p_temp2[4] = -sin(theta); p_temp2[5] = cos(theta); p_temp2[6] = 0; p_temp2[7] = 0;
		p_temp2[8] = 0; p_temp2[9] = 0; p_temp2[10] = 1; p_temp2[11] = 0;

		sNew = round(sqrt(sx * sx + sy*sy));
		p_temp3[0] = p_temp3[5] = p_temp3[10] = 1;
		p_temp3[3] = -sNew / 2; p_temp3[7] = -sNew / 2;
	}


	matrixmultiply(p_temp, p_temp1, p_temp2);
	matrixmultiply(p_out, p_temp, p_temp3);

	free(p_temp);
	free(p_temp1);
	free(p_temp2);
	free(p_temp3);
}

extern "C" void dof9tomatrix(float * p_out, float *p_dof, int dofNum){
	//p_out: 12 elements
	//p_dof: 10 elements: 0 x y z alpha beta theda a b c 
	//dofNum: 3, 6, 7 or 9
	float *p_temp1, *p_temp2, *p_temp3;
	p_temp1 = (float *)malloc(16 * sizeof(float));
	p_temp2 = (float *)malloc(16 * sizeof(float));
	p_temp3 = (float *)malloc(16 * sizeof(float));
	for (int i = 0; i < 15; i++){
		p_temp1[i] = p_temp2[i] = p_temp3[i] = 0;
	}
	p_temp1[15] = p_temp2[15] = p_temp3[15] = 1; //**** 12 13 14 15 never change ****

	float x, y, z, alpha, beta, theta, a, b, c;
	if (dofNum == 3){//translation
		x = p_dof[1];
		y = p_dof[2];
		z = p_dof[3];
		alpha = 0;
		beta = 0;
		theta = 0;
		a = 1;
		b = 1;
		c = 1;
	}
	else if (dofNum == 6){//rigid body: translation, rotation
		x = p_dof[1];
		y = p_dof[2];
		z = p_dof[3];
		alpha = p_dof[4] / 57.3;
		beta = p_dof[5] / 57.3;
		theta = p_dof[6] / 57.3;
		a = 1;
		b = 1;
		c = 1;
	}
	else if (dofNum == 7){//translation,rotation, scale equelly in 3 dimemsions 
		x = p_dof[1];
		y = p_dof[2];
		z = p_dof[3];
		alpha = p_dof[4] / 57.3;
		beta = p_dof[5] / 57.3;
		theta = p_dof[6] / 57.3;
		a = p_dof[7];
		b = p_dof[7];
		c = p_dof[7];
	}
	else if (dofNum == 9){//translation,rotation,scale
		x = p_dof[1];
		y = p_dof[2];
		z = p_dof[3];
		alpha = p_dof[4] / 57.3;
		beta = p_dof[5] / 57.3;
		theta = p_dof[6] / 57.3;
		a = p_dof[7];
		b = p_dof[8];
		c = p_dof[9];
	}

	//translation
	// 1	0	0	x
	// 0	1	0	y
	// 0	0	1	z
	// 0	0	0	1
	p_temp2[3] = x;
	p_temp2[7] = y;
	p_temp2[11] = z;
	// scaling
	// a	0	0	0
	// 0	b	0	0
	// 0	0	c	0
	// 0	0	0	1
	p_temp2[0] = a;
	p_temp2[5] = b;
	p_temp2[10] = c;
	// rotating by Z axis
	// cos(alpha)	sin(alpha)	0	0
	// -sin(alpha)	cos(alpha)	0	0
	// 0			0			1	0
	// 0			0			0	1
	p_temp3[0] = cos(alpha); p_temp3[1] = sin(alpha); p_temp3[2] = 0; p_temp3[3] = 0;
	p_temp3[4] = -sin(alpha); p_temp3[5] = cos(alpha); p_temp3[6] = 0; p_temp3[7] = 0;
	p_temp3[8] = 0; p_temp3[9] = 0; p_temp3[10] = 1; p_temp3[11] = 0;
	//p_temp3[15] = 1;
	matrixmultiply(p_temp1, p_temp2, p_temp3);
	// rotating by X axis
	// 1	0			0			0
	// 0	cos(beta)	sin(beta)	0
	// 0	-sin(beta)	cos(beta)	0
	// 0	0			0			1
	p_temp3[0] = 1; p_temp3[1] = 0; p_temp3[2] = 0; p_temp3[3] = 0;
	p_temp3[4] = 0; p_temp3[5] = cos(beta); p_temp3[6] = sin(beta); p_temp3[7] = 0;
	p_temp3[8] = 0; p_temp3[9] = -sin(beta); p_temp3[10] = cos(beta); p_temp3[11] = 0;
	//p_temp3[15] = 1;
	matrixmultiply(p_temp2, p_temp1, p_temp3);
	// rotating by Y axis
	// cos(theta)	0	-sin(theta)		0
	// 0			1	0				0
	// sin(theta)	0	cos(theta)		0
	// 0			0	0				1
	p_temp3[0] = cos(theta); p_temp3[1] = 0; p_temp3[2] = -sin(theta); p_temp3[3] = 0;
	p_temp3[4] = 0; p_temp3[5] = 1; p_temp3[6] = 0; p_temp3[7] = 0;
	p_temp3[8] = sin(theta); p_temp3[9] = 0; p_temp3[10] = cos(theta); p_temp3[11] = 0;
	//p_temp3[15] = 1;
	matrixmultiply(p_out, p_temp2, p_temp3);

	free(p_temp1);
	free(p_temp2);
	free(p_temp3);
}

template <class T>
void flipPSF(T *d_odata, T *d_idata, int sx, int sy, int sz){
	dim3 threads(blockSize3Dx, blockSize3Dy, blockSize3Dz);
	dim3 grids(iDivUp(sx, blockSize3Dx), iDivUp(sy, blockSize3Dy), iDivUp(sz, blockSize3Dz));
	flipPSFkernel<T> <<<grids, threads >>>(d_odata, d_idata, sx, sy, sz);
	cudaThreadSynchronize();
}
template void flipPSF<float>(float *d_odata, float *d_idata, int sx, int sy, int sz);
template void flipPSF<double>(double *d_odata, double *d_idata, int sx, int sy, int sz);

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
	){
	assert(d_PaddedPSF != d_PSF);
	dim3 threads(blockSize3Dx, blockSize3Dy, blockSize3Dz);
	dim3 grids(iDivUp(PSFx, threads.x), iDivUp(PSFy, threads.y), iDivUp(PSFz, threads.z));
	padPSFKernel<T> <<<grids, threads >>>(
		d_PaddedPSF,
		d_PSF,
		FFTx,
		FFTy,
		FFTz,
		PSFx,
		PSFy,
		PSFz,
		PSFox,
		PSFoy,
		PSFoz
		);
	cudaThreadSynchronize();
}

template void
padPSF<float>(
float *d_PaddedPSF,
float *d_PSF,
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

template void 
padPSF<double>(
	double *d_PaddedPSF,
	double *d_PSF,
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
	){
	assert(d_PaddedStack != d_Stack);
	dim3 threads(blockSize3Dx, blockSize3Dy, blockSize3Dz);
	dim3 grids(iDivUp(FFTx, threads.x), iDivUp(FFTy, threads.y), iDivUp(FFTz, threads.z));
	padStackKernel<T> <<< grids, threads >>> (
		d_PaddedStack,
		d_Stack,
		FFTx,
		FFTy,
		FFTz,
		sx,
		sy,
		sz,
		imox,
		imoy,
		imoz
		);
	cudaThreadSynchronize();
}

template void 
padStack<float>(
	float *d_PaddedStack,
	float *d_Stack,
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

template void
padStack<double>(
double *d_PaddedStack,
double *d_Stack,
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
	){
	assert(d_PaddedStack != d_Stack);
	dim3 threads(blockSize3Dx, blockSize3Dy, blockSize3Dz);
	dim3 grids(iDivUp(sx, threads.x), iDivUp(sy, threads.y), iDivUp(sz, threads.z));
	cropStackKernel<T> <<< grids, threads >>> (
		d_PaddedStack,
		d_Stack,
		FFTx,
		FFTy,
		FFTz,
		sx,
		sy,
		sz,
		imox,
		imoy,
		imoz
		);
	cudaThreadSynchronize();
}

template void 
cropStack<float>(
	float *d_PaddedStack,
	float *d_Stack,
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

template void
cropStack<double>(
double *d_PaddedStack,
double *d_Stack,
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

extern "C" void CopyTranMatrix(float *x, int dataSize){
	cudaMemcpyToSymbol(d_aff, x, dataSize, 0, cudaMemcpyHostToDevice);// copy host affine matrix to device const
	/*
	cudaMemcpyFromSymbolAsync(x, d_aff, dataSize, 0, cudaMemcpyDeviceToHost);
	for (int i = 0; i < 12; i++){
		printf(" %d th value: %f\n", i, x[i]);
	}
	*/
}

template <class T>
void cudacopyhosttoarray(cudaArray *d_Array, cudaChannelFormatDesc channelDesc, T *h_idata, int sx, int sy, int sz){
	cudaMemcpy3DParms copyParams = { 0 };
	copyParams.srcPtr = make_cudaPitchedPtr((void*)h_idata, sx*sizeof(T), sx, sy);
	copyParams.dstArray = d_Array;
	copyParams.extent = make_cudaExtent(sx, sy, sz);
	copyParams.kind = cudaMemcpyHostToDevice;
	cudaMemcpy3D(&copyParams);
	cudaThreadSynchronize();
}

template void
cudacopyhosttoarray<float>(cudaArray *d_Array, cudaChannelFormatDesc channelDesc, float *h_idata, int sx, int sy, int sz);
template void
cudacopyhosttoarray<unsigned short>(cudaArray *d_Array, cudaChannelFormatDesc channelDesc, unsigned short *h_idata, int sx, int sy, int sz);

extern "C"
void cudacopydevicetoarray(cudaArray *d_Array, cudaChannelFormatDesc channelDesc, float *d_idata, int sx, int sy, int sz){
	cudaMemcpy3DParms copyParams = { 0 };
	copyParams.srcPtr = make_cudaPitchedPtr((void*)d_idata, sx*sizeof(float), sx, sy);
	copyParams.dstArray = d_Array;
	copyParams.extent = make_cudaExtent(sx, sy, sz);
	copyParams.kind = cudaMemcpyDeviceToDevice;
	cudaMemcpy3D(&copyParams);
	cudaThreadSynchronize();
}


extern "C" void BindTexture(
	cudaArray *d_Array,
	cudaChannelFormatDesc channelDesc
	){
	// set texture parameters
	tex.addressMode[0] = cudaAddressModeWrap;
	tex.addressMode[1] = cudaAddressModeWrap;
	tex.addressMode[2] = cudaAddressModeWrap;
	tex.filterMode = cudaFilterModeLinear;
	tex.normalized = false; //NB coordinates in [0,1]
	// Bind the array to the texture
	cudaBindTextureToArray(tex, d_Array, channelDesc);
	cudaThreadSynchronize();
}

extern "C" void BindTexture16(
	cudaArray *d_Array,
	cudaChannelFormatDesc channelDesc
	){
	// set texture parameters
	tex.addressMode[0] = cudaAddressModeWrap;
	tex.addressMode[1] = cudaAddressModeWrap;
	tex.addressMode[2] = cudaAddressModeWrap;
	tex.filterMode = cudaFilterModeLinear;
	tex.normalized = false; //NB coordinates in [0,1]
	// Bind the array to the texture
	cudaBindTextureToArray(tex16, d_Array, channelDesc);
	cudaThreadSynchronize();
}

extern "C" void UnbindTexture(
	){
	cudaUnbindTexture(tex);
	cudaThreadSynchronize();
}

extern "C" void UnbindTexture16(
	){
	cudaUnbindTexture(tex16);
	cudaThreadSynchronize();
}

extern "C" void AccessTexture(float x, float y,float z){
	dim3 threads(2, 2, 2);
	accesstexturekernel <<<1, threads >>>(x, y, z);
	cudaThreadSynchronize();
}

template <class T> 
void affineTransform(
	T *d_Stack, 
	int sx, 
	int sy, 
	int sz,
	int sx2,
	int sy2,
	int sz2
	){
	dim3 threads(blockSize3Dx, blockSize3Dy, blockSize3Dz);
	dim3 grid(iDivUp(sx, threads.x), iDivUp(sy, threads.y), iDivUp(sz, threads.z));
	affinetransformkernel<T><<<grid, threads >>>(
		d_Stack,
		sx,
		sy,
		sz,
		sx2,
		sy2,
		sz2
		);
	cudaThreadSynchronize();
}

template void 
affineTransform<float>(
	float *d_Stack,
	int sx,
	int sy,
	int sz,
	int sx2,
	int sy2,
	int sz2
	);
template void
affineTransform<unsigned short>(
unsigned short *d_Stack,
int sx,
int sy,
int sz,
int sx2,
int sy2,
int sz2
);

double corrfunc(float *d_s, // source stack
	float *d_sqr,
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
	){
	//copy aff to GPU const
	cudaMemcpyToSymbol(d_aff, aff, 12 * sizeof(float), 0, cudaMemcpyHostToDevice);// copy host affine matrix to device const
	dim3 threads(blockSize3Dx, blockSize3Dy, blockSize3Dz);
	dim3 grids(iDivUp(sx, threads.x), iDivUp(sy, threads.y), iDivUp(sz, threads.z));
	corrkernel <<<grids, threads>>>( // StackB is texture, trans matrix is const
		d_s,
		d_sqr,
		d_corr,
		sx,
		sy,
		sz,
		sx2,
		sy2,
		sz2
		);
	cudaThreadSynchronize();
	double corrSum = sum3Dgpu(d_corr, d_temp, h_temp, sx, sy, sz); 
	double sqrSum = sum3Dgpu(d_sqr, d_temp, h_temp, sx, sy, sz);
	return (corrSum / sqrt(sqrSum));
}

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
	){
	//copy aff to GPU const
	cudaMemcpyToSymbol(d_aff, aff, 12 * sizeof(float), 0, cudaMemcpyHostToDevice);// copy host affine matrix to device const
	dim3 threads(blockSize2Dx, blockSize2Dy, 1);
	dim3 grids(iDivUp(sx, threads.x), iDivUp(sy, threads.y));
	corrkernel2 <<<grids, threads >>>( // StackB is texture, trans matrix is const
		d_s,
		d_temp1,
		d_temp2,
		sx,
		sy,
		sz,
		sx2,
		sy2,
		sz2
		);
	cudaThreadSynchronize();
	int sxy = sx * sy;
	double sqrSum = 0, corrSum = 0;
	cudaMemcpy(h_temp, d_temp1, sxy * sizeof(double), cudaMemcpyDeviceToHost);
	for (int i = 0; i < sxy; i++)
		sqrSum += h_temp[i];
	cudaMemcpy(h_temp, d_temp2, sxy * sizeof(double), cudaMemcpyDeviceToHost);
	for (int i = 0; i < sxy; i++)
		corrSum += h_temp[i];

	return (corrSum / sqrt(sqrSum));
}

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
	){
	//copy aff to GPU const
	cudaMemcpyToSymbol(d_aff, aff, 12 * sizeof(float), 0, cudaMemcpyHostToDevice);// copy host affine matrix to device const
	dim3 threads(blockSize2Dx, blockSize2Dy, 1);
	dim3 grids(iDivUp(sx, threads.x), iDivUp(sy, threads.y));
	corrkernel2<<<grids, threads>>>( // StackB is texture, trans matrix is const
		d_s,
		d_temp1,
		d_temp2,
		sx,
		sy,
		sz,
		sx2,
		sy2,
		sz2
		);
	cudaThreadSynchronize();
	int sxy = sx * sy;
	double sqrSum = 0, corrSum = 0;
	if (sxy > 100000){
		sqrSum = sumgpu1D(d_temp1, d_temp3, h_temp, sxy);
		corrSum = sumgpu1D(d_temp2, d_temp3, h_temp, sxy);
	}
	else{
		cudaMemcpy(h_temp, d_temp1, sxy * sizeof(double), cudaMemcpyDeviceToHost);
		for (int i = 0; i < sxy; i++)
			sqrSum += h_temp[i];
		cudaMemcpy(h_temp, d_temp2, sxy * sizeof(double), cudaMemcpyDeviceToHost);
		for (int i = 0; i < sxy; i++)
			corrSum += h_temp[i];
	}

	return (corrSum / sqrt(sqrSum));
}


extern "C"
void cudacopyhosttoarray2D(cudaArray *d_Array, cudaChannelFormatDesc channelDesc, float *h_idata, int totalSize){
	cudaMemcpyToArray(d_Array, 0, 0, h_idata, totalSize*sizeof(float), cudaMemcpyHostToDevice);
	cudaThreadSynchronize();
}

extern "C" void BindTexture2D(
	cudaArray *d_Array,
	cudaChannelFormatDesc channelDesc
	){
	// set texture parameters
	tex2D1.addressMode[0] = cudaAddressModeWrap;
	tex2D1.addressMode[1] = cudaAddressModeWrap;
	tex2D1.filterMode = cudaFilterModeLinear;
	tex2D1.normalized = false;    // access with normalized texture coordinates

	// Bind the array to the texture
	cudaBindTextureToArray(tex2D1, d_Array, channelDesc);
}

extern "C" void UnbindTexture2D(
	){
	cudaUnbindTexture(tex2D1);
}

extern "C"
void affineTransform2D(float *d_t, int sx, int sy, int sx2, int sy2){
	dim3 threads(blockSize2Dx, blockSize2Dy, 1);
	dim3 grids(iDivUp(sx, threads.x), iDivUp(sy, threads.y));
	affineTransform2Dkernel <<<grids, threads >>>(d_t, sx, sy, sx2, sy2);
	cudaThreadSynchronize();
}

double corrfunc2D(float *d_s, // source stack
	float *d_sqr,
	float *d_corr,
	float *aff,
	float *h_temp,
	int sx,
	int sy,
	int sx2,
	int sy2
	){
	//copy aff to GPU const
	cudaMemcpyToSymbol(d_aff, aff, 6 * sizeof(float), 0, cudaMemcpyHostToDevice);// copy host affine matrix to device const
	dim3 threads(blockSize2Dx, blockSize2Dy, 1);
	dim3 grids(iDivUp(sx, threads.x), iDivUp(sy, threads.y));
	corr2Dkernel <<<grids, threads >>>( // StackB is texture, trans matrix is const
		d_s,
		d_sqr,
		d_corr,
		sx,
		sy,
		sx2,
		sy2
		);
	cudaThreadSynchronize();
	int sxy = sx*sy;
	cudaMemcpy(h_temp, d_corr, sxy * sizeof(float), cudaMemcpyDeviceToHost);
	double corrSum = sumcpu(h_temp, sxy);
	cudaMemcpy(h_temp, d_sqr, sxy * sizeof(float), cudaMemcpyDeviceToHost);
	double sqrSum = sumcpu(h_temp, sxy);
	return (corrSum / sqrt(sqrSum));

}
extern "C" bool checkmatrix(float *m){
	bool tMatrix = true;
	float scaleLow = 0.7, scaleUp = 1.4, scaleSumLow = 2.4, scaleSumUp = 4;
	if (m[0]<scaleLow || m[0]>scaleUp || m[5]<scaleLow || m[5]>scaleUp || m[10]<scaleLow || m[10]>scaleUp){
		tMatrix = false;
	}
		
	if ((m[0] + m[5] + m[10]) < scaleSumLow || (m[0] + m[5] + m[10]) > scaleSumUp){
		tMatrix = false;
		
	}
	return tMatrix;
}

///// CPU interpolation
float lerp(float x, float x1, float x2, float q00, float q01) {
	return ((x2 - x) / (x2 - x1)) * q01 + ((x - x1) / (x2 - x1)) * q00;
}

float bilerp(float x, float y, float x1, float x2, float y1, float y2, float q11, float q12, float q21, float q22) {
	float r1 = lerp(x, x1, x2, q11, q12);
	float r2 = lerp(x, x1, x2, q21, q22);

	return lerp(y, y1, y2, r1, r2);
}

float trilerp(float x, float y, float z, float x1, float x2, float y1, float y2, float z1, float z2, 
	float q111, float q112, float q121, float q122, float q211, float q212, float q221, float q222) {
	float r1 = bilerp(x, y, x1, x2, y1, y2, q111, q112, q121, q122);
	float r2 = bilerp(x, y, x1, x2, y1, y2, q211, q212, q221, q222);
	return lerp(z, z1, z2, r1, r2);
}

float ilerp(float x, float x1, float x2, float q00, float q01) {
	return (x2 - x) * q00 + (x - x1) * q01;
}

float ibilerp(float x, float y, float x1, float x2, float y1, float y2, float q11, float q12, float q21, float q22) {
	float r1 = ilerp(x, x1, x2, q11, q12);
	float r2 = ilerp(x, x1, x2, q21, q22);

	return ilerp(y, y1, y2, r1, r2);
}

float itrilerp(float x, float y, float z, float x1, float x2, float y1, float y2, float z1, float z2,
	float q111, float q112, float q121, float q122, float q211, float q212, float q221, float q222) {
	float r1 = ibilerp(x, y, x1, x2, y1, y2, q111, q112, q121, q122);
	float r2 = ibilerp(x, y, x1, x2, y1, y2, q211, q212, q221, q222);
	return ilerp(z, z1, z2, r1, r2);
}

float ilerp2(float dx1, float dx2, float q00, float q01) {
	return dx2 * q00 + dx1 * q01;
}

float ibilerp2(float dx1, float dx2, float dy1, float dy2, float q11, float q12, float q21, float q22) {
	float r1 = ilerp2(dx1, dx2, q11, q12);
	float r2 = ilerp2(dx1, dx2, q21, q22);

	return ilerp2(dy1, dy2, r1, r2);
}

float itrilerp2(float dx1, float dx2, float dy1, float dy2, float dz1, float dz2,
	float q111, float q112, float q121, float q122, float q211, float q212, float q221, float q222) {
	float r1 = ibilerp2(dx1, dx2, dy1, dy2, q111, q112, q121, q122);
	float r2 = ibilerp2(dx1, dx2, dy1, dy2, q211, q212, q221, q222);
	return ilerp2(dz1, dz2, r1, r2);
}

//output[sz-k-1][j][i] = input[i][j][k]
//d_odata[(sz - k - 1)*sx*sy + j*sx + i] = d_idata[i*sy*sz + j*sz + k];
double corrfunccpu(float *h_s,
	float *h_t,// source stack
	float *aff,
	int sx,
	int sy,
	int sz,
	int sx2,
	int sy2,
	int sz2
	){
	double sqrSum = 0, corrSum = 0;
	int x1, y1, z1, x2, y2, z2;
	float q1, q2, q3, q4, q5, q6, q7, q8;
	float s, t;
	int sxy = sx*sy, sxy2 = sx2*sy2;
	for (int i = 0; i < sx; i++){
		for (int j = 0; j < sy; j++){
			for (int k = 0; k < sz; k++){
				float ix = (float)i;
				float iy = (float)j;
				float iz = (float)k;
				float tx = aff[0] * ix + aff[1] * iy + aff[2] * iz + aff[3];
				float ty = aff[4] * ix + aff[5] * iy + aff[6] * iz + aff[7];
				float tz = aff[8] * ix + aff[9] * iy + aff[10] * iz + aff[11];
				x1 = floor(tx); y1 = floor(ty); z1 = floor(tz);
				x2 = x1 + 1; y2 = y1 + 1; z2 = z1 + 1;
				if ((x1 >= 0) && (y1 >= 0) && (z1 >= 0) && (x2 < sx2) && (y2 < sy2) && (z2 < sz2)){
					// [k*sy*sx + j*sx + i]
					q1 = h_t[z1*sxy2 + y1*sx2 + x1];
					q2 = h_t[z1*sxy2 + y1*sx2 + x2];
					q3 = h_t[z1*sxy2 + y2*sx2 + x1];
					q4 = h_t[z1*sxy2 + y2*sx2 + x2];
					q5 = h_t[z2*sxy2 + y1*sx2 + x1];
					q6 = h_t[z2*sxy2 + y1*sx2 + x2];
					q7 = h_t[z2*sxy2 + y2*sx2 + x1];
					q8 = h_t[z2*sxy2 + y2*sx2 + x2];
					t = itrilerp(tx, ty, tz, x1, x2, y1, y2, z1, z2, q1, q2, q3, q4, q5, q6, q7, q8);
				}
				else
					t = 0;
				s = h_s[k*sxy + j*sx + i];

				sqrSum += (double)t*t;
				corrSum += (double)s*t;
			}
		}
	}
	return (corrSum / sqrt(sqrSum));
}


double corrfunccpu3(float *h_s,
	float *h_t,// source stack
	float *aff,
	int sx,
	int sy,
	int sz,
	int sx2,
	int sy2,
	int sz2
	){
	const float r0 = aff[0], r1 = aff[1], r2 = aff[2], r3 = aff[3], r4 = aff[4], r5= aff[5],
		r6 = aff[6], r7 = aff[7], r8 = aff[8], r9 = aff[9], r10 = aff[10], r11 = aff[11];

	double sqrSum = 0, corrSum = 0;
	float ix, iy, iz, tx, ty, tz;
	int x1, y1, z1, x2, y2, z2;
	float dx1, dy1, dz1, dx2, dy2, dz2;
	float q1, q2, q3, q4, q5, q6, q7, q8;
	float s, t;
	int syz = sy*sz, syz2 = sy2*sz2, x1syz2, x2syz2, y1sz2, y2sz2;
	for (int i = 0; i < sx; i++){
		ix = (float)i;
		for (int j = 0; j < sy; j++){
			iy = (float)j;
			for (int k = 0; k < sz; k++){
				iz = (float)k;
				
				tx = r0 * ix + r1 * iy + r2 * iz + r3;
				ty = r4 * ix + r5 * iy + r6 * iz + r7;
				tz = r8 * ix + r9 * iy + r10 * iz + r11;
				
				x1 = (int)tx; y1 = (int)ty; z1 = (int)tz;
				x2 = x1 + 1; y2 = y1 + 1; z2 = z1 + 1;

				dx1 = tx - (float)x1; dy1 = ty - (float)y1; dz1 = tz - (float)z1;
				dx2 = 1 - dx1; dy2 = 1 - dy1; dz2 = 1 - dz1;
				if (x1 >= 0 && y1 >= 0 && z1 >= 0 && x2 < sx2 && y2 < sy2 && z2 < sz2){
					// [i*sy*sz + j*sz + k]
					x1syz2 = x1*syz2;
					x2syz2 = x2*syz2;
					y1sz2 = y1*sz2;
					y2sz2 = y2*sz2;

					q1 = h_t[x1syz2 + y1sz2 + z1];
					q2 = h_t[x2syz2 + y1sz2 + z1];
					q3 = h_t[x1syz2 + y2sz2 + z1];
					q4 = h_t[x2syz2 + y2sz2 + z1];
					q5 = h_t[x1syz2 + y1sz2 + z2];
					q6 = h_t[x2syz2 + y1sz2 + z2];
					q7 = h_t[x1syz2 + y2sz2 + z2];
					q8 = h_t[x2syz2 + y2sz2 + z2];
					//t = itrilerp2(dx1, dx2, dy1, dy2, dz1, dz2, q1, q2, q3, q4, q5, q6, q7, q8);
					//t = itrilerp(tx, ty, tz, x1, x2, y1, y2, z1, z2, q1, q2, q3, q4, q5, q6, q7, q8);
					t = dz2*(dy2*dx2*q1 + dy2*dx1*q2 + dy1*dx2*q3 + dy1*dx1*q4) + dz1*(dy2*dx2*q5 + dy2*dx1*q6 + dy1*dx2*q7 + dy1*dx1*q8);
					//t = 1;

				}
				else
					t = 0;
				s = h_s[i*syz + j*sz + k];

				sqrSum += (double)t*t;
				corrSum += (double)s*t;
			}
		}
	}
	return (corrSum / sqrt(sqrSum));
}
double corrfunccpu2(float *h_s,
	float *h_t,// source stack
	float *aff,
	int sx,
	int sy,
	int sz,
	int sx2,
	int sy2,
	int sz2
	){
	const float r0 = aff[0], r1 = aff[1], r2 = aff[2], r3 = aff[3], r4 = aff[4], r5 = aff[5],
		r6 = aff[6], r7 = aff[7], r8 = aff[8], r9 = aff[9], r10 = aff[10], r11 = aff[11];

	double sqrSum = 0, corrSum = 0;
	float ix, iy, iz, tx, ty, tz;
	int x1, y1, z1, x2, y2, z2;
	float dx1, dy1, dz1, dx2, dy2, dz2;
	float q1, q2, q3, q4, q5, q6, q7, q8;
	float s, t;
	int sxy = sx*sy, sxy2 = sx2*sy2, z1sxy2, z2sxy2, y1sx2, y2sx2;
	for (int i = 0; i < sx; i++){
		ix = (float)i;
		for (int j = 0; j < sy; j++){
			iy = (float)j;
			for (int k = 0; k < sz; k++){
				iz = (float)k;

				tx = r0 * ix + r1 * iy + r2 * iz + r3;
				ty = r4 * ix + r5 * iy + r6 * iz + r7;
				tz = r8 * ix + r9 * iy + r10 * iz + r11;

				x1 = (int)tx; y1 = (int)ty; z1 = (int)tz;
				x2 = x1 + 1; y2 = y1 + 1; z2 = z1 + 1;

				dx1 = tx - (float)x1; dy1 = ty - (float)y1; dz1 = tz - (float)z1;
				dx2 = 1 - dx1; dy2 = 1 - dy1; dz2 = 1 - dz1;
				if (x1 >= 0 && y1 >= 0 && z1 >= 0 && x2 < sx2 && y2 < sy2 && z2 < sz2){
					// [i*sy*sz + j*sz + k]
					z1sxy2 = z1*sxy2;
					z2sxy2 = z2*sxy2;
					y1sx2 = y1*sx2;
					y2sx2 = y2*sx2;

					q1 = h_t[z1sxy2 + y1sx2 + x1];
					q2 = h_t[z1sxy2 + y1sx2 + x2];
					q3 = h_t[z1sxy2 + y2sx2 + x1];
					q4 = h_t[z1sxy2 + y2sx2 + x2];
					q5 = h_t[z2sxy2 + y1sx2 + x1];
					q6 = h_t[z2sxy2 + y1sx2 + x2];
					q7 = h_t[z2sxy2 + y2sx2 + x1];
					q8 = h_t[z2sxy2 + y2sx2 + x2];
					//t = itrilerp2(dx1, dx2, dy1, dy2, dz1, dz2, q1, q2, q3, q4, q5, q6, q7, q8);
					//t = itrilerp(tx, ty, tz, x1, x2, y1, y2, z1, z2, q1, q2, q3, q4, q5, q6, q7, q8);
					t = dz2*(dy2*dx2*q1 + dy2*dx1*q2 + dy1*dx2*q3 + dy1*dx1*q4) + dz1*(dy2*dx2*q5 + dy2*dx1*q6 + dy1*dx2*q7 + dy1*dx1*q8);
					//t = 1;

				}
				else
					t = 0;
				s = h_s[k*sxy + j*sx + i];

				sqrSum += (double)t*t;
				corrSum += (double)s*t;
			}
		}
	}
	return (corrSum / sqrt(sqrSum));
}

void affinetransformcpu(float *h_s,
	float *h_t,// source stack
	float *aff,
	int sx,
	int sy,
	int sz,
	int sx2,
	int sy2,
	int sz2
	){
	float ix, iy, iz, tx, ty, tz;
	int x1, y1, z1, x2, y2, z2;
	float dx1, dy1, dz1, dx2, dy2, dz2;
	float q1, q2, q3, q4, q5, q6, q7, q8;
	float t;
	int sxy = sx*sy, sxy2 = sx2*sy2, z1sxy2, z2sxy2, y1sx2, y2sx2;
	int syz = sy*sz, syz2 = sy2*sz2;
	for (int i = 0; i < sx; i++){
		ix = (float)i;
		for (int j = 0; j < sy; j++){
			iy = (float)j;
			for (int k = 0; k < sz; k++){
				iz = (float)k;
				tx = aff[0] * ix + aff[1] * iy + aff[2] * iz + aff[3];
				ty = aff[4] * ix + aff[5] * iy + aff[6] * iz + aff[7];
				tz = aff[8] * ix + aff[9] * iy + aff[10] * iz + aff[11];
				x1 = (int)tx; y1 = (int)ty; z1 = (int)tz;
				x2 = x1 + 1; y2 = y1 + 1; z2 = z1 + 1;

				dx1 = tx - (float)x1; dy1 = ty - (float)y1; dz1 = tz - (float)z1;
				dx2 = 1 - dx1; dy2 = 1 - dy1; dz2 = 1 - dz1;
				if (x1 >= 0 && y1 >= 0 && z1 >= 0 && x2 < sx2 && y2 < sy2 && z2 < sz2){
					// [i*sy*sz + j*sz + k]
					z1sxy2 = z1*sxy2;
					z2sxy2 = z2*sxy2;
					y1sx2 = y1*sx2;
					y2sx2 = y2*sx2;

					q1 = h_t[z1sxy2 + y1sx2 + x1];
					q2 = h_t[z1sxy2 + y1sx2 + x2];
					q3 = h_t[z1sxy2 + y2sx2 + x1];
					q4 = h_t[z1sxy2 + y2sx2 + x2];
					q5 = h_t[z2sxy2 + y1sx2 + x1];
					q6 = h_t[z2sxy2 + y1sx2 + x2];
					q7 = h_t[z2sxy2 + y2sx2 + x1];
					q8 = h_t[z2sxy2 + y2sx2 + x2];
					t = itrilerp2(dx1, dx2, dy1, dy2, dz1, dz2, q1, q2, q3, q4, q5, q6, q7, q8);
					//t = itrilerp(tx, ty, tz, x1, x2, y1, y2, z1, z2, q1, q2, q3, q4, q5, q6, q7, q8);
					//t = dz2*(dy2*dx2*q1 + dy2*dx1*q2 + dy1*dx2*q3 + dy1*dx1*q4) + dz1*(dy2*dx2*q5 + dy2*dx1*q6 + dy1*dx2*q7 + dy1*dx1*q8);
				}
				else
					t = 0;
				h_s[k*sxy + j*sx + j] = t;
			}
		}
	}
}
