# microImageLib
A collection of 3D image processing functions with GPU implementation: TIFF reading/writing, affine transformation, max projection, registration, deconvolution etc. While all GPU related functions have been tested, a few of CUP-only options are not functional and still under trouble-shooting.

# - microImageLib_vs2017 
Source code and configurations for Visual Studio 2017 with CUDA v9.0 on a Windows 7/10 system. The TIFF library and FFTW library are required and and already included as lib dependencies.

# - microImageLib_Linux (This is a pre-release version)
Source code and configurations for Linux OS
1) The compiling has been tested on Ubuntu 18.04 with CUDA v10;
2) TIFF library and FFTW library are required:
sudo apt-get install libfftw3-dev libfftw3-doc
sudo apt-get install libtiff5
