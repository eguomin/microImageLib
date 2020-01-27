# microImageLib
A collection of 3D image processing functions with GPU implementation: TIFF reading/writing, affine transformation, max projection, registration, deconvolution etc. 
1) the TIFF library is required: http://www.libtiff.org/;
2) while all GPU related functions have been tested, a few of CUP-only options are not functional and is still under trouble-shooting.

The setup is currently configured for Visual Studio 2017 with CUDA v9.0 on a Windows 7/10 system, a more flexible version compatible with Linux is coming around...
