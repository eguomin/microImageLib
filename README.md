microImageLib
=============

## Overview
A collection of 3D image processing functions and applications with GPU implementation, originally developped for fast 3D microscopic image processing [[1]](#1).
- libapi: a dynamic-link library including TIFF reading/writing, affine transformation, 2D/3D maximum intensity projections (MIP), registration, deconvolution etc. 
- checkGPUDevice: inqury GPU devices and CUDA version on the PC.
- reg3D: 3D registration based on affine transformation.
- deconSingleView: 3D single-view deconvolution.
- deconDualView: 3D dual-view joint deconvolution.
- spimFusion: 3D dual-view image fusion designed for diSPIM data, incorporating image rotation, interporation, registration and joint deconvolution.
- spimFusionBatch: a batch processing version of spimFusion for diSPIM time-lapse data, additionally including functions for generating 2D/3D maximum intensity projections (MIP) of the deconvolved images (functionally same with the ImageJ diSPIMFusion package).

## System Requirements and Compiling

- Windows 10 or Linux OS. 
- NVIDIA GPU
- CUDA toolkit
- TIFF and FFTW libraries

### Tested Environments:

**1. Windows 10**

The TIFF library and FFTW libraries are required and already included in this repo as lib dependencies. Users only need to open the Visual Studio solution file:

```posh
microImageLib.sln
```
 > a) First build solution for libApi project;
 
 > b) Then build solutions for other projects.

All projects are already configured for Visual Studio 2017 with CUDA 10.0. The compiling generates binary library and applications in the folder `./bin/win`. 

**2. Ubuntu 18.04 LTS**

Tested Linux PC has been installed with CUDA 10.0 and a Makefile has been configured in the `./src` folder. Before compiling the code, FFTW and TIFF libraries can be got by using commands:
```posh
sudo apt-get install libfftw3-dev libfftw3-doc
sudo apt-get install libtiff5
```
Then get to the source code folder and run the compiling:
```posh
cd path/to/microImageLib/src
make
```
The compiling generates binary library and applications in the folder `./bin/linux`.

To clean the built results, use option `-clean` to remove built objects or `-cleanAll` to remove both built objects and binary outputs:
```posh
make -clean
(or) make -cleanAll
```

The compiled binaries (for both Windows 10 and Ubuntu 18.04 LTS) along with the library dependencies are included [here](https://www.dropbox.com/sh/czn4kwzwcgy0s3x/AADipfEsUSwuCsEBg8P7wc4_a?dl=0), for users who want to use the compiled versions.

## Usage
Users can use command with option `-h` or `-help` to find the introduction and manual for each application, e.g.
```posh
spimFusion -h
```

Users can also download [a test dataset with a few example scripts](https://www.dropbox.com/sh/czn4kwzwcgy0s3x/AADipfEsUSwuCsEBg8P7wc4_a?dl=0). The example scripts are a group of *cmd* or *shell* commands that invoke the binary applications with configurations for the test dataset. To run the scripts, users need to open the command terminal and get to the directory of the scripts, i.e., the folder `./cudaLib`.

1) For Windows PC, run any of the `cmd_xx.bat` scripts, e.g.
    ```posh
    cmd_spimFusionBatch.bat
    ```
2) For Linux PC, run any of the `sh_xx.sh` scripts, e.g.
    ```posh
    sh sh_spimFusionBatch.sh
    ```
    In case the Linux PC does not have the CUDA or FFTW installed, users will need to add the dependencies directory to the path variable *LD_LIBRARY_PATH* so as to use the libraries provided within the compiled package, e.g., use command:
    ```posh
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:./bin/linux
    sh sh_spimFusionBatch.sh
    ```
    
Please cite our paper [[1]](#1) if you use the code provided in this repository.

## Reference

<a id="1">[1]</a>
Min Guo, *et al*.
"[Rapid image deconvolution and multiview fusion for optical microscopy](https://doi.org/10.1038/s41587-020-0560-x)." Nature Biotechnology 38.11 (2020): 1337-1346.
