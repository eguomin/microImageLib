#!/bin/sh

# set library path
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:./bin/linux

# set parameters
pathOut="./data/results_batch/"
pathSPIMA="./data/SPIMA/"
pathSPIMB="./data/SPIMB/"
nameA="SPIMA_"
nameB="SPIMB_"
filePSFA="./data/PSFA.tif"
filePSFB="./data/PSFB.tif"
fileiTmx="balabala"

# Set full 34 or 36 manatary parameters as required
# use help,  "spimFusionBatch -h" for more information

# run app with parameters: 
./bin/linux/spimFusionBatch $pathOut $pathSPIMA $pathSPIMB $nameA $nameB 0 2 1 0 \
0.1625 0.1625 1.0 0.1625 0.1625 1.0 2 -1 0 $fileiTmx 0.0001 3000 1 1 \
$filePSFA $filePSFA 10 0 0 1 0 1 16 1 0