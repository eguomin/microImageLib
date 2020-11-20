#!/bin/sh

# set library path
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:./bin/linux

# set parameters
fileSPIMA="./data/SPIMA/SPIMA_0.tif"
fileSPIMB="./data/SPIMB/SPIMB_0.tif"
filePSFA="./data/PSFA.tif"
filePSFB="./data/PSFB.tif"
fileDecon="./data/results/Decon_0.tif"
fileRegA="./data/results/RegA_0.tif"
fileRegB="./data/results/RegB_0.tif"
fileoTmx="./data/results/RegB_0.tmx"

# other parameters as default...
# use help,  "spimFusion -h" for more information

# run app with parameters: 
./bin/linux/spimFusion -i1 $fileSPIMA -i2 $fileSPIMB -fp1 $filePSFA -fp2 $filePSFB -o $fileDecon \
-it 10 -cOFF -imgrot -1 -dev 0 -verbON -oreg1 $fileRegA -oreg2 $fileRegB -otmx $fileoTmx
