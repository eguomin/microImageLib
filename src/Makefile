
# ** set environment for CUDA, libtiff and fftw libraries if necessary
# export PATH=$PATH:/usr/local/cuda/bin
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/lib:/usr/lib/x86_64-linux-gnu

CUDA_ROOT = /usr/local/cuda

ifeq ($(CUDA_ROOT),)
  $(error Cannot find CUDA_ROOT)
endif

CUDA_INC_DIR = $(CUDA_ROOT)/include
CUDA_LIB_FLAG = -L$(CUDA_ROOT)/lib64
NVCC = $(CUDA_ROOT)/bin/nvcc -g -std=c++11 

INC_DIR = ../include
BIN_DIR = ../bin/linux

CXXFLAGS  = -Wall -std=c++11 -O2  -fPIC
CXXFLAGS += -I$(CUDA_INC_DIR) -I$(INC_DIR) -I$(BIN_DIR)

NVFLAGS = -Xptxas -v,-dlcm=cg 
NVFLAGS += -ftz=true --compiler-options '-fPIC' --compiler-options="-O2" -I$(INC_DIR)
LDFLAGS = $(CUDA_LIB_FLAG) $(NVFLAGS)

LDLIBS = -ltiff -lcudart -lcufft -lfftw3f


%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $^ -o $@ 

%.o: %.c
	$(CC) $(CXXFLAGS) -c $^ -o $@ 

%.o: %.cu
	$(NVCC)  -c $^ $(LDFLAGS) $(LDLIBS)

all: libapi.so

libapi.so: api_reg.o api_decon.o api_powell.o api_subfunc.o apifunc.o
	$(CC) -o $(BIN_DIR)/$@ $^  -shared -Wl,-soname,$@ $(CUDA_LIB_FLAG) $(LDLIBS)

all: checkGPUDevice reg3D deconSingleView deconDualView spimFusion spimFusionBatch

checkGPUDevice: check_gpu.o api_reg.o api_decon.o api_powell.o api_subfunc.o apifunc.o
	nvcc -o $(BIN_DIR)/$@ $^ $(CUDA_LIB_FLAG) $(LDLIBS)

reg3D: reg3D.o api_reg.o api_decon.o api_powell.o api_subfunc.o apifunc.o
	nvcc -o $(BIN_DIR)/$@ $^ $(CUDA_LIB_FLAG) $(LDLIBS) 

deconSingleView: decon_sv.o api_reg.o api_decon.o api_powell.o api_subfunc.o apifunc.o
	nvcc -o $(BIN_DIR)/$@ $^ $(CUDA_LIB_FLAG) $(LDLIBS)

deconDualView: decon_dv.o api_reg.o api_decon.o api_powell.o api_subfunc.o apifunc.o
	nvcc -o $(BIN_DIR)/$@ $^ $(CUDA_LIB_FLAG) $(LDLIBS)

spimFusion: spim_fusion.o api_reg.o api_decon.o api_powell.o api_subfunc.o apifunc.o
	nvcc -o $(BIN_DIR)/$@ $^ $(CUDA_LIB_FLAG) $(LDLIBS)

spimFusionBatch: spim_fusion_batch.o api_reg.o api_decon.o api_powell.o api_subfunc.o apifunc.o
	nvcc -o $(BIN_DIR)/$@ $^ $(CUDA_LIB_FLAG) $(LDLIBS)


clean:
	$(RM) *.o *.so

cleanAll:
	$(RM) *.o *.so $(BIN_DIR)/*
