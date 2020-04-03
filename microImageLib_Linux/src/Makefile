
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

CXXFLAGS  = -Wall -std=c++11 -O2  -fPIC
CXXFLAGS += -I$(CUDA_INC_DIR) -I$(INC_DIR)

NVFLAGS = -Xptxas -v,-dlcm=cg 
NVFLAGS += -ftz=true --compiler-options '-fPIC' --compiler-options="-O2" -I$(INC_DIR)
CUDA_LIB_FLAG = -L$(CUDA_ROOT)/lib64 
LDFLAGS = $(CUDA_LIB_FLAG) $(NVFLAGS)

LDLIBS = -ltiff -lcudart -lcufft -lfftw3f

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $^ -o $@ 

%.o: %.c
	$(CC) $(CXXFLAGS) -c $^ -o $@ 

%.o: %.cu
	$(NVCC)  -c $^ $(LDFLAGS) $(LDLIBS)

all: libapi.so

libapi.so: apifunc.o powell.o  subfunc.o
	$(CC) -o ../bin/$@ $^  -shared -Wl,-soname,$@ $(CUDA_LIB_FLAG) $(LDLIBS)

all: reg3D decon_sv decon_dv
reg3D: reg3D.o apifunc.o powell.o  subfunc.o
	nvcc -o ../bin/$@ $^ $(CUDA_LIB_FLAG) $(LDLIBS)

decon_sv: decon_sv.o apifunc.o powell.o  subfunc.o 
	nvcc -o ../bin/$@ $^ $(CUDA_LIB_FLAG) $(LDLIBS)

decon_dv: decon_dv.o apifunc.o powell.o  subfunc.o
	nvcc -o ../bin/$@ $^ $(CUDA_LIB_FLAG) $(LDLIBS)

clean:
	$(RM) *.o ../bin/* *.so
