
NVCC_DIR := $(shell  dirname `which nvcc`)
CUDA_ROOT = $(NVCC_DIR)/../

ifeq ($(CUDA_ROOT),)
  $(error Cannot find CUDA_ROOT)
endif

CUDA_INC_DIR = $(CUDA_ROOT)/include
CUDA_LIB_FLAG = -L$(CUDA_ROOT)/lib64
NVCC = $(CUDA_ROOT)/bin/nvcc -g -std=c++11 

INC_DIR = ../../include 

CXXFLAGS  = -Wall -std=c++11 -O2  -fPIC
CXXFLAGS += -I$(CUDA_INC_DIR) -I$(INC_DIR)

NVFLAGS = -Xptxas -v,-dlcm=cg 
NVFLAGS += -ftz=true --compiler-options '-fPIC' --compiler-options="-O2" -I$(INC_DIR)
CUDA_LIB_FLAG = -L$(CUDA_ROOT)/lib64 
LDFLAGS = $(CUDA_LIB_FLAG) $(NVFLAGS)

LDLIBS = -ltiff -lcudart -lcufft

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $^ -o $@ 

%.o: %.c
	$(CC) $(CXXFLAGS) -c $^ -o $@ 

%.o: %.cu
	$(NVCC)  -c $^ $(LDFLAGS) $(LDLIBS)

all: libaim.so

libaim.so: apifunc.o brent.o  f1dim.o  linmin.o  mnbrak.o  nrutil.o  powell.o  subfunc.o
	$(CC) -o ../../bin/$@ $^  -shared -Wl,-soname,$@ $(CUDA_LIB_FLAG) $(LDLIBS)

clean:
	$(RM) *.o bin/* *.so
