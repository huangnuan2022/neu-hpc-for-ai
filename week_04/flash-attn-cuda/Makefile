NVCC ?= nvcc
CXXFLAGS := -O3 -std=c++17
DMAX ?= 128
BR   ?= 64
BC   ?= 64

all: flash_attn_cuda

flash_attn_cuda: flash_attn_cuda.cu
	$(NVCC) $(CXXFLAGS) -DDMAX=$(DMAX) -DBR=$(BR) -DBC=$(BC) $< -o $@

clean:
	rm -f flash_attn_cuda *.o
.PHONY: all clean
