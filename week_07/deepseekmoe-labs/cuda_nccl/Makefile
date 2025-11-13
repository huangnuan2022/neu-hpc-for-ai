APP=moe_nccl
SRC=src/moe_nccl.cu
NVCC?=nvcc
SM?= -gencode arch=compute_70,code=sm_70 \
     -gencode arch=compute_75,code=sm_75 \
     -gencode arch=compute_80,code=sm_80 \
     -gencode arch=compute_86,code=sm_86 \
     -gencode arch=compute_89,code=sm_89
CFLAGS?=-O3 -std=c++17 -Xcompiler -fPIC -lineinfo -use_fast_math -Xptxas=-v
LDFLAGS?=-lnccl
all: $(APP)
$(APP): $(SRC)
	$(NVCC) $(CFLAGS) $(SM) $< -o $@ $(LDFLAGS)
clean:
	$(RM) $(APP)
