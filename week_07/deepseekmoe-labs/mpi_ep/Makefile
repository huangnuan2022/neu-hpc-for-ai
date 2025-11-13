APP=moe_mpi
SRC=src/moe_mpi.c
MPICC?=mpicc
CFLAGS?=-O3 -std=c11 -Wall -Wextra -march=native
LDFLAGS?=-lm
all: $(APP)
$(APP): $(SRC)
	$(MPICC) $(CFLAGS) $^ -o $@ $(LDFLAGS)
clean:
	$(RM) $(APP)
