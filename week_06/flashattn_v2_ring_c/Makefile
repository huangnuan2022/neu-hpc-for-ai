# Simple portable Makefile
# Usage:
#   make           # build fa2_ring
#   make clean
#   OMP=0 make     # build without OpenMP

APP := fa2_ring
SRC := src/fa2_ring.c

CC  ?= gcc
OMP ?= 1

CFLAGS ?= -O3 -std=c11 -Wall -Wextra -march=native
LDFLAGS := -lm

ifeq ($(OMP),1)
  CFLAGS += -fopenmp
endif

$(APP): $(SRC)
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

clean:
	$(RM) $(APP)
