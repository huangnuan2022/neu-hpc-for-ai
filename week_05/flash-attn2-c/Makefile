# Minimal production Makefile
CC ?= gcc
CFLAGS ?= -O3 -std=c11 -Wall -Wextra -march=native
LDFLAGS ?= -lm

BIN_DIR := bin
SRC_DIR := src
TARGET  := $(BIN_DIR)/flash_attn2_seq
SRCS    := $(SRC_DIR)/flash_attn2_seq.c

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

all: $(BIN_DIR) $(TARGET)

$(TARGET): $(SRCS) | $(BIN_DIR)
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

run: all
	$(TARGET)

clean:
	rm -rf $(BIN_DIR)

.PHONY: all run clean
