CXX ?= c++
CXXFLAGS ?= -std=c++17 -O3 -Wall -Wextra -pedantic

BUILD_DIR := build
BIN := $(BUILD_DIR)/inference_harness
SRC := src/inference_harness.cpp

.PHONY: all test bench clean

all: $(BIN)

$(BIN): $(SRC)
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $< -o $@

test: $(BIN)
	$(BIN) attention --seq 64 --dim 16 --shards 2 --iters 1
	$(BIN) moe --tokens 64 --dim 16 --hidden 32 --experts 4 --shards 2 --iters 1

bench: $(BIN)
	python3 scripts/run_benchmarks.py

clean:
	rm -rf $(BUILD_DIR)
