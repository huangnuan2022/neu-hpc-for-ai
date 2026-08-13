CXX ?= c++
CXXFLAGS ?= -std=c++17 -O3 -Wall -Wextra -pedantic

BUILD_DIR := build
BIN := $(BUILD_DIR)/inference_harness
SRC := src/inference_harness.cpp
VENV := .venv
PYTHON := $(VENV)/bin/python
PYTEST := $(VENV)/bin/pytest
SERVING_STAMP := $(VENV)/.serving-installed

.PHONY: all test bench serving-setup serving-test serving-demo test-all clean

all: $(BIN)

$(BIN): $(SRC)
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $< -o $@

test: $(BIN)
	$(BIN) attention --seq 64 --dim 16 --shards 2 --iters 1
	$(BIN) moe --tokens 64 --dim 16 --hidden 32 --experts 4 --shards 2 --iters 1

bench: $(BIN)
	python3 scripts/run_benchmarks.py

$(SERVING_STAMP): pyproject.toml
	python3 -m venv $(VENV)
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -e '.[test]'
	@touch $(SERVING_STAMP)

serving-setup: $(SERVING_STAMP)

serving-test: $(SERVING_STAMP)
	$(PYTEST)

serving-demo: $(SERVING_STAMP)
	$(VENV)/bin/llm-serving-demo

test-all: test serving-test

clean:
	rm -rf $(BUILD_DIR)
