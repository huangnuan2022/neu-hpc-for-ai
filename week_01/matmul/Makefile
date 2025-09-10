CC := gcc
CFLAGS := -O3 -march=native -Wall -Wextra -pthread
INCLUDES := -Iinclude

SRC := src/mat.c src/util.c src/matmul_serial.c src/matmul_threads.c tests/test_matmul.c
BIN := matmul

all: $(BIN)

$(BIN): $(SRC)
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ $(SRC)

clean:
	rm -f $(BIN)
