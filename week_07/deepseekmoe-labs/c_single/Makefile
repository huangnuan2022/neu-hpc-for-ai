APP=moe_single
SRC=src/moe_single.c
CC?=cc
CFLAGS?=-O3 -std=c11 -Wall -Wextra -march=native
LDFLAGS?=-lm
all: $(APP)
$(APP): $(SRC)
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)
clean:
	$(RM) $(APP)
