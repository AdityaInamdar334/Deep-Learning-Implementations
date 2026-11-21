CC ?= cc
CFLAGS = -std=c11 -O2 -Wall -Wextra -Wpedantic
LDLIBS = -lm
all: xor
xor: main.c
	$(CC) $(CFLAGS) main.c -o $@ $(LDLIBS)
test: xor
	./xor test
run: xor
	./xor
finite: xor
	./xor finite
clean:
	rm -f xor
.PHONY: all test run finite clean
