CC = gcc
CFLAGS =
LIBS = -lOpenCL -lm

all: matrixMul-time-copy-local-multidevice-cl \
	vectorAdd-groupsize-cl \
	vectorAdd-dependency-cl vectorAdd-dependency-correct-cl vectorAdd-dependency-clfinish-cl

%-cl: %.c
	$(CC) $(CFLAGS) $< -o $@ $(LIBS)

vectorAdd-groupsize.png: vectorAdd-groupsize.dat
	gnuplot vectorAdd-groupsize.gplt

clean:
	rm *-cl




