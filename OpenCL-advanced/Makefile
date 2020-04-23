CC = gcc
CFLAGS = -std=c99 -msse2 -m64 -rdynamic
INCDIRS = -I/opt/AMDAPPSDK-3.0-0-Beta/include -I/opt/AMDAPPSDK-3.0-0-Beta/include/SDKUtil 
LIBS = /opt/AMDAPPSDK-3.0-0-Beta/lib/x86_64/libOpenCL.so -lrt
LINKOPT = -Wl,-rpath,/opt/AMDAPPSDK-3.0-0-Beta/lib/x86_64

all: matrixMul-time-copy-local-multidevice-cl \
	vectorAdd-groupsize-cl \
	vectorAdd-dependency-cl vectorAdd-dependency-correct-cl vectorAdd-dependency-clfinish-cl


%-cl: %.c
	$(CC) $(CFLAGS) $(INCDIRS) $< $(LIBS) $(LINKOPT) -o $@

vectorAdd-groupsize.png: vectorAdd-groupsize.dat
	gnuplot vectorAdd-groupsize.gplt


