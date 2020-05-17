CC = gcc
CFLAGS = -std=c99
CL_TARGET = getPlatformID-cl getPlatformInfo-cl getDeviceID-cl getDeviceInfo-cl vectorAdd-cl
LIBS = -lOpenCL -lm

all: $(CL_TARGET)

%-cl: %.c
	$(CC) $(CFLAGS) $< -o $@ $(LIBS)

opencltest: opencltest.cpp
	g++  $(CFLAGS) $< -o $@ $(LIBS)

tar:
	tar -cvf PP-OpenCL-basic.tar *.c kernel.cl Makefile

clean:
	rm -f $(CL_TARGET)
	rm -f opencltest
