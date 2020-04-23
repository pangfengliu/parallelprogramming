CC = gcc
CFLAGS = -std=c99 -msse2 -m64 -rdynamic
INCDIRS = -I/opt/AMDAPPSDK-3.0-0-Beta/include -I/opt/AMDAPPSDK-3.0-0-Beta/include/SDKUtil 
LIBS = /opt/AMDAPPSDK-3.0-0-Beta/lib/x86_64/libOpenCL.so -lrt
LINKOPT = -Wl,-rpath,/opt/AMDAPPSDK-3.0-0-Beta/lib/x86_64

all: getPlatformID-cl getPlatformInfo-cl getDeviceID-cl getDeviceInfo-cl \
	vectorAdd-cl 

%-cl: %.c
	$(CC) $(CFLAGS) $(INCDIRS) $< $(LIBS) $(LINKOPT) -o $@

opencltest: opencltest.cpp
	g++  $(CFLAGS) $(INCDIRS) $< $(LIBS) $(LINKOPT) -o $@

tar: 
	tar -cvf PP-OpenCL-basic.tar *.c kernel.cl Makefile

