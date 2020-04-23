CC = gcc
CFLAGS = -std=c99 -msse2 -m64 -rdynamic
INCDIRS = -I/opt/AMDAPPSDK-3.0-0-Beta/include -I/opt/AMDAPPSDK-3.0-0-Beta/include/SDKUtil 
LIBS = /opt/AMDAPPSDK-3.0-0-Beta/lib/x86_64/libOpenCL.so -lrt
LINKOPT = -Wl,-rpath,/opt/AMDAPPSDK-3.0-0-Beta/lib/x86_64

all:  vectorAdd-nofetchC-cl vectorAdd-nofetchC-finish-cl \
	get-global-id-cl get-global-local-id-cl \
	matrixMul-cl matrixMul-time-cl matrixMul-time-use-cl matrixMul-time-copy-cl matrixMul-time-copy-local-cl


%-cl: %.c
	$(CC) $(CFLAGS) $(INCDIRS) $< $(LIBS) $(LINKOPT) -o $@


