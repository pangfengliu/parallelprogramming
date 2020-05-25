CC = gcc
CFLAGS = 
LIBS = -lOpenCL -lm

all:  vectorAdd-nofetchC-cl vectorAdd-nofetchC-finish-cl vectorAdd-fetchC-cl \
	get-global-id-cl get-global-local-id-cl \
	matrixMul-cl matrixMul-time-cl matrixMul-time-use-cl matrixMul-time-copy-cl matrixMul-time-copy-local-cl


%-cl: %.c
	$(CC) $(CFLAGS) $< -o $@ $(LIBS)


