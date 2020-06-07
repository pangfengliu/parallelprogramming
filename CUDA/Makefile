CC = nvcc
CFLAGS = -Xcompiler "-Wall"
CUDA_TARGET =  globalID-cu globalID2D-cu matrixMul-cu matrixMultiming-cu matrixMulprivate-cu matrixMulprivatetiming-cu

all: $(CUDA_TARGET)

%-cu: %.cu
	$(CC) $(CFLAGS) $< -o $@
clean:
	rm -f $(CUDA_TARGET) *~

