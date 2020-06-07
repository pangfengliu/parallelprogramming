CC = nvcc
CFLAGS =
CUDA_TARGET = globalID-cu

all: $(CUDA_TARGET)

%-cu: %.cu
	$(CC) $(CFLAGS) $< -o $@
clean:
	rm -f $(CUDA_TARGET) *~

