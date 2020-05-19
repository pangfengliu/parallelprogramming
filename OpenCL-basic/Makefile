CC = gcc
CFLAGS = -std=c99
CL_TARGET = getPlatformID-cl getPlatformInfo-cl getDeviceID-cl getDeviceInfo-cl vectorAdd-cl
PDF_TARGET = buffer.pdf command_queue.pdf context.pdf deviceID.pdf kernel.pdf NDRange.pdf program.pdf readbuffer.pdf
LIBS = -lOpenCL -lm
all: $(CL_TARGET) $(PDF_TARGET)
%-cl: %.c
	$(CC) $(CFLAGS) $< -o $@ $(LIBS)
%.pdf: %.dot
	dot -Tpdf $< -o $@
tar:
	tar -cvf PP-OpenCL-basic.tar *.c kernel.cl Makefile
clean:
	rm -f $(CL_TARGET)

