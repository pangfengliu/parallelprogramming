CC = gcc
CFLAGS = -std=c99

source = 2for.c 2for-nowait.c life.c pi.c pi-critical.c pi-critical-small.c pi-array.c pi-reduction.c prime.c prime-inner.c prime-inner-nowait.c assign.c assign-private.c assign-heap.c

all:	PP-OpenMP-advanced.pdf \
	prime-uni prime-omp \
	prime-inner-omp prime-inner-nowait-omp \
	2for-omp 2for-nowait-omp \
	pi-omp pi-critical-omp pi-critical-small-omp pi-array-omp pi-reduction-omp\
	life-uni life-omp \
	assign-uni assign-omp assign-private-omp assign-heap-omp

run-pi:
	./pi-omp
	./pi-critical-omp
	./pi-critical-small-omp
	./pi-array-omp
	./pi-reduction-omp

run-assign:
	time ./assign-uni
	time ./assign-omp
	time ./assign-private-omp
	time ./assign-heap-omp

run-prime:
	time ./prime-uni 400000000
	time ./prime-omp 400000000
	time ./prime-inner-omp 400000000
	time ./prime-inner-nowait-omp 400000000

%-omp: %.c
	$(CC) -fopenmp $(CFLAGS) $< -lm -o $@ 

%-uni: %.c
	$(CC) $(CFLAGS) $< -lm -o $@

%.pdf: %.tex $(source)
	pdflatex $<

tar:
	tar -cvf PP-OpenMP-advanced.tar Makefile *.c

clean:
	rm *-uni *-omp *.aux *.log *.nav *.snm *.toc *.vrb *.out *~

