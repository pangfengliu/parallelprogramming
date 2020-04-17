CC = gcc
CFLAGS = -std=c99

all: 	for-static-chunk-4-omp for-static-chunk-1-omp \
	for-dynamic-chunk-1-omp \
	for-dynamic-chunk-1-reverse-omp \
	for-guided-chunk-1-omp \
	for-dynamic-chunk-1-reverse-omp for-guided-chunk-4-reverse-omp \
	for-runtime-omp \
	2loops-omp 2loops-stdc99-omp 2loops-stdc99-both-omp \
	2loops-stdc99-both-wrong-omp 2loops-stdc99-combined-omp \
	2loops-stdc99-combined-single-omp \
	rand-omp \
	prime-uni prime-omp \
	prime-inner-omp prime-inner-nowait-omp

%-omp: %.c
	$(CC) -fopenmp $(CFLAGS) $< -lm -o $@ 

%-uni: %.c
	$(CC) $(CFLAGS) $< -lm -o $@

%.pdf: %.tex
	pdflatex $<

tar:
	tar -cvf PP-OpenMP-intermediate.tar Makefile *.c

clean:
	rm *-uni *-omp *.aux *.log *.nav *.snm *.toc *.vrb *.out *~

