CC = gcc
CFLAGS = -std=c99

all: 	hello-uni hello-omp \
hello-get-num-omp \
hello-get-num-all-omp \
hello-get-num-all-set-omp \
hello-get-num-all-set-show-id-omp \
for-omp \
for-private-omp \
for-private-first-omp \
for-private-first-last-omp \
queen-uni queen-omp \
queen-private-omp \
queen-private-sum-uni queen-private-sum-omp \
queen-private-sum-array-uni queen-private-sum-array-omp \
queen-private-sum-num-uni queen-private-sum-num-omp \
queen-private-sum-reduction-uni queen-private-sum-reduction-omp \
queen-private-sum-race-uni queen-private-sum-race-omp \
queen-private-sum-race-critical-uni queen-private-sum-race-critical-omp \
sudoku-omp sudoku-uni


%-omp: %.c
	$(CC) -fopenmp $(CFLAGS) $< -o $@ 

%-uni: %.c
	$(CC) $(CFLAGS) $< -o $@ 

clean:
	rm *-uni *-omp
