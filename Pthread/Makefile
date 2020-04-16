CC = gcc
CFLAGS = -Wall -std=c99 -DVERBOSE
LIBS = -lpthread

.PHONY: all exe pthreadexe
all: hello queen hello-pthread hello-pthread-noexit hello-pthread-correct queen-pthread queen-pthread-sum queen-pthread-sum-join queen-pthread-sum-join-correct queen-pthread-sum-join-race queen-pthread-sum-join-race-mutex

%: %.c
	$(CC) $(CFLAGS) $< -o $@ $(LIBS)
