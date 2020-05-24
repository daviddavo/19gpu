


CC=gcc

CFLAGS=-O3 -std=c99

PGCC=pgcc -Minfo -fast -acc -ta=nvidia -tp=nehalem

LIB=-lm

NAME=nw
TESTARGS=10 10

default: host gpu

.PHONY: test clean

host:
	$(CC) $(CFLAGS) $(NAME).c -o $(NAME).host.exe $(LIB)

gpu: pgi


pgi:
	$(PGCC) $(NAME).c -o $(NAME).pgi.exe $(LIB)

test: SHELL:=/bin/bash
test: default
#	paste -d'|' <(./$(NAME).host.exe 1) <(./$(NAME).pgi.exe 1) | column -s '|' -t -o " | "
	paste -d'|' <(./$(NAME).host.exe $(TESTARGS)) <(./$(NAME).pgi.exe $(TESTARGS)) | column -s '|' -t -o " | "

clean: 
	rm -f _ *.exe
