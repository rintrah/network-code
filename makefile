# Makefile for building C stuff with GSL

CFLAGS=-g -Wall -I/usr/local/include/gsl -std=c99
LDFLAGS=-lgsl -lgslcblas -lm  -lGL -lGLU -lglut
CC=gcc

%: %.c
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

clean:
		rm -f *~ *.o core a.out 
