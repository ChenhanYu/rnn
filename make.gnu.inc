CC = gcc
CXX = g++

ARCH = ar
ARCHFLAGS = cr
RANLIB = ranlib

CFLAGS = -O3 -fopenmp -mavx

LIBRNN = $(RNN_DIR)/lib/librnn.a

LDFLAGS = -I$(RNN_DIR)/include

LDLIBS = $(LIBRNN) -lpthread -lm -fopenmp -lblas
