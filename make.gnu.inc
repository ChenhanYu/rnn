CC = gcc
CXX = g++

ARCH = ar
ARCHFLAGS = cr
RANLIB = ranlib

CFLAGS = -O3 -fopenmp -mavx

LIBGSKNN = $(GSKNN_DIR)/lib/librnn.a

LDFLAGS = -I$(GSKNN_DIR)/include -I$(GSKNN_DIR)/kernels/$(GSKNN_ARCH)

LDLIBS = $(LIBGSKNN) -lpthread -lm -fopenmp -lblas
