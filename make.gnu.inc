CC = gcc
CXX = g++

ARCH = ar
ARCHFLAGS = cr
RANLIB = ranlib

ifeq ($(GSKNN_USE_BLAS),true)
CFLAGS = -O3 -fopenmp -mavx -DUSE_BLAS
else
CFLAGS = -O3 -fopenmp -mavx
endif

LIBGSKNN = $(GSKNN_DIR)/lib/librnn.a

LDFLAGS = -I$(GSKNN_DIR)/include -I$(GSKNN_DIR)/kernels/$(GSKNN_ARCH)

LDLIBS = $(LIBGSKNN) -lpthread -lm -fopenmp -lblas
