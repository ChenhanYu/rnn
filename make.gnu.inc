CC = gcc
CXX = g++

ARCH = ar
ARCHFLAGS = cr
RANLIB = ranlib

ifeq ($(GSKNN_USE_BLAS),true)
CFLAGS = -O3 -fopenmp -mavx -fPIC -DUSE_BLAS
LDLIBS = $(LIBGSKNN) -lpthread -lm -fopenmp -lblas
else
CFLAGS = -O3 -fopenmp -mavx -fPIC
LDLIBS = $(LIBGSKNN) -lpthread -lm -fopenmp
endif

LIBGSKNN = $(GSKNN_DIR)/lib/libgsknn.a
SHAREDLIBGSKNN = $(GSKNN_DIR)/lib/libgsknn.so

LDFLAGS = -I$(GSKNN_DIR)/include -I$(GSKNN_DIR)/kernels/$(GSKNN_ARCH)
