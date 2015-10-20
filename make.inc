CC = icc
CXX = icpc

ARCH = ar
ARCHFLAGS = cr
RANLIB = ranlib

CFLAGS = -O3 -openmp -mavx

LIBGSKNN = $(GSKNN_DIR)/lib/librnn.a

LDFLAGS = -I$(GSKNN_DIR)/include -I$(GSKNN_DIR)/kernels/$(GSKNN_ARCH) -I/$(GSKNN_MKL_DIR)/include

LDLIBS = $(LIBGSKNN) -lpthread -lm -openmp -mkl=parallel -Werror -Wall -pedantic
