CC = icc
CXX = icpc

ARCH = ar
ARCHFLAGS = cr
RANLIB = ranlib

ifeq ($(GSKNN_USE_BLAS),true)
CFLAGS = -O3 -fopenmp -mavx -fPIC -DUSE_BLAS
else
CFLAGS = -O3 -fopenmp -mavx -fPIC
endif

LIBGSKNN = $(GSKNN_DIR)/lib/libgsknn.a
SHAREDLIBGSKNN = $(GSKNN_DIR)/lib/libgsknn.so

LDFLAGS = -I$(GSKNN_DIR)/include -I$(GSKNN_DIR)/kernels/$(GSKNN_ARCH) -I/$(GSKNN_MKL_DIR)/include

LDLIBS = $(LIBGSKNN) -lpthread -lm -openmp -mkl=parallel -Werror -Wall -pedantic
