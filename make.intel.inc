CC = icc
CXX = icpc

ARCH = ar
ARCHFLAGS = cr
RANLIB = ranlib

CFLAGS = -O3 -openmp -mavx

LIBRNN = $(RNN_DIR)/lib/librnn.a

#LDFLAGS = -I$(RNN_DIR)/include -I$(TACC_MKL_DIR)/include
LDFLAGS = -I$(RNN_DIR)/include -I/opt/intel/mkl/include

#LDLIBS = $(LIBRNN) -lpthread -lm -openmp -mkl=sequential -Werror -Wall -pedantic
LDLIBS = $(LIBRNN) -lpthread -lm -openmp -mkl=parallel -Werror -Wall -pedantic
