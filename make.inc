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
#LDLIBS = $(LIBRNN) -lpthread -lm -openmp -Wl,--start-group ${TACC_MKL_DIR}/lib/intel64/libmkl_intel_lp64.a ${TACC_MKL_DIR}/lib/intel64/libmkl_core.a ${TACC_MKL_DIR}/lib/intel64/libmkl_intel_thread.a -Wl,--end-group -lpthread -lm

