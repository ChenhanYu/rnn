ifeq ($(GSKNN_USE_INTEL),true)
include $(GSKNN_DIR)/make.intel.inc
else
include $(GSKNN_DIR)/make.gnu.inc
endif


FRAME_CC_SRC=  \
							 frame/dgsrnn.c \
							 frame/rnn_util.c \
							 frame/dgsrnn_ref.c \

FRAME_CPP_SRC= \
							 frame/dgsrnn_ref_stl.cpp \

KERNEL_SRC=    \
							 kernels/$(GSKNN_ARCH)/rnn_heap.c \
							 kernels/$(GSKNN_ARCH)/rnn_rank_k_asm_d8x4.c \
							 kernels/$(GSKNN_ARCH)/rnn_r_int_d8x4_row.c \
							 kernels/$(GSKNN_ARCH)/sq2nrm_asm_d8x4.c \
							 kernels/$(GSKNN_ARCH)/rnn_r_1norm_int_d8x4_row.c \
							 kernels/$(GSKNN_ARCH)/rnn_rank_k_abs_int_d8x4.c \

GSKNN_OBJ=$(FRAME_CC_SRC:.c=.o) $(FRAME_CPP_SRC:.cpp=.o) $(KERNEL_SRC:.c=.o)

all: $(LIBGSKNN) TESTGSKNN

TESTGSKNN: $(LIBGSKNN)
	cd $(GSKNN_DIR)/test && $(MAKE) && cd $(GSKNN_DIR)

$(LIBGSKNN): $(GSKNN_OBJ)
	$(ARCH) $(ARCHFLAGS) $@ $(GSKNN_OBJ)
	$(RANLIB) $@


# ---------------------------------------------------------------------------
# Object files compiling rules
# ---------------------------------------------------------------------------
%.o: %.c 
	$(CC) $(CFLAGS) -c $< -o $@ $(LDFLAGS)

%.o: %.cpp
	$(CXX) $(CFLAGS) -c $< -o $@ $(LDFLAGS)
# ---------------------------------------------------------------------------

clean:
	rm $(GSKNN_OBJ)
	cd $(GSKNN_DIR)/test && $(MAKE) clean && cd $(GSKNN_DIR)
