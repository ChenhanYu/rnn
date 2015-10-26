ifeq ($(GSKNN_USE_INTEL),true)
include $(GSKNN_DIR)/make.intel.inc
else
include $(GSKNN_DIR)/make.gnu.inc
endif


FRAME_CC_SRC=  \
							 frame/gsknn.c \
							 frame/gsknn_heap.c \
							 frame/gsknn_util.c \
							 frame/gsknn_ref.c \

#FRAME_CC_SRC_S= \
#							 frame/gsknn_heap.c \

FRAME_CPP_SRC= \
							 frame/gsknn_ref_stl.cpp \

KERNEL_SRC=    \
							 kernels/$(GSKNN_ARCH)/gsknn_heapselect_int_d4.c \
							 kernels/$(GSKNN_ARCH)/rnn_rank_k_asm_d8x4.c \
							 kernels/$(GSKNN_ARCH)/rnn_r_int_d8x4_row.c \
							 kernels/$(GSKNN_ARCH)/sq2nrm_asm_d8x4.c \
							 kernels/$(GSKNN_ARCH)/rnn_r_1norm_int_d8x4_row.c \
							 kernels/$(GSKNN_ARCH)/rnn_rank_k_abs_int_d8x4.c \

#KERNEL_SRC_S=  \
#							 kernels/$(GSKNN_ARCH)/knn_rank_k_ref_s8x8.c \
#							 kernels/$(GSKNN_ARCH)/knn_r_ref_s8x8_row.c \

GSKNN_OBJ=$(FRAME_CC_SRC:.c=.o) $(FRAME_CPP_SRC:.cpp=.o) $(KERNEL_SRC:.c=.o) $(FRAME_CC_SRC_S:.c=.os) $(KERNEL_SRC_S:.c=.os)

all: $(LIBGSKNN) $(SHAREDLIBGSKNN) TESTGSKNN

TESTGSKNN: $(LIBGSKNN)
	cd $(GSKNN_DIR)/test && $(MAKE) && cd $(GSKNN_DIR) $(LDFLAGS)

$(LIBGSKNN): $(GSKNN_OBJ)
	$(ARCH) $(ARCHFLAGS) $@ $(GSKNN_OBJ)
	$(RANLIB) $@

$(SHAREDLIBGSKNN): $(GSKNN_OBJ)
	$(CC) $(CFLAGS) -shared -o $@ $(GSKNN_OBJ) $(LDLIBS)

# ---------------------------------------------------------------------------
# Object files compiling rules
# ---------------------------------------------------------------------------
%.o: %.c 
	$(CC) $(CFLAGS) -c $< -o $@ $(LDFLAGS)

%.os: %.c 
	$(CC) $(CFLAGS) -DKNN_PREC_SINGLE -c $< -o $@ $(LDFLAGS)

%.o: %.cpp
	$(CXX) $(CFLAGS) -c $< -o $@ $(LDFLAGS)
# ---------------------------------------------------------------------------

clean:
	rm $(GSKNN_OBJ)
	cd $(GSKNN_DIR)/test && $(MAKE) clean && cd $(GSKNN_DIR)
