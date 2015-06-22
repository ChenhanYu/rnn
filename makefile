include $(RNN_DIR)/make.inc


FRAME_CC_SRC=  \
							 frame/dgsrnn.c \
							 frame/rnn_util.c \
							 frame/dgsrnn_ref.c \
							 frame/rnn_heap.c \
							 frame/rnn_mergesort.c \

FRAME_CPP_SRC= \
							 frame/dgsrnn_ref_stl.cpp \
							 frame/dgsrnn_directKQuery.cpp \

KERNEL_SRC=    \
							 kernels/x86_64/sandybridge/rnn_int_d8x4.c \
							 kernels/x86_64/sandybridge/rnn_int_d8x4_var2.c \
							 kernels/x86_64/sandybridge/rnn_asm_d8x4.c \
							 kernels/x86_64/sandybridge/rnn_asm_d8x4_case2.c \
							 kernels/x86_64/sandybridge/rnn_asm_d8x4_var3.c \
							 kernels/x86_64/sandybridge/rnn_asm_d8x4_var3_case2.c \
							 kernels/x86_64/sandybridge/rnn_rank_k_asm_d8x4.c \
							 kernels/x86_64/sandybridge/rnn_rank_int_d4.c \
							 kernels/x86_64/sandybridge/rnn_merge_int_d8.c \
							 kernels/x86_64/sandybridge/rnn_merge_int_d4plus4.c \
							 kernels/x86_64/sandybridge/rnn_sort_int_d4x4.c \
							 kernels/x86_64/sandybridge/rnn_r_int_d8x4.c \
							 kernels/x86_64/sandybridge/rnn_r_int_d8x4_row.c \
							 kernels/x86_64/sandybridge/rnn_r_int_d8x4_var3.c \
							 kernels/x86_64/sandybridge/rnn_merge_int_dn.c \
							 kernels/x86_64/sandybridge/rnn_merge_int_dn_var2.c \
							 kernels/x86_64/sandybridge/sq2nrm_asm_d8x4.c \

RNN_OBJ=$(FRAME_CC_SRC:.c=.o) $(FRAME_CPP_SRC:.cpp=.o) $(KERNEL_SRC:.c=.o)

all: $(LIBRNN) TESTRNN

TESTRNN: $(LIBRNN)
	cd $(RNN_DIR)/test && $(MAKE) && cd $(RNN_DIR)

$(LIBRNN): $(RNN_OBJ)
	$(ARCH) $(ARCHFLAGS) $@ $(RNN_OBJ)
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
	rm $(RNN_OBJ)
	cd $(RNN_DIR)/test && $(MAKE) clean && cd $(RNN_DIR)
