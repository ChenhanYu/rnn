include $(RNN_DIR)/make.inc

RNN_TEST_CC_SRC=  \
                 test_dgsrnn.c \
                 test_dgsnm.c \
#								 test_bitonic_d8.c \
#								 test_merge_d8.c \
#								 test_merge_dn.c \
#								 test_merge_d4plus4.c \
#								 test_rank_d4.c \
#								 test_sort_d4x4.c \

RNN_TEST_CPP_SRC= \
									test_dgsrnn_stl.cpp \
#									test_dgsrnn_directKQuery.cpp \


RNN_TEST_EXE= $(RNN_TEST_CC_SRC:.c=.x) $(RNN_TEST_CPP_SRC:.cpp=.x)

all: $(RNN_TEST_EXE)


# ---------------------------------------------------------------------------
# Executable files compiling rules
# ---------------------------------------------------------------------------
%.x: %.c 
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS) $(LDLIBS)

%.x: %.cpp 
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS) $(LDLIBS)
# ---------------------------------------------------------------------------

clean:
	rm -f $(RNN_TEST_EXE)
