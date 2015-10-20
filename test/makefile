include $(RNN_DIR)/make.inc

RNN_TEST_CC_SRC=  \
                 test_dgsrnn.c \

RNN_TEST_CPP_SRC= \
									test_dgsrnn_stl.cpp \


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
