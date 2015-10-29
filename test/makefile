ifeq ($(GSKNN_USE_INTEL),true)
include $(GSKNN_DIR)/make.intel.inc
else
include $(GSKNN_DIR)/make.gnu.inc
endif


GSKNN_TEST_CC_SRC= \
									 test_dgsknn.c \
									 test_sgsknn.c \

GSKNN_TEST_CPP_SRC=\
									 test_dgsknn_stl.cpp \
									 test_sgsknn_stl.cpp \


GSKNN_TEST_EXE= $(GSKNN_TEST_CC_SRC:.c=.x) $(GSKNN_TEST_CPP_SRC:.cpp=.x)

all: $(GSKNN_TEST_EXE)

clean:
	rm -f $(GSKNN_TEST_EXE)

# ---------------------------------------------------------------------------
# Executable files compiling rules
# ---------------------------------------------------------------------------
%.x: %.c 
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS) $(LDLIBS)

%.x: %.cpp 
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS) $(LDLIBS)
# ---------------------------------------------------------------------------
