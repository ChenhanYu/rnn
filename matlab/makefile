ifeq ($(GSKNN_USE_INTEL),true)
include $(GSKNN_DIR)/make.intel.inc
else
include $(GSKNN_DIR)/make.gnu.inc
endif

MEX=mex

GSKNN_MEX_SRC= \
						   dgsknn_matlab.c \

GSKNN_MEX_EXE= $(GSKNN_MEX_SRC:.c=.mex)

all: $(GSKNN_MEX_EXE)

# ---------------------------------------------------------------------------
# Executable files compiling rules
# ---------------------------------------------------------------------------
%.mex: %.c 
	$(MEX) -v $< $(LDFLAGS) $(LIBGSKNN)
# ---------------------------------------------------------------------------

clean:
	rm -f $(GSKNN_MEX_EXE)
