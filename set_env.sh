#!/bin/bash
export RNN_DIR=$PWD
echo "RNN_DIR = $RNN_DIR"

export KMP_AFFINITY=compact
#export KMP_AFFINITY=scatter

export OMP_NUM_THREADS=10
export RNN_IC_NT=10

# For macbook pro
export DYLD_LIBRARY_PATH=/opt/intel/lib:/opt/intel/mkl/lib
echo "DYLD_LIBRARY_PATH = $DYLD_LIBRARY_PATH"
#ulimit -s 65532
