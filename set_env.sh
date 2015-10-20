#!/bin/bash
export RNN_DIR=$PWD
echo "RNN_DIR = $RNN_DIR"

# Manually set the target architecture.
export RNN_ARCH_MAJOR=x86_64
export RNN_ARCH_MINOR=sandybridge
export RNN_ARCH=$RNN_ARCH_MAJOR/$RNN_ARCH_MINOR
echo "RNN_ARCH = $RNN_ARCH"

# For macbook pro
export DYLD_LIBRARY_PATH=/opt/intel/lib:/opt/intel/mkl/lib
echo "DYLD_LIBRARY_PATH = $DYLD_LIBRARY_PATH"

# Manually set the mkl path
#export RNN_MKL_DIR=/opt/intel/mkl
export RNN_MKL_DIR=$TACC_MKL_DIR
echo "RNN_MKL_DIR = $RNN_MKL_DIR"

# Parallel options
export KMP_AFFINITY=compact
export OMP_NUM_THREADS=10
export RNN_IC_NT=10
