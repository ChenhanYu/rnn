#!/bin/bash
export GSKNN_DIR=$PWD
echo "GSKNN_DIR = $GSKNN_DIR"

## Manually set the target architecture.
#export GSKNN_ARCH_MAJOR=x86_64
#export GSKNN_ARCH_MINOR=sandybridge

export GSKNN_ARCH_MAJOR=x86_64
export GSKNN_ARCH_MINOR=haswell

#export GSKNN_ARCH_MAJOR=arm
#export GSKNN_ARCH_MINOR=neon

export GSKNN_ARCH=$GSKNN_ARCH_MAJOR/$GSKNN_ARCH_MINOR
echo "GSKNN_ARCH = $GSKNN_ARCH"

## For Macbook Pro
export DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH:/opt/intel/lib:/opt/intel/mkl/lib
echo "DYLD_LIBRARY_PATH = $DYLD_LIBRARY_PATH"

## Compiler options (if false, then use GNU compilers)
export GSKNN_USE_INTEL=false
echo "GSKNN_USE_INTEL = $GSKNN_USE_INTEL"
export GSKNN_USE_GNU=true
echo "GSKNN_USE_GNU = $GSKNN_USE_GNU"

## Whether use BLAS or not?
export GSKNN_USE_BLAS=false
echo "GSKNN_USE_BLAS = $GSKNN_USE_BLAS"

## Manually set the mkl or openblas path
#export GSKNN_MKL_DIR=/opt/intel/mkl
export GSKNN_MKL_DIR=$TACC_MKL_DIR
echo "GSKNN_MKL_DIR = $GSKNN_MKL_DIR"

export GSKNN_OPENBLAS_DIR=
echo "GSKNN_OPENBLAS_DIR = $GSKNN_OPENBLAS_DIR"


## Parallel options
export KMP_AFFINITY=compact
export OMP_NUM_THREADS=24
export GSKNN_IC_NT=24
