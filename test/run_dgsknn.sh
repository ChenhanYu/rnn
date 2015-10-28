#!/bin/bash
export DYLD_LIBRARY_PATH=/opt/intel/lib:/opt/intel/mkl/lib

m=4
n=4

r=4
echo "var1_k16=["
for (( k=4; k<128; k+=31 ))
do
#  ./test_dgsknn.x     $m $n $k $r
#  ./test_dgsknn_stl.x $m $n $k $r
  ./test_sgsknn.x     $m $n $k $r
  ./test_sgsknn_stl.x $m $n $k $r
  echo ""
done
echo "];"
