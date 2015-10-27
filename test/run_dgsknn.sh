#!/bin/bash
export DYLD_LIBRARY_PATH=/opt/intel/lib:/opt/intel/mkl/lib

m=257
n=4097

r=10
echo "var1_k16=["
for (( k=4; k<1024; k+=31 ))
do
  ./test_dgsknn.x     $m $n $k $r
  ./test_dgsknn_stl.x $m $n $k $r
  echo ""
done
echo "];"
