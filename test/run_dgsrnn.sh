#!/bin/bash
export DYLD_LIBRARY_PATH=/opt/intel/lib:/opt/intel/mkl/lib

m=1027
n=1027

r=10
echo "var1_k16=["
for (( k=4; k<1024; k+=15 ))
do
  ./test_dgsrnn.x     $m $n $k $r
  ./test_dgsrnn_stl.x $m $n $k $r
  echo ""
done
echo "];"
