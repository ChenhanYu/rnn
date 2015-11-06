#!/bin/bash
export DYLD_LIBRARY_PATH=/opt/intel/lib:/opt/intel/mkl/lib

m=4097
n=4097
r=10
kmax=600

echo "run_gsknn=["
for (( k=4; k<kmax; k+=31 ))
do
  ./test_dgsknn.x     $m $n $k $r
  ./test_dgsknn_stl.x $m $n $k $r
  ./test_sgsknn.x     $m $n $k $r
  ./test_sgsknn_stl.x $m $n $k $r
done
echo "];"
