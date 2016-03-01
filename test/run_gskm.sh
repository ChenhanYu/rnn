#!/bin/bash
export DYLD_LIBRARY_PATH=/opt/intel/lib:/opt/intel/mkl/lib

m=4097
n=120
kmax=600

echo "run_gskm=["
for (( k=4; k<kmax; k+=31 ))
do
  ./test_dgskm.x     $m $n $k 
done
echo "];"
