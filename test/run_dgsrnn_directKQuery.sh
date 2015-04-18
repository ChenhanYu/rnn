#!/bin/bash


#./test_dgsrnn_directKQuery.x 8 4 3 4
#./test_dgsrnn_directKQuery.x 4097 4097 10 2048

m=4097
n=4097
r=2048

for (( k=4; k<1040; k+=1 ))
do
  ./test_dgsrnn_directKQuery.x $m $n $k $r
done


#m=8192
#n=8192
#k=16
#
#for (( r=4; r<2052; r+=16 ))
#do
#  ./test_dgsrnn.x $m $n $k $r
#done



#n=8192
#k=4
#r=2048
#
#for (( m=8; m<8196; m+=32 ))
#do
#  ./test_dgsrnn.x $m $n $k $r
#done


#m=8192
#k=256
#r=4
#
#for (( n=8; n<8196; n+=32 ))
#do
#  ./test_dgsrnn.x $m $n $k $r
#done


#k=4
#r=2048
#
#for (( n=8; n<8196; n+=32 ))
#do
#  ./test_dgsrnn.x $n $n $k $r
#done


#./test_dgsrnn.x 8192 8192 4    2048 
#./test_dgsrnn.x 8192 8192 8    2048
#./test_dgsrnn.x 8192 8192 16   2048
#./test_dgsrnn.x 8192 8192 32   2048
#./test_dgsrnn.x 8192 8192 64   2048
#./test_dgsrnn.x 8192 8192 128  2048 
#./test_dgsrnn.x 8192 8192 256  2048
#./test_dgsrnn.x 8192 8192 512  2048
#./test_dgsrnn.x 8192 8192 1024 2048

#./test_dgsrnn.x 8 64 1 40
