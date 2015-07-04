#!/bin/bash

#m=2048
#n=2048

#r=16
#
#echo "r16=["
#for (( k=4; k<1040; k+=16 ))
#do
##  ./test_dgsrnn.x $m $n $k $r
#  ./test_dgsrnn_stl.x $m $n $k $r
#done
#echo "];"
#
#
#r=128
#
#echo "r128=["
#for (( k=4; k<1040; k+=16 ))
#do
##  ./test_dgsrnn.x $m $n $k $r
#  ./test_dgsrnn_stl.x $m $n $k $r
#done
#echo "];"
#
#
#r=512
#
#echo "r512=["
#for (( k=4; k<1040; k+=16 ))
#do
##  ./test_dgsrnn.x $m $n $k $r
#  ./test_dgsrnn_stl.x $m $n $k $r
#done
#echo "];"

#r=2048
#
#echo "r2048=["
#for (( k=4; k<1040; k+=16 ))
#do
#  #./test_dgsrnn.x $m $n $k $r
#  ./test_dgsrnn_stl.x $m $n $k $r
#done
#echo "];"



m=8192
n=8192


#k=16
r=10
echo "var1_k16=["
for (( k=4; k<1024; k+=32 ))
#for (( r=4; r<2060; r+=32 ))
do
  ./test_dgsrnn.x $m $n $k $r
#  ./test_dgsrnn_stl.x $m $n $k $r
done
echo "];"
#
#
#k=256
#echo "var3_k256=["
#for (( r=4; r<2060; r+=32 ))
#do
##  ./test_dgsrnn.x $m $n $k $r
#  ./test_dgsrnn_stl.x $m $n $k $r
#done
#echo "];"
#
#
k=64
echo "var1_k64=["
for (( r=4; r<2060; r+=32 ))
do
#  ./test_dgsrnn.x $m $n $k $r
  ./test_dgsrnn_stl.x $m $n $k $r
done
echo "];"
#
#
#k=256
#echo "var3_k256=["
#for (( r=4; r<2060; r+=32 ))
#do
##  ./test_dgsrnn.x $m $n $k $r
#  ./test_dgsrnn_stl.x $m $n $k $r
#done
#echo "];"
#
#
#k=1024
#echo "var3_k1024=["
#for (( r=4; r<2060; r+=32 ))
#do
##  ./test_dgsrnn.x $m $n $k $r
#  ./test_dgsrnn_stl.x $m $n $k $r
#done
#echo "];"



#n=8192
#k=4
#r=2048
#
#for (( m=8; m<8196; m+=32 ))
#do
#  ./test_dgsrnn.x $m $n $k $r
#done


#m=8192
#k=1024
#r=1
#
#for (( n=3; n<512; n+=1 ))
#do
#  ./test_dgsrnn.x $m $n $k $r
#done


#k=64
#r=128
#
#echo "d64_k128=["
#for (( n=96; n<8196; n+=96 ))
#do
#  ./test_dgsrnn_stl.x $n $n $k $r
#done
#echo "];"


#k=1024
#r=2048
#
#echo "d1024_k2048=["
#for (( n=96; n<8196; n+=96 ))
#do
#  ./test_dgsrnn_stl.x $n $n $k $r
#done
#echo "];"



#./test_dgsrnn_stl.x 8192 8192 1024  1
#./test_dgsrnn_stl.x 8192 8192 1024  16
#./test_dgsrnn_stl.x 8192 8192 1024  128 
#./test_dgsrnn_stl.x 8192 8192 1024  512 
#./test_dgsrnn_stl.x 8192 8192 64   128 
#./test_dgsrnn_stl.x 8192 8192 256  128 
#./test_dgsrnn_stl.x 8192 8192 1024 128 
#./test_dgsrnn_stl.x 4096 4096 1024 16
#./test_dgsrnn.x 8192 8192 8    2048
#./test_dgsrnn.x 8192 8192 16   2048
#./test_dgsrnn.x 8192 8192 32   2048
#./test_dgsrnn.x 8192 8192 64   2048
#./test_dgsrnn.x 8192 8192 128  2048 
#./test_dgsrnn.x 8192 8192 256  2048
#./test_dgsrnn.x 8192 8192 512  2048
#./test_dgsrnn.x 8192 8192 1024 2048

#./test_dgsrnn.x 8 64 1 40
