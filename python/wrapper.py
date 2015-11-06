import os
import ctypes
from numpy import *
import numpy



#
#
#
def dgsknn_ref( d, k, Xq, Xq2, q, Xr, Xr2, r, D, I ):
  m = r.size
  n = q.size
  libgsknn_path = os.environ.get( 'GSKNN_DIR' ) + '/lib/libgsknn.so'
  libgsknn = ctypes.cdll.LoadLibrary( libgsknn_path )
  libgsknn.dgsknn_ref(
    ctypes.c_int( m ),
    ctypes.c_int( n ),
    ctypes.c_int( d ),
    ctypes.c_int( k ),
    ctypes.c_void_p( Xr.ctypes.data ),
    ctypes.c_void_p( Xr2.ctypes.data ),
    ctypes.c_void_p( r.ctypes.data ),
    ctypes.c_void_p( Xq.ctypes.data ),
    ctypes.c_void_p( Xq2.ctypes.data ),
    ctypes.c_void_p( q.ctypes.data ),
    ctypes.c_void_p( D.ctypes.data ),
    ctypes.c_void_p( I.ctypes.data )
    )
  return



#
#
#
def dgsknn( d, k, Xq, Xq2, q, Xr, Xr2, r, D, I ):
  m = r.size
  n = q.size
  libgsknn_path = os.environ.get( 'GSKNN_DIR' ) + '/lib/libgsknn.so'
  libgsknn = ctypes.cdll.LoadLibrary( libgsknn_path )
  libgsknn.dgsknn_var1(
    ctypes.c_int( n ),
    ctypes.c_int( m ),
    ctypes.c_int( d ),
    ctypes.c_int( k ),
    ctypes.c_void_p( Xq.ctypes.data ),
    ctypes.c_void_p( Xq2.ctypes.data ),
    ctypes.c_void_p( q.ctypes.data ),
    ctypes.c_void_p( Xr.ctypes.data ),
    ctypes.c_void_p( Xr2.ctypes.data ),
    ctypes.c_void_p( r.ctypes.data ),
    ctypes.c_void_p( D.ctypes.data ),
    ctypes.c_void_p( I.ctypes.data )
    )
  return



#
#
#
N = 10
m = 4
n = 3
d = 3
k = 3

# Randomly generate data points.
X  = numpy.random.rand( N, d )
X2 = numpy.ndarray( shape = ( N ),    dtype = numpy.double )
q  = numpy.ndarray( shape = ( n ),    dtype = numpy.int32 )
r  = numpy.ndarray( shape = ( m ),    dtype = numpy.int32 )
D  = numpy.ndarray( shape = ( n, k ), dtype = numpy.double )
I  = numpy.ndarray( shape = ( n, k ), dtype = numpy.int32 )

for j in range( n ): q[ j ] = j
for i in range( m ): r[ i ] = i

for i in range( N ):
  X2[ i ] = 0.0
  for j in range( d ):
    X2[ i ] += X[ i, j ] * X[ i, j ]

for i in range( k ): 
  for j in range( n ):
    D[ i, j ] = 999999.9
    I[ i, j ] = -1

print "gsknn_ref"
dgsknn_ref( d, k, X, X2, q, X, X2, r, D, I )
print D
print I

for i in range( k ): 
  for j in range( n ):
    D[ i, j ] = 999999.9
    I[ i, j ] = -1

print "gsknn_ref"
dgsknn( d, k, X, X2, q, X, X2, r, D, I )
print D
print I
