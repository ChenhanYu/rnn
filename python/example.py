import numpy
from gsknn import *

N = 10
m = 4
n = 3
d = 3
k = 3
isdouble = False

if ( isdouble ):
  X  = numpy.random.rand( N, d ).astype( numpy.double )
  X2 = numpy.ndarray( shape = ( N ),    dtype = numpy.double )
  D  = numpy.ndarray( shape = ( n, k ), dtype = numpy.double )
else :
  X  = numpy.random.rand( N, d ).astype( numpy.float32 )
  X2 = numpy.ndarray( shape = ( N ),    dtype = numpy.float32 )
  D  = numpy.ndarray( shape = ( n, k ), dtype = numpy.float32 )

q  = numpy.ndarray( shape = ( n ),    dtype = numpy.int32 )
r  = numpy.ndarray( shape = ( m ),    dtype = numpy.int32 )
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

print "gsknn_ref:"
if ( isdouble ):
  dgsknn_ref( d, k, X, X2, q, X, X2, r, D, I )
else :
  sgsknn_ref( d, k, X, X2, q, X, X2, r, D, I )
print D
print I

for i in range( k ): 
  for j in range( n ):
    D[ i, j ] = 999999.9
    I[ i, j ] = -1

print "gsknn:"
if ( isdouble ):
  dgsknn( d, k, X, X2, q, X, X2, r, D, I )
else :
  sgsknn( d, k, X, X2, q, X, X2, r, D, I )

print D
print I
