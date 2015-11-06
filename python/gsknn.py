import os
import ctypes
from numpy import *
import numpy


#
#
#
def sgsknn_ref( d, k, Xq, Xq2, q, Xr, Xr2, r, D, I ):
  M = Xr.shape[ 0 ]
  N = Xq.shape[ 0 ]
  m = r.size
  n = q.size
  # Range checking
  assert Xr.shape[ 1 ] == Xq.shape[ 1 ]
  assert M >= m
  assert N >= n
  assert Xq2.size == N
  assert Xr2.size == M
  assert D.size == n * d
  assert I.size == n * d
  # Type checking
  assert Xq.dtype == numpy.float32
  assert Xr.dtype == numpy.float32
  assert Xq2.dtype == numpy.float32
  assert Xr2.dtype == numpy.float32
  assert q.dtype == numpy.int32
  assert r.dtype == numpy.int32
  assert D.dtype == numpy.float32
  assert I.dtype == numpy.int32
  libgsknn_path = os.environ.get( 'GSKNN_DIR' ) + '/lib/libgsknn.so'
  libgsknn = ctypes.cdll.LoadLibrary( libgsknn_path )
  libgsknn.sgsknn_ref(
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
def sgsknn( d, k, Xq, Xq2, q, Xr, Xr2, r, D, I ):
  M = Xr.shape[ 0 ]
  N = Xq.shape[ 0 ]
  m = r.size
  n = q.size
  # Range checking
  assert Xr.shape[ 1 ] == Xq.shape[ 1 ]
  assert M >= m
  assert N >= n
  assert Xq2.size == N
  assert Xr2.size == M
  assert D.size == n * d
  assert I.size == n * d
  # Type checking
  assert Xq.dtype == numpy.float32
  assert Xr.dtype == numpy.float32
  assert Xq2.dtype == numpy.float32
  assert Xr2.dtype == numpy.float32
  assert q.dtype == numpy.int32
  assert r.dtype == numpy.int32
  assert D.dtype == numpy.float32
  assert I.dtype == numpy.int32
  libgsknn_path = os.environ.get( 'GSKNN_DIR' ) + '/lib/libgsknn.so'
  libgsknn = ctypes.cdll.LoadLibrary( libgsknn_path )
  libgsknn.sgsknn_var1(
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
def dgsknn_ref( d, k, Xq, Xq2, q, Xr, Xr2, r, D, I ):
  M = Xr.shape[ 0 ]
  N = Xq.shape[ 0 ]
  m = r.size
  n = q.size
  # Range checking
  assert Xr.shape[ 1 ] == Xq.shape[ 1 ]
  assert M >= m
  assert N >= n
  assert Xq2.size == N
  assert Xr2.size == M
  assert D.size == n * d
  assert I.size == n * d
  # Type checking
  assert Xq.dtype == numpy.double
  assert Xr.dtype == numpy.double
  assert Xq2.dtype == numpy.double
  assert Xr2.dtype == numpy.double
  assert q.dtype == numpy.int32
  assert r.dtype == numpy.int32
  assert D.dtype == numpy.double
  assert I.dtype == numpy.int32
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
  M = Xr.shape[ 0 ]
  N = Xq.shape[ 0 ]
  m = r.size
  n = q.size
  # Range checking
  assert Xr.shape[ 1 ] == Xq.shape[ 1 ]
  assert M >= m
  assert N >= n
  assert Xq2.size == N
  assert Xr2.size == M
  assert D.size == n * d
  assert I.size == n * d
  # Type checking
  assert Xq.dtype == numpy.double
  assert Xr.dtype == numpy.double
  assert Xq2.dtype == numpy.double
  assert Xr2.dtype == numpy.double
  assert q.dtype == numpy.int32
  assert r.dtype == numpy.int32
  assert D.dtype == numpy.double
  assert I.dtype == numpy.int32
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
