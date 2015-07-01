#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <limits.h>
#include <rnn.h>


void test_dgsnm(
    int    m,
    int    n,
    int    k,
    int    niter,
    double tol
    ) 
{
  int    i, j, p, nx, iter;
  int    *amap, *pi;
  double *XA, *XA2;
  double tmp, error, flops;
  double ref_beg, ref_time, dgsnm_beg, dgsnm_time;

  nx = 4096 * 10;

  pi    = (int*)malloc( sizeof(int) * m );
  amap  = (int*)malloc( sizeof(int) * m );
  XA    = (double*)malloc( sizeof(double) * k * nx );
  XA2   = (double*)malloc( sizeof(double) * nx ); 

  // Compute the universal k-mean.
  for ( i = 0; i < m; i ++ ) amap[ i ] = i;

  // random[ 0, 0.1 ]
  for ( i = 0; i < nx; i ++ ) {
    for ( p = 0; p < k; p ++ ) {
      XA[ i * k + p ] = (double)( rand() % 100 ) / 1000.0;	
    }
  }

  // Compute XA2
  for ( i = 0; i < nx; i ++ ) {
    tmp = 0.0;
    for ( p = 0; p < k; p ++ ) {
      tmp += XA[ i * k + p ] * XA[ i * k + p ];
    }
    XA2[ i ] = tmp;
  }

  // Initialize pi in a periodic distribution.
  for ( i = 0; i < m; i ++ ) pi[ i ]  = i % n;

  dgsnm_beg = omp_get_wtime();

  dgsbnm(
      m,
      n,
      k,
      XA,
      XA2,
      amap,
      pi,
      tol,
      niter
      );

  dgsnm_time = omp_get_wtime() - dgsnm_beg;

  printf( "%d, %d, %d, %5.2lf\n", m, n, k, dgsnm_time );

  free( XA );
  free( XA2 );
  free( amap );
  free( pi );
}





int main( int argc, char *argv[] )
{
  int    m, n, k, niter;
  double tol;

  if ( argc < 4 ) {
    printf( "argc: %d\n", argc );
    printf( "we need at least 3 arguments!\n" );
    exit( 0 );
  }

  niter = 10;
  tol   = 0.0001;

  sscanf( argv[ 1 ], "%d", &m );
  sscanf( argv[ 2 ], "%d", &n );
  sscanf( argv[ 3 ], "%d", &k );

  if ( argc > 4 ) 
    sscanf( argv[ 4 ], "%d", &niter );

  if ( argc > 5 ) 
    sscanf( argv[ 5 ], "%lf", &tol );


  test_dgsnm( m, n, k, niter, tol );

  return 0;
}
