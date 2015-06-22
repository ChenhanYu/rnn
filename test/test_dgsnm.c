#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <limits.h>
#include <rnn.h>


void test_dgsnm(
    int m,
    int n,
    int k
    ) 
{
  int    i, j, p, nx, iter, n_iter;
  int    *amap, *pi;
  double *XA, *XA2;
  double tmp, error, flops;
  double ref_beg, ref_time, dgsrnn_beg, dgsrnn_time;


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

  printf( "here1\n" );

  dgsbnm(
      m,
      n,
      k,
      XA,
      XA2,
      amap,
      pi,
      0.0,
      10
      );

  printf( "here2\n" );


  free( XA );
  free( XA2 );
  free( pi );
}





int main( int argc, char *argv[] )
{
  int    m, n, k; 
  fflush( stdout );

  if ( argc != 4 ) {
    printf( "argc: %d\n", argc );
    printf( "we need 3 arguments!\n" );
    exit( 0 );
  }

  sscanf( argv[ 1 ], "%d", &m );
  sscanf( argv[ 2 ], "%d", &n );
  sscanf( argv[ 3 ], "%d", &k );

  test_dgsnm( m, n, k );

  return 0;
}
