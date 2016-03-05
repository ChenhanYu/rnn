#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <limits.h>
#include <gsknn.h>

void test_dstrassen(
    int    m,
    int    n,
    int    k
    ) 
{
  int    i, j, p, nx;
  int    *amap;
  double *XA, *C, *Cgoal;
  double tmp, error, flops;
  double ref_beg, ref_time, dstrassen_beg, dstrassen_time;

  nx = 8192;

  amap  = (int*)malloc( sizeof(int) * m );
  XA    = (double*)malloc( sizeof(double) * k * nx );
  C     = (double*)malloc( sizeof(double) * m * n );
  Cgoal  = (double*)malloc( sizeof(double) * m * n );

  // Compute the universal k-mean.
  for ( i = 0; i < m; i ++ ) amap[ i ] = i;

  // random[ 0, 0.1 ]
  for ( i = 0; i < nx; i ++ ) {
    for ( p = 0; p < k; p ++ ) {
      XA[ i * k + p ] = (double)( rand() % 1000 ) / 1000.0;	
    }
  }

  // Initialize C
  for ( j = 0; j < n; j ++ ) {
	for ( i = 0; i < m; i ++ ) {
	  C[ j * m + i ] = 0.0;
	  Cgoal[ j * m + i ] = 0.0;
	}
  }


  dstrassen_beg = omp_get_wtime();

  dstrrk(
	m,
	n,
	k,
	k,
	XA,
	amap,
	XA,
	amap,
	C,
	m
	);

  dstrassen_time = omp_get_wtime() - dstrassen_beg;


  ref_beg = omp_get_wtime();
#ifdef USE_BLAS
  double fone = 1.0, fzero = 0.0;
  dgemm_( "T", "N", &m, &n, &k, &fone,
        XA, &k, XA, &k, &fzero, Cgoal, &m );
#else
  #pragma omp parallel for private( i, j, p )
  for ( j = 0; j < n; j ++ ) {
	for ( i = 0; i < m; i ++ ) {
	  for ( p = 0; p < k; p ++ ) {
		Cgoal[ j * m + i ] += XA[ amap[ i ] * k + p ] * XA[ amap[ j ] * k + p ];
	  }
	}
  }
#endif
  ref_time = omp_get_wtime() - ref_beg;
  

  double rel_error;

  for ( j = 0; j < n; j ++ ) {
	for ( i = 0; i < m; i ++ ) {
	  rel_error = ( C[ j * m + i ] - Cgoal[ j * m + i ] ) / Cgoal[ j * m + i ];
	  if ( fabs( rel_error ) > 1E-10 ) {
		printf( "ERROR: %d, %d, %lf, %lf\n", i, j, C[ j * m + i ], Cgoal[ j * m + i ] );
	  }
	}
  }

  flops = ( m * n / ( 1000.0 * 1000.0 * 1000.0 ) ) * ( 2 * k );

  printf( "%d, %d, %d, %5.2lf, %5.2lf\n", m, n, k, 
	  flops / dstrassen_time, flops / ref_time );

  free( XA );
  free( C );
  free( Cgoal );
  free( amap );
}





int main( int argc, char *argv[] )
{
  int    m, n, k;

  if ( argc < 4 ) {
    printf( "argc: %d\n", argc );
    printf( "we need at least 3 arguments!\n" );
    exit( 0 );
  }

  sscanf( argv[ 1 ], "%d", &m );
  sscanf( argv[ 2 ], "%d", &n );
  sscanf( argv[ 3 ], "%d", &k );

  test_dstrassen( m, n, k );

  return 0;
}
