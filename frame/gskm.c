#include <stdio.h>
#include <omp.h>
#include <gsknn.h>
#define min( i, j ) ( (i)<(j) ? (i): (j) )


double quality(
    int    m,
    int    n,
    int    k,
    double *X,
    int    *amap,
    double *C,
    int    *pi
    )
{
  int    i, ii, j, p;
  double q;

  q = 0.0;

  // sum_i || X_i - C_pi(i) ||_2
  for ( ii = 0; ii < m; ii ++ ) {
    double sum = 0.0;
    i = amap[ ii ];
    j = pi[ ii ];
    for ( p = 0; p < k; p ++ ) {
      double tmp;
      tmp = X[ i * k + p ] - C[ j * k + p ];
      sum += tmp * tmp;
    }
    q += sqrt( sum );
  }

  return q;
}

void centroid(
    int    m,
    int    n,
    int    k,
    double *X,
    int    *amap,
    double *C,
    double *C2,
    int    *pi
    )
{
  int    i, ii, j, p;
  int    *np;

  np = (int*)malloc( sizeof(int) * n );

  for ( j = 0; j < n; j ++ ) np[ j ] = 0;

  for ( j = 0; j < n; j ++ ) {
    for ( p = 0; p < k; p ++ ) {
      C[ j * k + p ] = 0.0;
    }
    C2[ j ] = 0.0;
  }

  for ( ii = 0; ii < m; ii ++ ) {
    // Map the local index ii to its global index i.
    i = amap[ ii ];
    j = pi[ i ]; // TODO: need an assert here.
    for ( p = 0; p < k; p ++ ) {
      C[ j * k + p ] += X[ i * k + p ];
    }
    np[ j ] ++;
  }
  
  for ( j = 0; j < n; j++ ) {
    if ( np[ j ] != 0 ) {
      double sum = 0.0;
      for ( p = 0; p < k; p ++ ) {
        C[ j * k + p ] /= np[ j ];
        sum += C[ j * k + p ] * C[ j * k + p ];
      }
      C2[ j ] = sum;
    }
  }
}


void dgskm(
    int    m,
    int    n,
    int    d,
    double *X,
    double *X2,
    int    *amap,
    int    *pi,
    double tol,
    int    niter
    )
{
  double *C, *C2, *D;
  double delta, q, qnext;
  int    *cmap, *activity;
  int    i, j, p, iter;

  // Allocate and compute centroid C and C2.
  C  = gsknn_malloc_aligned( d, n, sizeof(double) );
  C2 = gsknn_malloc_aligned( 1, n, sizeof(double) );
  centroid( m, n, d, X, amap, C, C2, pi ); 
  
  // Allocate cmap [0,n-1].
  cmap = (int*)malloc( sizeof(int)*n );
  for ( j = 0; j < n; j ++ ) cmap[ j ] = j;

  // Allocate D (similarity).
  D = gsknn_malloc_aligned( m, 1, sizeof(double) );

  q     = quality( m, n, d, X, amap, C, pi );
  delta = q;
  iter  = 0;

  // Main loop
  while ( delta > tol && iter < niter ) {
    printf( "iter#%d ---> %lf\n", iter, q );
    for ( i = 0; i < m; i ++ ) D[ i ] = 99999.99;

    dgsknn_var1(
        m,
        n,
        d,
        1,
        X,
        X2,
        amap,
        C,
        C2,
        cmap,
        D,
        pi
        );

    centroid( m, n, d, X, amap, C, C2, pi ); 
    qnext = quality( m, n, d, X, amap, C, pi );
    delta = q - qnext;
    q     = qnext;

    iter ++;
  }

  free( C );
  free( C2 );
  free( D );
  free( cmap );
}
