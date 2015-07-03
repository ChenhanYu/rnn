#include <stdlib.h>
#include <stdio.h>
#include <rnn.h>

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


void dgsbnm(
    int    m,
    int    n,
    int    k,
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
  int    *cmap;
  int    i, j, p, iter;

  // Allocate and compute centroid C and C2.
  C  = rnn_malloc_aligned( k, n, sizeof(double) );
  C2 = rnn_malloc_aligned( 1, n, sizeof(double) );

  centroid( m, n, k, X, amap, C, C2, pi ); 
  
  // Allocate cmap [0,n-1].
  cmap = (int*)malloc( sizeof(int)*n );
  for ( j = 0; j < n; j ++ ) cmap[ j ] = j;

  // Allocate D.
  D = rnn_malloc_aligned( m, 1, sizeof(double) );

  q     = quality( m, n, k, X, amap, C, pi );
  delta = q;
  iter  = 0;

  // Main loop
  while ( delta > tol && iter < niter ) {
    
    printf( "iter#%d ---> %lf\n", iter, q );

    // Reset D vector.
    for ( i = 0; i < m; i ++ ) D[ i ] = 99999.99;

    // Recompute the distance and reassign the group.
    //printf( "gsrnn\n" );
    dgsrnn_var1(
        m,
        n,
        k,
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

    // Recompute centroids.
    //printf( "Centroid\n" );
    centroid( m, n, k, X, amap, C, C2, pi ); 


    qnext = quality( m, n, k, X, amap, C, pi );
    delta = q - qnext;
    q     = qnext;

    // Increase the counter.
    iter ++;
    
  }

  free( C );
  free( C2 );
  free( D );
  free( cmap );
}


void dgsodnm(
    int    m,
    int    n,
    int    k,
    double *X,
    double *X2,
    int    *amap,
    int    *pi,
    double tol,
    int    niter
    )
{
  double *C, *C2, *D;
  double delta;
  int    *cmap, *acti;
  int    i, j, p, iter;

  // Allocate and compute centroid C and C2.
  C  = rnn_malloc_aligned( k, n, sizeof(double) );
  C2 = rnn_malloc_aligned( 1, n, sizeof(double) );

  centroid( m, n, k, X, amap, C, C2, pi ); 

  // Allocate cmap [0,n-1].
  cmap = (int*)malloc( sizeof(int)*n );
  for ( j = 0; j < n; j ++ ) cmap[ j ] = j;

  // Allocate D.
  D = rnn_malloc_aligned( m, 1, sizeof(double) );

  // Activate all points.
  for ( i = 0; i < m; i ++ ) acti[ i ] = i;


  delta = 99999.99;
  iter  = 0;


  // Main loop
  while ( delta > tol && iter < niter ) {

  }

}
