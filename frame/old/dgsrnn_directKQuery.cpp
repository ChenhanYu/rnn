#include <vector>
#include <cmath>
#include <cfloat>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

extern "C" {
#include <rnn.h>
}

#define min( i, j ) ( (i)<(j) ? (i): (j) )

void dgsrnn_directKQuery(
    double *ref,
    double *query,
    long   n,
    long   m,
    long   k,
    long   dim,
    std::pair<double, long> *result,
    double *dist,                    // The following 3 parameters are
    double *sqnormr,                 // ignored in this routine.
    double *sqnormq                  // 
    )
{
  int    i, j, p;
  int    *amap, *bmap;
  int    *I;
  double *XA2, *XB2, *D;
  double  beg, setup_time, kernel_time, sort_time;

  // Query map = 0:m-1
  amap  = (int*)malloc( sizeof(int) * m );
  // Reference map = 0:n-1
  bmap  = (int*)malloc( sizeof(int) * n );
  XA2   = (double*)malloc( sizeof(double) * m ); 
  XB2   = (double*)malloc( sizeof(double) * n ); 
  I     = (int*)malloc( sizeof(int) * k * m );
  D     = (double*)malloc( sizeof(double) * k * m );


  beg = omp_get_wtime();
  // ------------------------------------------------------------------------
  // Initialization
  // ------------------------------------------------------------------------
  for ( i = 0; i < m; i ++ ) amap[ i ] = i;
  for ( j = 0; j < n; j ++ ) bmap[ j ] = j;
  //#pragma omp parallel for
  for ( i = 0; i < k * m; i ++ ) {
    I[ i ] = -1;
    D[ i ] = DBL_MAX;
  }
  // ------------------------------------------------------------------------

  
  // ------------------------------------------------------------------------
  // Compute the square distance
  // ------------------------------------------------------------------------
  #pragma omp parallel for private( p )
  for ( i = 0; i < m; i ++ ) {
    double tmp, sum;
    sum = 0.0;
    for ( p = 0; p < dim; p ++ ) {
      tmp = query[ i * dim + p ];
      sum += tmp * tmp;
    }
    XA2[ i ] = sum;
  }
  #pragma omp parallel for private( p )
  for ( j = 0; j < n; j ++ ) {
    double tmp, sum;
    sum = 0.0;
    for ( p = 0; p < dim; p ++ ) {
      tmp = ref[ j * dim + p ];
      sum += tmp * tmp;
    }
    XB2[ j ] = sum;
  }
  // ------------------------------------------------------------------------
  setup_time = omp_get_wtime() - beg;
  

  //printf( "beg dgsrnn_var2, %d, %d, %d, %d\n", m, n, dim, k );
  beg = omp_get_wtime();
  // ------------------------------------------------------------------------
  // Call knn search
  // ------------------------------------------------------------------------
  dgsrnn_var2(
      (int)m,
      (int)n,
      (int)dim,
      (int)k,
      query, // XA
      XA2,
      amap,
      ref,   // XB
      XB2,
      bmap,
      D,
      I
      );
  // ------------------------------------------------------------------------
  kernel_time = omp_get_wtime() - beg;
  //printf( "end dgsrnn_var2, %d, %d, %d, %d\n", m, n, dim, k );

  
  beg = omp_get_wtime();
  // ------------------------------------------------------------------------
  // Convert D and I to std::pair<double, long>
  // ------------------------------------------------------------------------
  #pragma omp parallel for private( j )
  for ( i = 0; i < m; i ++ ) {
    for ( j = 0; j < k; j ++ ) {
      HeapAdjust( &(D[ i * k ]), 0, k - j, &(I[ i * k ]) );
      result[ i * k + k - j - 1 ].first  = D[ i * k ]; 
      result[ i * k + k - j - 1 ].second = (long)I[ i * k ];

      // swap the first element
      D[ i * k ] = D[ i * k + k - j - 1 ];
      I[ i * k ] = I[ i * k + k - j - 1 ];
    }
  }
  // ------------------------------------------------------------------------
  sort_time = omp_get_wtime() - beg;


  //printf( "dgsrnn_var2, %d, %d, %d, %d, %5.2lf, %5.2lf, %5.2lf (sec)\n", 
  //    m, n, dim, k, setup_time, kernel_time, sort_time );
  
  
  free( amap );
  free( bmap );
  free( XA2 );
  free( XB2 );
  free( I );
  free( D );
}


void dgsrnn_directKQuery_var2(
    int    m,
    int    n,
    int    dim,
    int    k,
    double *query,
    double *XA2,
    int    *amap,
    double *ref,
    double *XB2,
    int    *bmap,
    std::pair<double, long> *result
    )
{
  int    i, j;
  double *D;
  int    *I;

  I     = (int*)malloc( sizeof(int) * k * m );
  D     = (double*)malloc( sizeof(double) * k * m );


  // ------------------------------------------------------------------------
  // Initialization
  // ------------------------------------------------------------------------
  //#pragma omp parallel for
  for ( i = 0; i < k * m; i ++ ) {
    I[ i ] = -1;
    D[ i ] = DBL_MAX;
  }
  // ------------------------------------------------------------------------


  //printf( "beg dgsrnn_var2()\n" );
  // ------------------------------------------------------------------------
  // Call knn search
  // ------------------------------------------------------------------------
  dgsrnn_var2(
      (int)m,
      (int)n,
      (int)dim,
      (int)k,
      query, // XA
      XA2,
      amap,
      ref,   // XB
      XB2,
      bmap,
      D,
      I
      );
  // ------------------------------------------------------------------------
  //printf( "end dgsrnn_var2()\n" );


  // ------------------------------------------------------------------------
  // Convert D and I to std::pair<double, long>
  // ------------------------------------------------------------------------
  //#pragma omp parallel for private( j )
  //for ( i = 0; i < m; i ++ ) {
  //  for ( j = 0; j < k; j ++ ) {
  //    HeapAdjust( &(D[ i * k ]), 0, k - j, &(I[ i * k ]) );
  //    result[ i * k + k - j - 1 ].first  = D[ i * k ]; 
  //    result[ i * k + k - j - 1 ].second = (long)I[ i * k ];

  //    // swap the first element
  //    D[ i * k ] = D[ i * k + k - j - 1 ];
  //    I[ i * k ] = I[ i * k + k - j - 1 ];
  //  }
  //}

  #pragma omp parallel for private( j )
  for ( i = 0; i < m; i ++ ) {
    for ( j = 0; j < k; j ++ ) {
    result[ i * k + j ].first  = D[ i * k + j ];
    result[ i * k + j ].second = I[ i * k + j ];
    }
  }
  // ------------------------------------------------------------------------
  

  free( I );
  free( D );
}
