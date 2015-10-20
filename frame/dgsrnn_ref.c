//#include <mkl.h>
#include <rnn.h>
#include <omp.h>
#include <stdlib.h>
#include <limits.h>
#include <float.h>

#define USE_BLAS 0


// dgemm prototype
void dgemm(char*, char*, int*, int*, int*, double*, double*, 
    int*, double*, int*, double*, double*, int*);

// sgemm prototype
void sgemm(char*, char*, int*, int*, int*, float*, float*, 
    int*, float*, int*, float*, float*, int*);


// This reference function will call BLAS.
void dgsrnn_ref(
    int    m,
    int    n,
    int    k,
    int    r,
    prec_t *XA,
    prec_t *XA2,
    int    *alpha,
    prec_t *XB,
    prec_t *XB2,
    int    *beta,
    prec_t *D,
    int    *I
    )
{
  // Local variables.
  int    i, j, p;
  double beg, time_collect, time_dgemm, time_square, time_heap;
  prec_t *As, *Bs, *Cs;
  prec_t fneg2 = -2.0, fzero = 0.0;


  // Sanity check for early return.
  if ( m == 0 || n == 0 || k == 0 || r == 0 ) return;


  As = (prec_t*)malloc( sizeof(prec_t) * m * k );
  Bs = (prec_t*)malloc( sizeof(prec_t) * n * k );
  Cs = (prec_t*)malloc( sizeof(prec_t) * m * n );


  // Collect As from XA and XB.
  beg = omp_get_wtime();
  #pragma omp parallel for private( p )
  for ( i = 0; i < m; i ++ ) {
    for ( p = 0; p < k; p ++ ) {
      As[ i * k + p ] = XA[ alpha[ i ] * k + p ];
    }
  }
  #pragma omp parallel for private( p )
  for ( j = 0; j < n; j ++ ) {
    for ( p = 0; p < k; p ++ ) {
      Bs[ j * k + p ] = XB[ beta[ j ] * k + p ];
    }
  }
  time_collect = omp_get_wtime() - beg;


  // Compute the inner-product term.
  beg = omp_get_wtime();
  if ( USE_BLAS ) {
#ifdef KNN_PREC_SINGLE
    sgemm( "T", "N", &m, &n, &k, &fneg2,
        As, &k, Bs, &k, &fzero, Cs, &m );
#else
    dgemm( "T", "N", &m, &n, &k, &fneg2,
        As, &k, Bs, &k, &fzero, Cs, &m );
#endif
  }
  else {
    #pragma omp parallel for private( i, p )
    for ( j = 0; j < n; j ++ ) {
      for ( i = 0; i < m; i ++ ) {
        Cs[ j * m + i ] = 0.0;
        for ( p = 0; p < k; p ++ ) {
          Cs[ j * m + i ] -= 2.0 * As[ i * k + p ] * Bs[ j * k + p ];
        }
      }
    }
  }
  time_dgemm = omp_get_wtime() - beg;

  /*
  // 1-norm
  #pragma omp parallel for private( i, p )
  for ( j = 0; j < n; j ++ ) {
    for ( i = 0; i < m; i ++ ) {
      Cs[ j * m + i ] = 0.0;
      for ( p = 0; p < k; p ++ ) {
        Cs[ j * m + i ] += fabs( As[ i * k + p ] - Bs[ j * k + p ] );
      }
    }
  }
  */

  time_dgemm = omp_get_wtime() - beg;

  beg = omp_get_wtime();
  #pragma omp parallel for private( i )
  for ( j = 0; j < n; j ++ ) {
    for ( i = 0; i < m; i ++ ) {
      Cs[ j * m + i ] += XA2[ alpha[ i ] ];
      Cs[ j * m + i ] += XB2[ beta[ j ] ];
    }
  }
  time_square = omp_get_wtime() - beg;


  beg = omp_get_wtime();
  #pragma omp parallel for schedule( dynamic )
  for ( j = 0; j < n; j ++ ) {
    heap_sort( m, r, &Cs[ j * m ], alpha, &D[ j * r ], &I[ j * r ] );
  }
  time_heap = omp_get_wtime() - beg;

  
  /*
  printf( "collect: %5.3lf, gemm: %5.3lf, square: %5.3lf, heap: %5.3lf sec\n", 
      time_collect, time_dgemm, time_square, time_heap );
  */

  // Free the temporary buffers
  free( As );
  free( Bs );
  free( Cs );
}
