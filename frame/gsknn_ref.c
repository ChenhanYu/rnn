#include <omp.h>
#include <stdlib.h>
#include <limits.h>
#include <float.h>

#include <gsknn.h>
#include <gsknn_ref.h>

#ifdef USE_BLAS
// dgemm prototype
void dgemm(char*, char*, int*, int*, int*, double*, double*, 
    int*, double*, int*, double*, double*, int*);
// sgemm prototype
void sgemm(char*, char*, int*, int*, int*, float*, float*, 
    int*, float*, int*, float*, float*, int*);
#endif


// This reference function will call BLAS.
#ifdef KNN_PREC_SINGLE
void sgsknn_ref
#else
void dgsknn_ref
#endif
    (
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


  // Allocate buffers.
  As = (prec_t*)malloc( sizeof(prec_t) * m * k );
  Bs = (prec_t*)malloc( sizeof(prec_t) * n * k );
  Cs = (prec_t*)malloc( sizeof(prec_t) * m * n );


  #include "gsknn_ref_impl.h"


  // Pure C Max Heap implementation. 
  beg = omp_get_wtime();
  #pragma omp parallel for schedule( dynamic )
  for ( j = 0; j < n; j ++ ) {
#ifdef KNN_PREC_SINGLE
    heap_sort_s( m, r, &Cs[ j * m ], alpha, &D[ j * r ], &I[ j * r ] );
#else
    heap_sort_d( m, r, &Cs[ j * m ], alpha, &D[ j * r ], &I[ j * r ] );
#endif
  }
  time_heap = omp_get_wtime() - beg;

  
  /*
  printf( "collect: %5.3lf, gemm: %5.3lf, square: %5.3lf, heap: %5.3lf sec\n", 
      time_collect, time_dgemm, time_square, time_heap );
  */

  // Free  buffers
  free( As );
  free( Bs );
  free( Cs );
}
