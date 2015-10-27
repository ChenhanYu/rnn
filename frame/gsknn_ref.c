#include <omp.h>
#include <stdlib.h>
#include <limits.h>
#include <float.h>

#include <gsknn.h>
#include <gsknn_ref.h>

#ifdef USE_BLAS
/* 
 * dgemm and sgemm prototypes
 *
 */ 
void dgemm(char*, char*, int*, int*, int*, double*, double*, 
    int*, double*, int*, double*, double*, int*);
void sgemm(char*, char*, int*, int*, int*, float*, float*, 
    int*, float*, int*, float*, float*, int*);
#endif


/*
 * This reference function will call BLAS.
 *
 */
void sgsknn_ref(
    int    m,
    int    n,
    int    k,
    int    r,
    float  *XA,
    float  *XA2,
    int    *alpha,
    float  *XB,
    float  *XB2,
    int    *beta,
    float  *D,
    int    *I
    )
{
  printf( "sgsknn_ref(): Not implemented yet." );
  //int    i, j, p;
  //double beg, time_collect, time_dgemm, time_square, time_heap;
  //float  *As, *Bs, *Cs;
  //float  fneg2 = -2.0, fzero = 0.0;

  //// Sanity check for early return.
  //if ( m == 0 || n == 0 || k == 0 || r == 0 ) return;

  //// Allocate buffers.
  //As = (float*)malloc( sizeof(float) * m * k );
  //Bs = (float*)malloc( sizeof(float) * n * k );
  //Cs = (float*)malloc( sizeof(float) * m * n );

  //#include "sgsknn_ref_impl.h"

  //// Pure C Max Heap implementation. 
  //beg = omp_get_wtime();
  //#pragma omp parallel for schedule( dynamic )
  //for ( j = 0; j < n; j ++ ) {
  //  heap_sort_s( m, r, &Cs[ j * m ], alpha, &D[ j * r ], &I[ j * r ] );
  //}
  //time_heap = omp_get_wtime() - beg;

  ////printf( "collect: %5.3lf, gemm: %5.3lf, square: %5.3lf, heap: %5.3lf sec\n", 
  ////    time_collect, time_dgemm, time_square, time_heap );

  //// Free  buffers
  //free( As );
  //free( Bs );
  //free( Cs );
}


void dgsknn_ref(
    int    m,
    int    n,
    int    k,
    int    r,
    double *XA,
    double *XA2,
    int    *alpha,
    double *XB,
    double *XB2,
    int    *beta,
    double *D,
    int    *I
    )
{
  // Local variables.
  int    i, j, p;
  double beg, time_collect, time_dgemm, time_square, time_heap;
  double *As, *Bs, *Cs;
  double fneg2 = -2.0, fzero = 0.0;


  // Sanity check for early return.
  if ( m == 0 || n == 0 || k == 0 || r == 0 ) return;


  // Allocate buffers.
  As = (double*)malloc( sizeof(double) * m * k );
  Bs = (double*)malloc( sizeof(double) * n * k );
  Cs = (double*)malloc( sizeof(double) * m * n );


  #include "gsknn_ref_impl.h"


  // Pure C Max Heap implementation. 
  beg = omp_get_wtime();
  #pragma omp parallel for schedule( dynamic )
  for ( j = 0; j < n; j ++ ) {
    heap_sort_d( m, r, &Cs[ j * m ], alpha, &D[ j * r ], &I[ j * r ] );
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
