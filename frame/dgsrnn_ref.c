//#include <mkl.h>
#include <rnn.h>
#include <omp.h>
#include <stdlib.h>
#include <limits.h>
#include <float.h>


// dgemm prototype
void dgemm(char*, char*, int*, int*, int*, double*, double*, 
    int*, double*, int*, double*, double*, int*);


// This reference function will call MKL
void dgsrnn_ref(
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
  int    i, j, p;
  double *As, *Bs, *Cs;
  double beg, time_collect, time_dgemm, time_square, time_heap;
  double dneg2 = -2.0;
  double dzero = 0.0;

  // Sanity check for early return.
  if ( m == 0 || n == 0 || k == 0 || r == 0 ) {
    return;
  }

  As = (double*)malloc( sizeof(double) * m * k );
  Bs = (double*)malloc( sizeof(double) * n * k );
  Cs = (double*)malloc( sizeof(double) * m * n );


  beg = omp_get_wtime();

  // Collect As from XA
  #pragma omp parallel for
  for ( i = 0; i < m; i ++ ) {
    for ( p = 0; p < k; p ++ ) {
      As[ i * k + p ] = XA[ alpha[ i ] * k + p ];
    }
  }

  // Collect Bs from XB
  #pragma omp parallel for
  for ( j = 0; j < n; j ++ ) {
    for ( p = 0; p < k; p ++ ) {
      Bs[ j * k + p ] = XB[ beta[ j ] * k + p ];
    }
  }

  time_collect = omp_get_wtime() - beg;



  beg = omp_get_wtime();

  // C = -2.0 * A^t * B
  //cblas_dgemm(
  //    CblasColMajor,
  //    CblasTrans,
  //    CblasNoTrans,
  //    m,
  //    n,
  //    k,
  //    -2.0,
  //    As,
  //    k,
  //    Bs,
  //    k,
  //    0.0,
  //    Cs,
  //    m
  //    );


  dgemm(
      "T",
      "N",
      &m,
      &n,
      &k,
      &dneg2,
      As,
      &k,
      Bs,
      &k,
      &dzero,
      Cs,
      &m
      );

  /*
  // 1-norm
  #pragma omp parallel for
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
  #pragma omp parallel for
  for ( j = 0; j < n; j ++ ) {
    for ( i = 0; i < m; i ++ ) {
      Cs[ j * m + i ] += XA2[ alpha[ i ] ];
      Cs[ j * m + i ] += XB2[ beta[ j ] ];
    }
  }
  time_square = omp_get_wtime() - beg;


  beg = omp_get_wtime();
  // C: mxn
  #pragma omp parallel for schedule( dynamic )
  for ( j = 0; j < n; j ++ ) {
    heap_sort( m, r, &Cs[ j * m ], alpha, &D[ j * r ], &I[ j * r ] );
  }
  time_heap = omp_get_wtime() - beg;

  
  /*
  printf( "collect: %5.3lf, gemm: %5.3lf, square: %5.3lf, heap: %5.3lf sec\n", 
      time_collect, time_dgemm, time_square, time_heap );
  */


  //I: output -> pxn
  //output I...


//  printf( "HeapSort result:\nI:\n" );
//  for ( j = 0; j < n; j ++ ) {
//    printf( "Col %d:\t", j );
//    for ( i = 0; i < r; i++ ) {
//      printf( "%d ", I[ j * r + i ] );
//    }
//    printf( "\n" );
//  }


//Reference Result with QSORT
//C: mxn

//  for ( j = 0; j < n; j ++ ) {
//    qsort(&Cs[j*m], m, sizeof(double), comp);
//  }
//
//  printf("Reference Result:\n");
//  for ( j = 0; j < n; j ++) {
//    printf("Col %d:\t", j);
//    for ( i = 0; i < m; i++ ) {
//      printf("%lf ", Cs[j * m + i]);
//    }
//    printf("\n");
//  }








  // Free the temporary buffers
  free( As );
  free( Bs );
  free( Cs );
}




