//#include <mkl.h>
#include <omp.h>
#include <stdlib.h>
#include <limits.h>
#include <float.h>
#include<vector>
#include<algorithm>
#include<utility>
#include<stdio.h>
#include<iostream>

extern "C" {
#include <rnn.h>
}
#include <dgsrnn_ref_stl.hpp>


// dgemm prototype
extern "C" void dgemm(char*, char*, int*, int*, int*, double*, double*, 
                   int*, double*, int*, double*, double*, int*);



// Heap operator
struct lessthan {
  bool operator()(const std::pair<int, double> &a, const std::pair<int, double> &b ) const {
    return a.second < b.second;
  }
};


// This reference function will call MKL
void dgsrnn_ref_stl(
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
  int    i, j, l, p;
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
#pragma omp parallel for private( i )
  for ( j = 0; j < n; j ++ ) {
    std::vector<std::pair<int, double> > myheap( r );
    for ( i = 0; i < r; i ++ ) {
      myheap[ i ] = std::make_pair( I[ j * r + i ], D[ j * r + i] );
    }
    for ( i = 0; i < m; i ++ ) {
      if ( myheap.front().second > Cs[ j * m + i] ) {
        myheap.front().first  = i;
        myheap.front().second = Cs[ j * m + i ];
        std::pop_heap( myheap.begin(), myheap.end(), lessthan() );
        std::push_heap( myheap.begin(), myheap.end(), lessthan() );
      }
    }
    for ( i = 0; i < r; i ++ ) {
      I[ j * r + i ] = myheap[ i ].first;
      D[ j * r + i ] = myheap[ i ].second;
    }
  }
  time_heap = omp_get_wtime() - beg;

  //printf( "Collect: %5.3lf, Gemm: %5.3lf, Sq2nrm: %5.3lf, Select: %5.3lf\n",
  //    time_collect, time_dgemm, time_square, time_heap );




  // Free the temporary buffers
  free( As );
  free( Bs );
  free( Cs );
}
