#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <iostream>
#include <vector>
#include <omp.h>
#include <rnn.h>

#include <dgsrnn_directKQuery.hpp>

void test_dgsrnn_directKQuery(
    int m,
    int n,
    int d,
    int k
    ) 
{

  int    i, j, p;
  double *XA, *XB;
  double ref_beg, ref_time, dgsrnn_beg, dgsrnn_time;
  
  std::pair<double, long> *result = new std::pair<double, long>[ m * k ];

  XA    = (double*)malloc( sizeof(double) * m * d );
  XB    = (double*)malloc( sizeof(double) * n * d );


  // random[ 0, 0.1 ]
  for ( i = 0; i < m; i ++ ) {
    for ( p = 0; p < d; p ++ ) {
      XA[ i * d + p ] = (double)( rand() % 100 ) / 1000.0;	
    }
  }


  // random[ 0, 0.1 ]
  for ( j = 0; j < n; j ++ ) {
    for ( p = 0; p < d; p ++ ) {
      XB[ j * d + p ] = (double)( rand() % 100 ) / 1000.0;	
    }
  }


  dgsrnn_beg = omp_get_wtime();

  dgsrnn_directKQuery(
      XB,    // ref
      XA,    // query
      n,
      m,
      k,
      d,     // dim
      result,
      NULL,
      NULL,
      NULL
      );

  dgsrnn_time = omp_get_wtime() - dgsrnn_beg;


  ref_beg = omp_get_wtime();
  ref_time = omp_get_wtime() - ref_beg;

  
  //for ( i = 0; i < m; i ++ ) {
  //  for ( j = 0; j < k; j ++ ) {
  //    std::cout << "(" << result[ i * k + j ].first;
  //    std::cout << "," << result[ i * k + j ].second;
  //    std::cout << ")";
  //  }
  //  std::cout << "\n";
  //}


  delete result;

  //free( XA );
  //free( XB );
}

int main( int argc, char *argv[] )
{
  int    m, n, k, r; 

  if ( argc != 5 ) {
    printf("argc: %d\n", argc);
    printf("we need 4 arguments!\n");
    exit(0);
  }

  sscanf( argv[ 1 ], "%d", &m );
  sscanf( argv[ 2 ], "%d", &n );
  sscanf( argv[ 3 ], "%d", &k );
  sscanf( argv[ 4 ], "%d", &r );

  test_dgsrnn_directKQuery( m, n, k, r );


  return 0;
}
