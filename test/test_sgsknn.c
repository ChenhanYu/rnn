#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <limits.h>

#include <gsknn.h>
#include <gsknn_ref.h>

#define NUM_POINTS 10240
#define USE_SET_DIFF 1
#define TOLERANCE 1E-4

void computeError(
    int    r,
    int    n,
    float  *D,
    int    *I,
    float  *D_gold,
    int    *I_gold
    )
{
  int    i, j, p;
  float  *D1, *D2;
  int    *I1, *I2, *Set1, *Set2;

  if ( USE_SET_DIFF ) {
    Set1 = (int*)malloc( sizeof(int) * NUM_POINTS );
    Set2 = (int*)malloc( sizeof(int) * NUM_POINTS );

    // Check set equvilent.
    for ( j = 0; j < n; j ++ ) {
      for ( i = 0; i < NUM_POINTS; i ++ ) {
        Set1[ i ] = 0;
        Set2[ i ] = 0;
      }
      for ( i = 0; i < r; i ++ ) {
        p = I[ j * r + i ];
        Set1[ p ] = i;
        Set2[ p ] = 1;
      }
      for ( i = 0; i < r; i ++ ) {
        p = I_gold[ j * r + i ];
        if ( Set2[ p ] == 0 ) {
          Set1[ p ] = i;
          Set2[ p ] = 2;
        }
        else {
          Set2[ p ] = 0;
        }
      }
      for ( i = 0; i < NUM_POINTS; i ++ ) {
        if ( Set2[ i ] == 1 && D[ j * r ] != D[ j * r + Set1[ i ] ] ) {
          printf( "(%E,%E,%d,%d,%E,1,%d)\n", D[ j * r ], D_gold[ j * r ], 
              j, i, D[ j * r + Set1[ i ] ], I[ j * r ] );
        }
        if ( Set2[ i ] == 2 && D_gold[ j * r ] != D_gold[ j * r + Set1[ i ] ] ) {
          printf( "(%E,%E,%d,%d,%E,2,%d)\n", D[ j * r ], D_gold[ j * r ], 
              j, i, D_gold[ j * r + Set1[ i ] ], I_gold[ j * r ] );
          if ( D_gold[ j * r ] < D_gold[ j * r + Set1[ i ] ] ) {
            printf( "bug\n" );
          }
        }
      }
    }

    free( Set1 );
    free( Set2 );
  }
  else {
    D1 = (float*)malloc( sizeof(float) * r * n );
    D2 = (float*)malloc( sizeof(float) * r * n );
    I1 = (int*)malloc( sizeof(int) * r * n );
    I2 = (int*)malloc( sizeof(int) * r * n );

    // Check error using bubbleSort.
    for ( j = 0; j < n; j ++ ) {
      for ( i = 0; i < r; i ++ ) {
        D1[ j * r + i ] = D[ j * r + i ];
        I1[ j * r + i ] = I[ j * r + i ];
        D2[ j * r + i ] = D_gold[ j * r + i ];
        I2[ j * r + i ] = I_gold[ j * r + i ];
      }
      bubbleSort_s( r, &D1[ j * r ], &I1[ j * r ] );
      bubbleSort_s( r, &D2[ j * r ], &I2[ j * r ] );
    }

    for ( j = 0; j < n; j ++ ) {
      for ( i = 0; i < r; i ++ ) {
        if ( I1[ j * r + i ] != I2[ j * r + i ] ) {
          if ( fabs( D1[ j * r + i ] - D2[ j * r + i ] ) > TOLERANCE ) {
            printf( "D[ %d ][ %d ] != D_gold, %E, %E\n", i, j, D1[ j * r + i ], D2[ j * r + i ] );
            printf( "I[ %d ][ %d ] != I_gold, %d, %d\n", i, j, I1[ j * r + i ], I2[ j * r + i ] );
            break;
          }
        }
      }
    }

    free( D1 );
    free( D2 );
    free( I1 );
    free( I2 );
  }
}

void test_sgsknn(
    int m,
    int n,
    int k,
    int r
    ) 
{
  int    i, j, p, nx;
  int    *amap, *bmap, *I, *I_mkl;
  float  *XA, *XB, *XA2, *XB2, *D, *D_mkl;
  float  tmp, error;
  double flops, ref_beg, ref_time, gsknn_beg, gsknn_time;

  nx    = NUM_POINTS;

  amap  = (int*)malloc( sizeof(int) * m );
  bmap  = (int*)malloc( sizeof(int) * n );
  I     = (int*)malloc( sizeof(int) * r * n );
  I_mkl = (int*)malloc( sizeof(int) * r * n );
  XA    = (float*)malloc( sizeof(float) * k * nx );
  XA2   = (float*)malloc( sizeof(float) * nx ); 
  D     = (float*)malloc( sizeof(float) * r * n );
  D_mkl = (float*)malloc( sizeof(float) * r * n );

  // Initilize the heap structure.
  heap_t *heap = heapCreate_s( n, r, 1.79E+30 );

  // Assign reference indecies.
  for ( i = 0; i < m; i ++ ) {
    amap[ i ] = i;
  }

  // Assign query indecies.
  for ( j = 0; j < n; j ++ ) {
    bmap[ j ] = j;
  }

  // Randonly generate points in [ 0, 1 ].
  for ( i = 0; i < nx; i ++ ) {
    for ( p = 0; p < k; p ++ ) {
      XA[ i * k + p ] = (float)( rand() % 1000000 ) / 1000000.0;	
    }
  }

  // Compute XA2
  for ( i = 0; i < nx; i ++ ) {
    tmp = 0.0;
    for ( p = 0; p < k; p ++ ) {
      tmp += XA[ i * k + p ] * XA[ i * k + p ];
    }
    XA2[ i ] = tmp;
  }

  // Use the same coordinate table
  XB  = XA;
  XB2 = XA2;

  // Initialize D ( distance ) to the maximum double and I ( index ) to -1.
  for ( j = 0; j < n; j ++ ) {
    for ( i = 0; i < r; i ++ ) {
      D[ j * r + i ]     = 1.79E+30;
      I[ j * r + i ]     = -1;
      D_mkl[ j * r + i ] = 1.79E+30;
      I_mkl[ j * r + i ] = -1;
    }
  }

  gsknn_beg = omp_get_wtime();
  {
    sgsknn(
        m,
        n,
        k,
        r,
        XA,
        XA2,
        amap,
        XB,
        XB2,
        bmap,
        heap
        );
  }
  gsknn_time = omp_get_wtime() - gsknn_beg;

  ref_beg = omp_get_wtime();
  {
    sgsknn_ref(
        m,
        n,
        k,
        r,
        XA,
        XA2,
        amap,
        XB,
        XB2,
        bmap,
        D_mkl,
        I_mkl
        );
  }
  ref_time = omp_get_wtime() - ref_beg;



  // Reformat the neighbor pair for comparison.
  for ( j = 0; j < n; j ++ ) {
    for ( i = 0; i < r; i ++ ) {
      D[ j * r + i ] = heap->D_s[ j * heap->ldk + i ];
      I[ j * r + i ] = heap->I[ j * heap->ldk + i ];
    }
  }
  
  // Free the memory of the heap structure.
  heapFree_s( heap );

  // Compute error
  computeError(
      r,
      n,
      D,
      I,
      D_mkl,
      I_mkl
      );

  // Compute overall floating point operations.
  flops = ( m * n / ( 1000.0 * 1000.0 * 1000.0 ) ) * ( 2 * k + 3 );

  printf( "%5d, %5d, %5d, %5d, %5.2lf GFLOPS, %5.2lf GFLOPS;\n", 
      m, n, k, r, flops / gsknn_time, flops / ref_time );

  free( XA );
  free( XA2 );
  free( D );
  free( I );
  free( D_mkl );
  free( I_mkl );
}







int main( int argc, char *argv[] )
{
  int    m, n, k, r; 

  if ( argc != 5 ) {
    printf( "Error: require 4 arguments, but only %d provided.\n", argc - 1 );
    exit( 0 );
  }

  sscanf( argv[ 1 ], "%d", &m );
  sscanf( argv[ 2 ], "%d", &n );
  sscanf( argv[ 3 ], "%d", &k );
  sscanf( argv[ 4 ], "%d", &r );

  test_sgsknn( m, n, k, r );

  return 0;
}
