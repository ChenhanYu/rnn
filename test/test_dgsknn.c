#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <limits.h>

#include <gsknn.h>
#include <gsknn_ref.h>

void bubble_sort(
    int    r,
    double *D,
    int    *I
    )
{
  int    i, j;

  for ( i = 0; i < r - 1; i ++ ) {
    for ( j = 0; j < r - 1 - i; j ++ ) {
       if ( D[ j ] > D[ j + 1 ] ) {
         double dtmp;
         int    itmp;
         dtmp = D[ j ];
         D[ j ] = D[ j + 1 ];
         D[ j + 1 ] = dtmp;
         itmp = I[ j ];
         I[ j ] = I[ j + 1 ];
         I[ j + 1 ] = itmp;
       }
    }
  }
}

void compute_error(
    int    r,
    int    n,
    double *D,
    int    *I,
    double *D_gold,
    int    *I_gold
    )
{
  int    i, j;
  double *D1, *D2;
  int    *I1, *I2;

  D1 = (double*)malloc( sizeof(double) * r * n );
  D2 = (double*)malloc( sizeof(double) * r * n );
  I1 = (int*)malloc( sizeof(int) * r * n );
  I2 = (int*)malloc( sizeof(int) * r * n );


  // bubble sort
  for ( j = 0; j < n; j ++ ) {
    for ( i = 0; i < r; i ++ ) {
      D1[ j * r + i ] = D[ j * r + i ];
      I1[ j * r + i ] = I[ j * r + i ];
      D2[ j * r + i ] = D_gold[ j * r + i ];
      I2[ j * r + i ] = I_gold[ j * r + i ];
    }
    bubble_sort( r, &D1[ j * r ], &I1[ j * r ] );
    bubble_sort( r, &D2[ j * r ], &I2[ j * r ] );
  }

  for ( j = 0; j < n; j ++ ) {
    for ( i = 0; i < r; i ++ ) {
      if ( I1[ j * r + i ] != I2[ j * r + i ] ) {
        if ( fabs( D1[ j * r + i ] - D2[ j * r + i ] ) > 1E-13 ) {
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



void test_dgsknn(
    int m,
    int n,
    int k,
    int r
    ) 
{
  int    i, j, p, nx, iter, n_iter;
  int    *amap, *bmap, *I, *I_mkl;
  double *XA, *XB, *XA2, *XB2, *D, *D_mkl;
  double tmp, error, flops;
  double ref_beg, ref_time, dgsknn_beg, dgsknn_time;

  nx     = 4096 * 5;
  n_iter = 1;


  amap  = (int*)malloc( sizeof(int) * m );
  bmap  = (int*)malloc( sizeof(int) * n );
  I     = (int*)malloc( sizeof(int) * r * n );
  I_mkl = (int*)malloc( sizeof(int) * r * n );
  XA    = (double*)malloc( sizeof(double) * k * nx );
  XA2   = (double*)malloc( sizeof(double) * nx ); 
  D     = (double*)malloc( sizeof(double) * r * n );
  D_mkl = (double*)malloc( sizeof(double) * r * n );


  heap_t *heap = heapCreate_d( n, r, 1.79E+308 );

  for ( i = 0; i < m; i ++ ) {
    amap[ i ] = 2 * i;
  }

  for ( j = 0; j < n; j ++ ) {
    bmap[ j ] = 2 * j + 1;
  }


  // random[ 0, 0.1 ]
  for ( i = 0; i < nx; i ++ ) {
    for ( p = 0; p < k; p ++ ) {
      XA[ i * k + p ] = (double)( rand() % 1000 ) / 1000.0;	
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


  // Initialize D to the maximum double and I to -1.
  for ( j = 0; j < n; j ++ ) {
    for ( i = 0; i < r; i ++ ) {
      D[ j * r + i ]     = 1.79E+308;
      D_mkl[ j * r + i ] = 1.79E+308;
      I[ j * r + i ]     = -1;
      I_mkl[ j * r + i ] = -1;
    }
  }


  dgsknn_beg = omp_get_wtime();
  {
    dgsknn(
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
  dgsknn_time = omp_get_wtime() - dgsknn_beg;


  ref_beg = omp_get_wtime();
  {
    dgsknn_ref(
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



  if ( r > KNN_VAR_THRES ) { 
    for ( j = 0; j < n; j ++ ) {
      for ( i = 0; i < r; i ++ ) {
        D[ j * r + i ] = heap->D[ j * heap->ldk + i + 3 ];
        I[ j * r + i ] = heap->I[ j * heap->ldk + i + 3 ];
      }
    }
  }
  else {
    for ( j = 0; j < n; j ++ ) {
      for ( i = 0; i < r; i ++ ) {
        D[ j * r + i ] = heap->D[ j * heap->ldk + i ];
        I[ j * r + i ] = heap->I[ j * heap->ldk + i ];
      }
    }
  }


  // Compute error
  compute_error(
      r,
      n,
      D,
      I,
      D_mkl,
      I_mkl
      );



  ref_time    /= ( n_iter - 0 );
  dgsknn_time /= ( n_iter - 0 );
  flops = ( m * n / ( 1024.0 * 1024.0 * 1024.0 ) )* ( 2 * k + 3 );


  printf( "%d, %d, %d, %d, %5.2lf, %5.2lf;\n", 
      m, n, k, r, flops / dgsknn_time, flops / ref_time );
  //printf( "%d, %d, %d, %d, %5.2lf, %5.2lf;\n", 
  //    m, n, k, r, dgsknn_time, ref_time );


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
  //printf("start the test dgsknn!\n");
  fflush( stdout );



  if ( argc != 5 ) {
    printf("argc: %d\n", argc);
    printf("we need 4 arguments!\n");
    exit(0);
  }

  sscanf( argv[ 1 ], "%d", &m );
  sscanf( argv[ 2 ], "%d", &n );
  sscanf( argv[ 3 ], "%d", &k );
  sscanf( argv[ 4 ], "%d", &r );


  test_dgsknn( m, n, k, r );


  return 0;
}
