#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#include <gsknn.h>
#include <gsknn_config.h>


/*
 *
 *
 */ 
inline void swap_float( float *x, int i, int j ) {
  float  tmp = x[ i ];
  x[ i ] = x[ j ];
  x[ j ] = tmp;
}

inline void swap_double( double *x, int i, int j ) {
  double tmp = x[ i ];
  x[ i ] = x[ j ];
  x[ j ] = tmp;
}

inline void swap_int( int *x, int i, int j ) {
  int    tmp = x[ i ];
  x[ i ] = x[ j ];
  x[ j ] = tmp;
}



/*
 *
 *
 */ 
void bubbleSort_s(
    int    n,
    float  *D,
    int    *I
    )
{
  int    i, j;

  for ( i = 0; i < n - 1; i ++ ) {
    for ( j = 0; j < n - 1 - i; j ++ ) {
      if ( D[ j ] > D[ j + 1 ] ) {
        swap_float( D, j, j + 1 );
        swap_int( I, j, j + 1 );
      }
    }
  }
}

void bubbleSort_d(
    int    n,
    double *D,
    int    *I
    )
{
  int    i, j;

  for ( i = 0; i < n - 1; i ++ ) {
    for ( j = 0; j < n - 1 - i; j ++ ) {
      if ( D[ j ] > D[ j + 1 ] ) {
        swap_double( D, j, j + 1 );
        swap_int( I, j, j + 1 );
      }
    }
  }
}



/*
 *
 *
 */ 
double *gsknn_malloc_aligned(
    int    m,
    int    n,
    int    size
    )
{
  double *ptr;
  int    err;

  err = posix_memalign( (void**)&ptr, (size_t)KNN_SIMD_ALIGN_SIZE, size * m * n );

  if ( err ) {
    printf( "gsknn_malloc_aligned(): posix_memalign() failures" );
    exit( 1 );    
  }

  return ptr;
}
