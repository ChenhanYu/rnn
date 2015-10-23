#include <stdio.h>
#include <stdlib.h>
#include <immintrin.h>

#include <gsknn.h>
#include <gsknn_config.h>


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


inline void swap_int( int *I, int i, int j ) {
  int    tmp = I[ i ];
  I[ i ] = I[ j ];
  I[ j ] = tmp;
}


// Maintain a max heap
#ifdef KNN_PREC_SINGLE 
inline void HeapAdjust_s
#else
inline void HeapAdjust_d
#endif
    (
    prec_t *D, 
    int    s, 
    int    n, 
    int    *I
    ) 
{
  int    j;

  while ( 2 * s + 1 < n ) {
    j = 2 * s + 1;
    if ( ( j + 1 ) < n ) {
      if ( D[ j ] < D[ j + 1 ] ) j ++;
    }
    if ( D[ s ] < D[ j ] ) {
#ifdef KNN_PREC_SINGLE 
      swap_float( D, s, j );
#else
      swap_double( D, s, j );
#endif
      swap_int( I, s, j );
      s = j;
    } 
    else {
      break;
    }
  }
}


// Heap Sort the first largest r elements in an double array of length len)
#ifdef KNN_PREC_SINGLE 
inline void heap_sort_s
#else
inline void heap_sort_d
#endif
    (
    int    m,
    int    r,
    prec_t *x, 
    int    *alpha, 
    prec_t *D,
    int    *I
    ) 
{
  int    i;

  __asm__ volatile( "prefetcht0 0(%0)    \n\t" : :"r"( x ) );
  __asm__ volatile( "prefetcht0 0(%0)    \n\t" : :"r"( alpha ) );
  __asm__ volatile( "prefetcht0 0(%0)    \n\t" : :"r"( D ) );
  __asm__ volatile( "prefetcht0 0(%0)    \n\t" : :"r"( I ) );

  //For the rest n-r elements, compare with the first element in the min heap
  //if x[i] < x[0], then x[i] will not be in the largest r elements;
  //else replace x[0] (the minimum number in the largest r elements) with x[i], and maintain the min heap
  for ( i = 0; i < m; i ++ ) {
    if ( x[ i ] > D[ 0 ] ) {
      continue;
    }
    else {
      D[ 0 ] = x[ i ];  
      I[ 0 ] = alpha[ i ];
#ifdef KNN_PREC_SINGLE 
      HeapAdjust_s( D, 0, r, I );
#else
      HeapAdjust_d( D, 0, r, I );
#endif
    }
  }
}


heap_t *gsknn_heapAttach(
    int    m,
    int    k,
    double *D,
    int    *I
    )
{
  heap_t *heap = malloc( sizeof(heap_t) );
  heap->m   = m;
  heap->k   = k;
  heap->d   = 2;
  heap->ro  = 0.0;
  heap->ldk = k;
  heap->D   = D;
  heap->I   = I;
  return heap;
}


heap_t *gsknn_heapCreate(
    int    m,
    int    k,
    double ro
    )
{
  int    ldk, i, j;
 
  heap_t *heap = malloc( sizeof(heap_t) );

  if ( k > KNN_VAR_THRES ) {
    ldk = ( ( k + KNN_HEAP_OFFSET - 1 ) / 4 + 1 ) * 4;
    heap->d = 4;
  }
  else {
    ldk = k;
    heap->d = 2;
  }


  heap->m   = m;
  heap->k   = k;
  heap->ro  = ro;
  heap->ldk = ldk;

  //printf( "ldk = %d\n", ldk );

  if ( posix_memalign( (void**)&(heap->D), (size_t)DKNN_SIMD_ALIGN_SIZE, 
        sizeof(double) * ldk * m ) ) {
    printf( "gsknn_heapCreate(): posix_memalign() failures" );
    exit( 1 );    
  }

  if ( posix_memalign( (void**)&(heap->I), (size_t)DKNN_SIMD_ALIGN_SIZE, 
        sizeof(int) * ldk * m ) ) {
    printf( "gsknn_heapCreate(): posix_memalign() failures" );
    exit( 1 );    
  }
  

  //printf( "Create finish\n" );

  if ( k > KNN_VAR_THRES ) {
    for ( i = 0; i < m; i ++ ) {
      heap->D[ i * ldk     ] = ro;   // filter radius
      heap->D[ i * ldk + 1 ] = 0.0;  // Currently useless
      heap->D[ i * ldk + 2 ] = 0.0;  // ..
      heap->I[ i * ldk     ] = k;
      heap->I[ i * ldk + 1 ] = 0;
      heap->I[ i * ldk + 2 ] = ldk;  //

      for ( j = 0; j < k; j ++ ) {
        heap->D[ i * ldk + 3 + j ] = ro;
        heap->I[ i * ldk + 3 + j ] = -1;
      }
    }
  }
  else {
    for ( i = 0; i < m; i ++ ) {
      for ( j = 0; j < k; j ++ ) {
        heap->D[ i * ldk + j ] = ro;
        heap->I[ i * ldk + j ] = -1;
      }
    }
  }

  return heap;
}
