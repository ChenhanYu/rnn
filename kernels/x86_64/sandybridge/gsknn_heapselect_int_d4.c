#include <gsknn.h>
#include <gsknn_config.h>
#include <immintrin.h> // AVX
#include <avx_types.h>

inline void heapadjust_int_d4(
    double *D, 
    int    s, 
    int    k, 
    int    *I
    ) 
{

  int     j, l, p;
  v4df_t  d0, p0, p1, p2, p3;

  while ( 4 * s + 1 < k ) {

    // First child
    j = 4 * s + 1;
    l = k - j;

    if ( l > 4 ) {
      d0.v    = _mm256_load_pd( D + KNN_HEAP_OFFSET + j );
      p0.v    = _mm256_permute_pd( d0.v, 0x5 );      // 1 0 3 2
      p1.v    = _mm256_max_pd( p0.v, d0.v );
      p2.v    = _mm256_permute2f128_pd( p1.v, p1.v, 0x1 ); // 3 2 1 0
      p3.v    = _mm256_max_pd( p2.v, p1.v );
      d0.v    = _mm256_cmp_pd( d0.v, p3.v, 0 );
      j +=  __builtin_ctz( _mm256_movemask_pd( d0.v ) );


      __asm__ volatile( "prefetcht0 0(%0)    \n\t" : :"r"( D + KNN_HEAP_OFFSET + j * 4 + 1 ) );
      __asm__ volatile( "prefetcht0 0(%0)    \n\t" : :"r"( I + KNN_HEAP_OFFSET + j * 4 + 1 ) );
    }
    else {
      for ( p = 4 * s + 2; p < 4 * s + 1 + l; p ++ ) {
        if ( D[ p + KNN_HEAP_OFFSET ] > D[ j + KNN_HEAP_OFFSET ] ) {
          j = p;
        }
      }
    }

    if ( D[ s + KNN_HEAP_OFFSET ] < D[ j + KNN_HEAP_OFFSET ] ) {
      swap_double( D, s + KNN_HEAP_OFFSET, j + KNN_HEAP_OFFSET );
      swap_int( I, s + KNN_HEAP_OFFSET, j + KNN_HEAP_OFFSET );
      s = j;
    } 
    else {
      break;
    }

  }
}



void gsknn_heapselect_int_d4(
    int    m,
    int    k,
    double *key,
    int    *val,
    double *D,
    int    *I
    )
{
  int    i, j, s, p, l, l_left;

  // prefetch the head of key
  __asm__ volatile( "prefetcht0 0(%0)    \n\t" : :"r"( key ) );
  __asm__ volatile( "prefetcht0 0(%0)    \n\t" : :"r"( val ) );
  __asm__ volatile( "prefetcht0 0(%0)    \n\t" : :"r"( D + KNN_HEAP_OFFSET ) );
  __asm__ volatile( "prefetcht0 0(%0)    \n\t" : :"r"( I + KNN_HEAP_OFFSET ) );

  for ( i = 0; i < m; i++ ) {
    if ( I[ 1 ] < I[ 0 ] ) {
      int p = I[ 0 ] - 1 - I[ 1 ];
      D[ p + KNN_HEAP_OFFSET ] = key[ i ];
      I[ p + KNN_HEAP_OFFSET ] = val[ i ];
      if ( p * 4 < k - 1 ) {
        heapadjust_int_d4( D, p, k, I );
      }
      I [ 1 ] = I[ 1 ] + 1;
    } 
    else {
      if ( key[ i ] < D[ KNN_HEAP_OFFSET ] ) {
        D[ KNN_HEAP_OFFSET ] = key[ i ];
        I[ KNN_HEAP_OFFSET ] = val[ i ];
        heapadjust_int_d4( D, 0, k, I );
      }
    }
  }
}
