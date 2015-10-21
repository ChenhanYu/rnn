#include <stdio.h>
#include <stdlib.h>
#include <immintrin.h>

#include <gsknn.h>

#define RNN_HEAP_OFFSET 3
#define DARRAY 4

inline void swap_float( float *x, int i, int j ) {
  float  tmp = x[i];
  x[ i ] = x[ j ];
  x[ j ] = tmp;
}


inline void swap_double( double *x, int i, int j ) {
  double tmp = x[i];
  x[ i ] = x[ j ];
  x[ j ] = tmp;
}


inline void swap_int( int *I, int i, int j ) {
  int    temp = I[ i ];
  I[ i ] = I[ j ];
  I[ j ] = temp;
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


heap_t *rnn_heapAttach(
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

heap_t *rnn_heapCreate(
    int    m,
    int    k,
    double ro
    )
{
  int    ldk, i, j;
 
  heap_t *heap = malloc( sizeof(heap_t) );

  if ( k > RNN_VAR_THRES ) {
    ldk = ( ( k + RNN_HEAP_OFFSET - 1 ) / 4 + 1 ) * 4;
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

  if ( posix_memalign( (void**)&(heap->D), (size_t)DRNN_SIMD_ALIGN_SIZE, 
        sizeof(double) * ldk * m ) ) {
    printf( "rnn_heapCreate(): posix_memalign() failures" );
    exit( 1 );    
  }

  if ( posix_memalign( (void**)&(heap->I), (size_t)DRNN_SIMD_ALIGN_SIZE, 
        sizeof(int) * ldk * m ) ) {
    printf( "rnn_heapCreate(): posix_memalign() failures" );
    exit( 1 );    
  }
  

  //printf( "Create finish\n" );

  if ( k > RNN_VAR_THRES ) {
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



void heapSelect_int_d4(
    int    m,
    int    k,
    double *key,
    int    *val,
    double *D,
    int    *I
    )
{
  int    i, j, s, p, l, l_left;

  v4df_t d0, p0, p1, p2, p3;

  for ( i = 0; i < m; i++ ) {
    if ( key[ i ] < D[ RNN_HEAP_OFFSET ] ) {
      D[ RNN_HEAP_OFFSET ] = key[ i ];
      I[ RNN_HEAP_OFFSET ] = val[ i ];
      
      s = 0;

      while ( 4 * s + 1 < k ) {

        j = 4 * s + 1;
        l = k - j;

        l_left = l % 4;
        l = l - l_left;

        d0.v    = _mm256_load_pd( D + RNN_HEAP_OFFSET + j );
        p0.v    = _mm256_permute_pd( d0.v, 0x5 );      // 1 0 3 2
        p1.v    = _mm256_max_pd( p0.v, d0.v );
        p2.v    = _mm256_permute2f128_pd( p1.v, p1.v, 0x1 ); // 3 2 1 0
        p3.v    = _mm256_max_pd( p2.v, p1.v );

        d0.v    = _mm256_cmp_pd( d0.v, p3.v, 0 );

        j +=  __builtin_ctz( _mm256_movemask_pd( d0.v ) );

        if ( D[ s + RNN_HEAP_OFFSET ] < D[ j + RNN_HEAP_OFFSET ] ) {
          swap_double( D, s + RNN_HEAP_OFFSET, j + RNN_HEAP_OFFSET );
          swap_int( I, s + RNN_HEAP_OFFSET, j + RNN_HEAP_OFFSET );
          s = j;
        } 
        else {
          break;
        }
      }
    }
  }
}



void heapSelect_int_d16(
    int    m,
    int    k,
    double *key,
    int    *val,
    double *D,
    int    *I
    )
{
  int    i, j, s, p, l, l_left;
  //double max_key;

  v4df_t d0, d1, d2, d3;
  v4df_t p0, p1, p2, p3;

  for ( i = 0; i < m; i++ ) {
    if ( key[ i ] < D[ 3 ] ) {
      D[ 3 ] = key[ i ];
      I[ 3 ] = val[ i ];
      
      s = 0;

      while ( 16 * s + 1 < k ) {

        j = 16 * s + 4;

        l_left = l % 4;
        l = l - l_left;

        d0.v    = _mm256_load_pd( D + j      );
        d1.v    = _mm256_load_pd( D + j +  4 );
        d2.v    = _mm256_load_pd( D + j +  8 );
        d3.v    = _mm256_load_pd( D + j + 12 );


        p0.v    = _mm256_max_pd( d0.v, d1.v );
        p1.v    = _mm256_max_pd( d2.v, d3.v );
        p3.v    = _mm256_max_pd( p0.v, p1.v );
        
        
        p0.v    = _mm256_permute_pd( p3.v, 0x5 );      // 1 0 3 2
        p1.v    = _mm256_max_pd( p0.v, p3.v );
        p2.v    = _mm256_permute2f128_pd( p0.v, p0.v, 0x1 ); // 3 2 1 0
        p3.v    = _mm256_max_pd( p2.v, p1.v );

        d0.v    = _mm256_cmp_pd( d0.v, p3.v, 0 );
        d1.v    = _mm256_cmp_pd( d1.v, p3.v, 0 );
        d2.v    = _mm256_cmp_pd( d2.v, p3.v, 0 );
        d3.v    = _mm256_cmp_pd( d3.v, p3.v, 0 );

        if ( !_mm256_testz_pd( d0.v, d0.v ) ) {
          j = __builtin_ctz(_mm256_movemask_pd( d0.v ));
        }
        else if ( !_mm256_testz_pd( d1.v, d1.v ) ) {
          j = __builtin_ctz(_mm256_movemask_pd( d1.v ));
        }
        else if ( !_mm256_testz_pd( d2.v, d2.v ) ) {
          j = __builtin_ctz(_mm256_movemask_pd( d2.v ));
        }
        else if ( !_mm256_testz_pd( d3.v, d3.v ) ) {
          j = __builtin_ctz(_mm256_movemask_pd( d3.v ));
        }

        if ( D[ s ] < D[ j ] ) {
          swap_double( D, s, j );
          swap_int( I, s, j );
          s = j;
        } 
        else {
          break;
        }
      }
    }
  }
}

void heapSelect_dheap(
    int    m,
    int    k,
    double *key,
    int    *val,
    double *D,
    int    *I
    )
{
  int    i, j, s, p, l, l_left;
  
  
  for ( i = 0; i < m; i++ ) {
    if ( key[ i ] < D[ RNN_HEAP_OFFSET ] ) {
      D[ RNN_HEAP_OFFSET ] = key[ i ];
      I[ RNN_HEAP_OFFSET ] = val[ i ];
      
      s = 0;

      while ( DARRAY * s + 1 < k ) {

        // First child
        j = DARRAY * s + 1;

        l = k - j;

        if ( l > DARRAY ) l = DARRAY;

        // Find the max child
        for ( p = DARRAY * s + 2; p < DARRAY * s + 1 + l; p ++ ) {
          if ( D[ p + RNN_HEAP_OFFSET ] > D[ j + RNN_HEAP_OFFSET ] ) {
            j = p;
          }
        }

        if ( D[ s + RNN_HEAP_OFFSET ] < D[ j + RNN_HEAP_OFFSET ] ) {
          swap_double( D, s + RNN_HEAP_OFFSET, j + RNN_HEAP_OFFSET );
          swap_int( I, s + RNN_HEAP_OFFSET, j + RNN_HEAP_OFFSET );
          s = j;
        } 
        else {
          break;
        }
      }
    }
  }
}

inline void HeapAdjust_int_d4(
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
      d0.v    = _mm256_load_pd( D + RNN_HEAP_OFFSET + j );
      p0.v    = _mm256_permute_pd( d0.v, 0x5 );      // 1 0 3 2
      p1.v    = _mm256_max_pd( p0.v, d0.v );
      p2.v    = _mm256_permute2f128_pd( p1.v, p1.v, 0x1 ); // 3 2 1 0
      p3.v    = _mm256_max_pd( p2.v, p1.v );
      d0.v    = _mm256_cmp_pd( d0.v, p3.v, 0 );
      j +=  __builtin_ctz( _mm256_movemask_pd( d0.v ) );


      __asm__ volatile( "prefetcht0 0(%0)    \n\t" : :"r"( D + RNN_HEAP_OFFSET + j * 4 + 1 ) );
      __asm__ volatile( "prefetcht0 0(%0)    \n\t" : :"r"( I + RNN_HEAP_OFFSET + j * 4 + 1 ) );
    }
    else {
      for ( p = 4 * s + 2; p < 4 * s + 1 + l; p ++ ) {
        if ( D[ p + RNN_HEAP_OFFSET ] > D[ j + RNN_HEAP_OFFSET ] ) {
          j = p;
        }
      }
    }

    if ( D[ s + RNN_HEAP_OFFSET ] < D[ j + RNN_HEAP_OFFSET ] ) {
      swap_double( D, s + RNN_HEAP_OFFSET, j + RNN_HEAP_OFFSET );
      swap_int( I, s + RNN_HEAP_OFFSET, j + RNN_HEAP_OFFSET );
      s = j;
    } 
    else {
      break;
    }

  }
}




//Maintain a max heap
inline void HeapAdjust_dheap(
    double *D, 
    int    s, 
    int    k, 
    int    *I
    ) 
{

  int j, l, p;

  while ( DARRAY * s + 1 < k ) {

    // First child
    j = DARRAY * s + 1;

    l = k - j;

    if ( l > DARRAY ) l = DARRAY;

    // Find the max child
    for ( p = DARRAY * s + 2; p < DARRAY * s + 1 + l; p ++ ) {
      if ( D[ p + RNN_HEAP_OFFSET ] > D[ j + RNN_HEAP_OFFSET ] ) {
        j = p;
      }
    }

    if ( D[ s + RNN_HEAP_OFFSET ] < D[ j + RNN_HEAP_OFFSET ] ) {
      swap_double( D, s + RNN_HEAP_OFFSET, j + RNN_HEAP_OFFSET );
      swap_int( I, s + RNN_HEAP_OFFSET, j + RNN_HEAP_OFFSET );
      s = j;
    } 
    else {
      break;
    }
  }


}


void heapSelect_dheap_var2(
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
  __asm__ volatile( "prefetcht0 0(%0)    \n\t" : :"r"( D + RNN_HEAP_OFFSET ) );
  __asm__ volatile( "prefetcht0 0(%0)    \n\t" : :"r"( I + RNN_HEAP_OFFSET ) );


  for ( i = 0; i < m; i++ ) {
    if ( I[ 1 ] < I[ 0 ] ) {

      int p = I[ 0 ] - 1 - I[ 1 ];



      //if ( p * DARRAY >= k - 1 ) {
      //  D[ p + RNN_HEAP_OFFSET ] = key[ i ];
      //  I[ p + RNN_HEAP_OFFSET ] = val[ i ];
      //} else {
      //  D[ p + RNN_HEAP_OFFSET ] = key[ i ];
      //  I[ p + RNN_HEAP_OFFSET ] = val[ i ];
      //  //HeapAdjust_dheap( D, p, k, I );
      //  HeapAdjust_int_d4( D, p, k, I );
      //}


      D[ p + RNN_HEAP_OFFSET ] = key[ i ];
      I[ p + RNN_HEAP_OFFSET ] = val[ i ];

      if ( p * DARRAY < k - 1 ) {
        //HeapAdjust_dheap( D, p, k, I );
        HeapAdjust_int_d4( D, p, k, I );
      }

      I [ 1 ] = I[ 1 ] + 1;
    } 
    else {

      if ( key[ i ] < D[ RNN_HEAP_OFFSET ] ) {
        D[ RNN_HEAP_OFFSET ] = key[ i ];
        I[ RNN_HEAP_OFFSET ] = val[ i ];


        //HeapAdjust_dheap( D, 0, k, I );
        HeapAdjust_int_d4( D, 0, k, I );
      }
    }
  }
}

void heapSelect_d16(
    int    m,
    int    k,
    double *key,
    int    *val,
    double *D,
    int    *I
    )
{
  int    i, j, s, p, l, l_left;
  //double max_key;
  
  //printf("come inside select_d16\n");


  for ( i = 0; i < m; i++ ) {
    if ( key[ i ] < D[ 0 ] ) {
      D[ 0 ] = key[ i ];
      I[ 0 ] = val[ i ];
      
      s = 0;

      while ( 16 * s + 1 + RNN_HEAP_OFFSET - 1 < k + RNN_HEAP_OFFSET - 1 ) {

        j = 16 * s + 1;
        l = k - j;

        if ( l > 16 ) l = 16;

        // Find the max child
        for ( p = 16 * s + 2; p < 16 * s + 1 + l; p ++ ) {
          if ( D[ p + RNN_HEAP_OFFSET - 1 ] > D[ j + RNN_HEAP_OFFSET - 1 ] ) {
            j = p;
          }
        }

        if ( D[ s + RNN_HEAP_OFFSET - 1 ] < D[ j + RNN_HEAP_OFFSET - 1 ] ) {
          swap_double( D, s + RNN_HEAP_OFFSET - 1, j + RNN_HEAP_OFFSET - 1);
          swap_int( I, s + RNN_HEAP_OFFSET - 1, j + RNN_HEAP_OFFSET - 1 );
          s = j;
        } 
        else {
          break;
        }
      }
    }
  }
}

