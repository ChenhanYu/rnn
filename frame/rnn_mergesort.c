#include <stdio.h>
#include <stdlib.h>

//#include <algorithm>
//#include <functional>
//#include <array>
//#include <iostream>

typedef unsigned long long dim_t;

//class rnnPair {
//  public:
//    double key;
//    int    val;
//
//    bool operator<(const rnnPair &a) const {
//      if ( key < a.key ) {
//        return true;
//      }
//      else {
//        return false;
//      }
//    };
//};
//
//extern "C"
//void mergesort_pack(
//    double *key,
//    int    *val,
//    double *D,
//    int    *I,
//    int    n,
//    int    k
//    )
//{
//  //typedef typename std::iterator_traits<rnnPair>
//
//  rnnPair *arr = new rnnPair[ n ];
//
//  for ( int i = 0; i < n; i++ ) {
//    arr[ i ].key = key[ i ];
//    arr[ i ].val = val[ i ];
//  }
//
//  //std::sort( &(arr[ 0 ]), &(arr[ n - 1 ]), rnnPair.less );
//  std::sort( &(arr[ 0 ]), &(arr[ n ]) );
//
//  for ( int i = 0; i < k; i++ ) {
//    D[ i ] = arr[ i ].key;
//    I[ i ] = arr[ i ].val;
//  }
//}


void rnn_merge_ref_var2(
    double *a,
    dim_t  *amap,
    double *b,
    dim_t *bmap,
    double *D,
    dim_t  *I,
    dim_t  na,
    dim_t  nb
    )
{
  dim_t  a_ptr = 0;
  dim_t  b_ptr = 0;
  dim_t  d_ptr = 0;

  while ( a_ptr < na && b_ptr < nb ) {
    if ( a[ a_ptr ] < b[ b_ptr ] ) {
      D[ d_ptr ] = a[ a_ptr ];    
      I[ d_ptr ] = amap[ a_ptr ];
      a_ptr ++;
    }
    else {
      D[ d_ptr ] = b[ b_ptr ];    
      I[ d_ptr ] = bmap[ b_ptr ];
      b_ptr ++;
    }
    d_ptr ++;
  }

  while ( a_ptr < na ) {
    D[ d_ptr ] = a[ a_ptr ];    
    I[ d_ptr ] = amap[ a_ptr ];
    a_ptr ++;
    d_ptr ++;
  }

  while ( b_ptr < nb ) {
    D[ d_ptr ] = b[ b_ptr ];    
    I[ d_ptr ] = bmap[ b_ptr ];
    b_ptr ++;
    d_ptr ++;
  }
}





void rnn_mergesort(
    dim_t  n,
    dim_t  k,
    double *key,
    dim_t  *val,
    double *D,
    dim_t  *I
    )
{
  double tkey[ 4096 ];
  dim_t  tval[ 4096 ];
  double *pkey, *skey;
  dim_t  *pval, *sval;
  dim_t  l, i;

  pkey = tkey;
  pval = tval;

  for ( l = 1; l < k; l *= 2 ) {
    // In this level, we merge two length l arrays.
    for ( i = 0; i < n; i += 2 * l ) {
      rnn_merge_ref_var2( 
          key  + i, 
          val  + i, 
          key  + i + l, 
          val  + i + l, 
          pkey + i,
          pval + i,
          l,
          l
          );
    }

    // swap pointers
    skey = key;
    sval = val;
    key  = pkey;
    val  = pval;
    pkey = skey;
    pval = sval;
  }

  // merge length k arrays
  for ( i = 0; i < k; i ++ ) {
    D[ i ] = key[ i ];
    I[ i ] = val[ i ];
  }
}

