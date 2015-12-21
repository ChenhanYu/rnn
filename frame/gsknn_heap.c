/*
 * --------------------------------------------------------------------------
 * GSKNN (General Stride K-Nearest Neighbors)
 * --------------------------------------------------------------------------
 * Copyright (C) 2015, The University of Texas at Austin
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 *
 *
 * gsknn_heap.c
 *
 * Chenhan D. Yu - Department of Computer Science,
 *                 The University of Texas at Austin
 *
 *
 * Purpose:
 * Implement binary heap sort, adjust and it's contructor.
 *
 * Todo:
 *
 *
 * Modification:
 *
 * 
 * */


#include <stdio.h>
#include <stdlib.h>
#include <immintrin.h>

#include <gsknn.h>
#include <gsknn_config.h>


/*
 * Maintain a max heap
 *
 */ 
inline void HeapAdjust_s(
    float  *D, 
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
      swap_float( D, s, j );
      swap_int( I, s, j );
      s = j;
    } 
    else break;
  }
}

inline void HeapAdjust_d(
    double *D, 
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
      swap_double( D, s, j );
      swap_int( I, s, j );
      s = j;
    } 
    else break;
  }
}



/*
 * Heap Sort the first largest r elements in an double array of length len)
 *
 */ 
inline void heapSelect_s(
    int    m,
    int    r,
    float  *x, 
    int    *alpha, 
    float  *D,
    int    *I
    ) 
{
  int    i;

  __asm__ volatile( "prefetcht0 0(%0)    \n\t" : :"r"( x ) );
  __asm__ volatile( "prefetcht0 0(%0)    \n\t" : :"r"( alpha ) );
  __asm__ volatile( "prefetcht0 0(%0)    \n\t" : :"r"( D ) );
  __asm__ volatile( "prefetcht0 0(%0)    \n\t" : :"r"( I ) );

  for ( i = 0; i < m; i ++ ) {
    if ( x[ i ] > D[ 0 ] ) {
      continue;
    }
    else {
      D[ 0 ] = x[ i ];  
      I[ 0 ] = alpha[ i ];
      HeapAdjust_s( D, 0, r, I );
    }
  }
}


inline void heapSelect_d(
    int    m,
    int    r,
    double *x, 
    int    *alpha, 
    double *D,
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
      HeapAdjust_d( D, 0, r, I );
    }
  }
}


/*
 *
 *
 */ 
heap_t *heapAttach_s(
    int    m,
    int    k,
    float  *D,
    int    *I
    )
{
  heap_t *heap = malloc( sizeof(heap_t) );
  heap->m    = m;
  heap->k    = k;
  heap->d    = 2;
  heap->ro_s = 0.0;
  heap->ldk  = k;
  heap->D_s  = D;
  heap->I    = I;
  heap->type = KNN_2NORM;
  heap->prec = KNN_SINGLE;
  return heap;
}

heap_t *heapAttach_d(
    int    m,
    int    k,
    double *D,
    int    *I
    )
{
  heap_t *heap = malloc( sizeof(heap_t) );
  heap->m    = m;
  heap->k    = k;
  heap->d    = 2;
  heap->ro   = 0.0;
  heap->ldk  = k;
  heap->D    = D;
  heap->I    = I;
  heap->type = KNN_2NORM;
  heap->prec = KNN_DOUBLE;
  return heap;
}



/*
 *
 *
 */ 
heap_t *heapCreate_s(
    int    m,
    int    k,
    float  ro_s
    )
{
  int    ldk, i, j;

  ldk = k;

  heap_t *heap = malloc( sizeof(heap_t) );
  heap->m    = m;
  heap->k    = k;
  heap->d    = 2;
  heap->ro_s = ro_s;
  heap->ldk  = ldk;
  heap->type = KNN_2NORM;
  heap->prec = KNN_SINGLE;

  heap->D_s = (float*)malloc( ldk * m * sizeof(float) );
  heap->I   = (int*)malloc( ldk * m * sizeof(int) );

  for ( i = 0; i < m; i ++ ) {
    for ( j = 0; j < k; j ++ ) {
      heap->D_s[ i * ldk + j ] = ro_s;
      heap->I[ i * ldk + j ]   = -1;
    }
  }

  return heap;
}

heap_t *heapCreate_d(
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
  heap->type = KNN_2NORM;
  heap->prec = KNN_DOUBLE;

  //printf( "ldk = %d\n", ldk );

  heap->D = (double*)gsknn_malloc_aligned( ldk, m, sizeof(double) );
  heap->I = (int*)gsknn_malloc_aligned( ldk, m, sizeof(int) );


  //if ( posix_memalign( (void**)&(heap->D), (size_t)DKNN_SIMD_ALIGN_SIZE, 
  //      sizeof(double) * ldk * m ) ) {
  //  printf( "heapCreate_d(): posix_memalign() failures" );
  //  exit( 1 );    
  //}

  //if ( posix_memalign( (void**)&(heap->I), (size_t)DKNN_SIMD_ALIGN_SIZE, 
  //      sizeof(int) * ldk * m ) ) {
  //  printf( "heapCreate_d(): posix_memalign() failures" );
  //  exit( 1 );    
  //}
  

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

void heapFree_s(
    heap_t *heap
    )
{
  free( heap->D_s );
  free( heap->I );
  free( heap );
}

void heapFree_d(
    heap_t *heap
    )
{
  free( heap->D );
  free( heap->I );
  free( heap );
}

