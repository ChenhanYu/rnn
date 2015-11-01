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
 * gsknn_util.c
 *
 * Chenhan D. Yu - Department of Computer Science,
 *                 The University of Texas at Austin
 *
 *
 * Purpose:
 * Implement bubble sort for error check. All binary comparision functions.
 * Aligned memory allocation.
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
