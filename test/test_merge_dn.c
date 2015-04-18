#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#include <rnn.h>

void rnn_merge_int_dn( double*, double*, double*, double*, double*, double*, int, int );
void rnn_merge_int_dn_var2( double*, double*, double*, double*, double*, double*, int, int );


void rnn_merge_ref(
    double *a,
    double *amap,
    double *b,
    double *bmap,
    double *D,
    double *I,
    int    na,
    int    nb
    )
{
  int    a_ptr = 0;
  int    b_ptr = 0;
  int    d_ptr = 0;

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



int main() 
{
  int    i, j;
  int    na = 4096;
  int    nb = 4096;
  double beg, ref_time, avx_time;
  int    iter = 1;
  //double a[ 8 ]    = {1.0, 3.0, 6.0, 8.0, 9.0, 14.0, 15.0, 18.0};
  //double amap[ 8 ] = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0};
  //double b[ 8 ]    = {2.0, 4.0, 5.0, 7.0, 10.0, 11.0, 12.0, 13.0};
  //double bmap[ 8 ] = {8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0};
  //double *d;
  //double *index;

  //d     = (double *) malloc ((a_len + b_len) * sizeof(double));
  //index = (double *) malloc ((a_len + b_len) * sizeof(double));

  double a[ 4096 ];
  double amap[ 4096 ];
  double b[ 4096 ];
  int bmap[ 4096 ];
  double D[ 8192 ];
  double D_ref[ 8192 ];
  double I[ 8192 ];
  int I_ref[ 8192 ];

  for ( j = 0; j < na; j ++ ) {
    D_ref[ 8192 ] = 10000000000.0;
    I_ref[ 8192 ] = -1;
  }


  for ( j = 0; j < na; j ++ ) {
    a[ j ] = na - j;
    amap[ j ] = na - j;
  }


  for ( j = 0; j < nb; j ++ ) {
    b[ j ] = nb - j;
    bmap[ j ] = nb - j;
  }

  beg = omp_get_wtime();
  for ( i = 0; i < iter; i ++ ) {
    //rnn_merge_int_dn( a, amap, b, bmap, D, I, na, nb);
    rnn_mergesort( 2048, 2048, a, amap, D, I );
  }
  //rnn_merge_ref( a, amap, b, bmap, D_ref, I_ref, na, nb);
  avx_time = omp_get_wtime() - beg;

  //rnn_merge_int_dn( a, amap, b, bmap, D, I, na, nb);
  beg = omp_get_wtime();
  for ( i = 0; i < iter; i ++ ) {
    //rnn_merge_ref( a, amap, b, bmap, D_ref, I_ref, na, nb);
    heap_sort( 2048, 2048, b, bmap, D_ref, I_ref ); 
  }
  //rnn_merge_int_dn_var2( a, amap, b, bmap, D, I, na, nb);
  ref_time = omp_get_wtime() - beg;


  //for ( j = 0; j < 64; j ++ ) {
  //  printf( "%5.2lf, %5.2lf\n", a[ j ], amap[ j ] );
  //}

  //for ( j = 0; j < na + nb; j ++ ) {
  //  if ( D[ j ] != D_ref[ j ] ) {
  //    printf( "error D[ %d ]\n", j);
  //    break;
  //  }
  //  if ( I[ j ] != I_ref[ j ] ) {
  //    printf( "error I[ %d ]\n", j);
  //    break;
  //  }
  //}

  //printf( "sorted d:\n" );
  //for ( j = 0; j < 16; j ++ ) {
  //  printf( "%5.2lf ", d[ j ] );
  //}
  //printf( "\n" );

  //printf( "sorted i:\n" );
  //for ( j = 0; j < 16; j ++ ) {
  //  printf( "%5.2lf ", index[ j ] );
  //}
  //printf( "\n" );


  printf( "avx_time: %E, ref_time: %E\n", avx_time, ref_time );

  return 0;
}

