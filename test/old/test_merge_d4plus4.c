#include <stdlib.h>
#include <stdio.h>
#include <rnn.h>

void rnn_merge_int_d4plus4(
    double *a,
    double *amap,
    double *b,
    double *bmap,
    double *d,
    double *i
    );

int main() 
{
  //unsigned long long i[ 16 ] = { 7, 2, 14, 1, 9, 8, 12, 10, 3, 11, 4, 0, 15, 5, 13, 6 };
  //double i[ 8 ] = { 1.0, 3.0, 5.0, 7.0, 0.0, 2.0, 4.0, 6.0 };
  //double d[ 8 ] = { 1.0, 3.0, 5.0, 7.0, 0.0, 2.0, 4.0, 6.0 };

  double a[ 4 ]    = { 1.0, 2.0, 5.0, 7.0 };
  double b[ 4 ]    = { 0.0, 3.0, 4.0, 6.0 };
  double amap[ 4 ] = { 1.0, 2.0, 5.0, 7.0 };
  double bmap[ 4 ]    = { 0.0, 3.0, 4.0, 6.0 };

  double i[ 8 ] = { 0.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.0, 0.0 };
  double d[ 8 ] = { 0.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.0, 0.0 };

  int    j;

  //rnn_merge_int_d4plus4( a, amap, b, bmap, i, d );
  rnn_merge_int_d4plus4( a, amap, b, bmap, d, i);

  printf( "sorted d:\n" );
  for ( j = 0; j < 8; j ++ ) {
    printf( "%5.2lf ", d[ j ] );
  }
  printf( "\n" );

  printf( "sorted i:\n" );
  for ( j = 0; j < 8; j ++ ) {
    printf( "%5.2lf ", i[ j ] );
  }
  printf( "\n" );


  /*
     printf( "sorted a:\n" );
     for ( j = 0; j < 4; j ++ ) {
     printf( "%5.2lf ", a[ j ] );
     }
     printf( "\n" );

  printf( "sorted b:\n" );
  for ( j = 0; j < 4; j ++ ) {
    printf( "%5.2lf ", b[ j ] );
  }
  printf( "\n" );

  printf( "sorted amap:\n" );
  for ( j = 0; j < 4; j ++ ) {
    printf( "%5.2lf ", amap[ j ] );
  }
  printf( "\n" );

  printf( "sorted bmap:\n" );
  for ( j = 0; j < 4; j ++ ) {
    printf( "%5.2lf ", bmap[ j ] );
  }
  printf( "\n" );
  */

  return 0;
}

