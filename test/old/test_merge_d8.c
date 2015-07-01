#include <stdlib.h>
#include <stdio.h>
#include <rnn.h>


void rnn_merge_int_d8( double*, double* );

int main() 
{
  //unsigned long long i[ 16 ] = { 7, 2, 14, 1, 9, 8, 12, 10, 3, 11, 4, 0, 15, 5, 13, 6 };
  //double i[ 8 ] = { 1.0, 3.0, 5.0, 7.0, 0.0, 2.0, 4.0, 6.0 };
  //double d[ 8 ] = { 1.0, 3.0, 5.0, 7.0, 0.0, 2.0, 4.0, 6.0 };


  double i[ 8 ] = { 1.0, 2.0, 5.0, 7.0,  0.0, 3.0, 4.0, 6.0 };
  double d[ 8 ] = { 1.0, 2.0, 5.0, 7.0,  0.0, 3.0, 4.0, 6.0 };

  int    j;


  rnn_merge_int_d8( i, d );

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

  return 0;
}

