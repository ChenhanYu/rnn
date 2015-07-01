#include <stdlib.h>
#include <stdio.h>
#include <rnn.h>


void rnn_sort_int_d4x4( double*, double* );

int main() 
{
  //unsigned long long i[ 16 ] = { 7, 2, 14, 1, 9, 8, 12, 10, 3, 11, 4, 0, 15, 5, 13, 6 };
  double i[ 16 ] = { 7.0, 2.0, 14.0, 1.0, 9.0, 8.0, 12.0, 10.0, 3.0, 11.0, 4.0, 0.0, 15.0, 5.0, 13.0, 6.0  };
  double d[ 16 ] = { 7.0, 2.0, 14.0, 1.0, 9.0, 8.0, 12.0, 10.0, 3.0, 11.0, 4.0, 0.0, 15.0, 5.0, 13.0, 6.0  };
  int    j;


  printf( "input d:\n" );
  for ( j = 0; j < 16; j ++ ) {
    printf( "%5.2lf ", d[ j ] );
  }
  printf( "\n" );

  printf( "input i:\n" );
  for ( j = 0; j < 16; j ++ ) {
    printf( "%5.2lf ", i[ j ] );
  }
  printf( "\n" );

  rnn_sort_int_d4x4( i, d );

  printf( "sorted d:\n" );
  for ( j = 0; j < 16; j ++ ) {
    printf( "%5.2lf ", d[ j ] );
  }
  printf( "\n" );

  printf( "sorted i:\n" );
  for ( j = 0; j < 16; j ++ ) {
    printf( "%5.2lf ", i[ j ] );
  }
  printf( "\n" );

  return 0;
}

