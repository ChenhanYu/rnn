#include <stdlib.h>
#include <stdio.h>
#include <rnn.h>


void rnn_bitonic_int_d8( int*, double* );

int main() 
{
  int    i[ 4 ] = { 0, 1, 2, 3 };
  double d[ 4 ] = { 2.0, 1.0, 4.0, 3.0 };
 
  rnn_bitonic_int_d8( i, d );

  return 0;
}

