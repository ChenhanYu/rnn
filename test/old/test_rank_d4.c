#include <stdlib.h>
#include <stdio.h>
#include <rnn.h>


void rnn_rank_int_d4( double* );

int main() 
{
  double c[ 4 ] = { 2.0, 1.0, 4.0, 3.0 };
 
  rnn_rank_int_d4( c );

  return 0;
}

