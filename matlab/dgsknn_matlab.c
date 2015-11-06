#include <stdlib.h>
#include <stdio.h>
#include <mex.h>

#include <gsknn.h>


void mexFunction( 
    int nlhs, 
    mxArray *plhs[],
    int nrhs, 
    const mxArray *prhs[]
    )
{
  int    i, j, m, n, d, k;

  m = (int)mxGetScalar( prhs[ 0 ] );
  n = (int)mxGetScalar( prhs[ 1 ] );
  d = (int)mxGetScalar( prhs[ 2 ] );
  d = (int)mxGetScalar( prhs[ 3 ] );
  double *q  = mxGetPr( prhs[ 4 ] );
  double *s  = mxGetPr( prhs[ 5 ] );
  double *X  = mxGetPr( prhs[ 6 ] );
  double *X2 = mxGetPr( prhs[ 7 ] );
  plhs[ 0 ] = mxCreateDoubleMatrix( m, k, mxREAL );
  plhs[ 1 ] = mxCreateDoubleMatrix( m, k, mxREAL );
  double *I = mxGetPr( plhs[ 0 ] );
  double *D = mxGetPr( plhs[ 1 ] );

  int    *qint = (int*)malloc( sizeof(int) * m );
  int    *sint = (int*)malloc( sizeof(int) * n );

  for ( i = 0; i < m; i ++ ) qint[ i ] = (int)q[ i ];
  for ( j = 0; j < n; j ++ ) sint[ j ] = (int)s[ j ];

  printf( "%d, %d, %d\n", m, n, d );

  heap_t *heap = rnn_heapCreate( m, k, 1.79E+308 );

  /*
  {
    dgsrnn(
        m,
        n,
        d,
        k,
        X,
        X2,
        qint,
        X,
        X2,
        sint,
        heap
        );
  }
  */



  free( qint );
  free( sint ); 
}
