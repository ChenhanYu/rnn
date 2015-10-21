#include <stdlib.h>
#include <stdio.h>
#include <mex.h>

void mexFunction( 
    int nlhs, 
    mxArray *plhs[],
    int nrhs, 
    const mxArray *prhs[]
    )
{
  double alpha = mxGetScalar( prhs[ 0 ] );
  double *A = mxGetPr( prhs[ 1 ] );
  size_t N = mxGetN( prhs[ 1 ] ); 
  printf( "alpha %lf, N %d\n", alpha, N );
}
