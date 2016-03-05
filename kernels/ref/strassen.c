#include <math.h>
#include <gsknn.h>
#include <gsknn_config.h>

void strassen_ref_d(
    int    k,              
    double alpha0,         
    double alpha1,         
    double *a,             
    double *b,             
    double beta0,          
    double *c0,            
    double beta1,          
	double *c1,            
    int    ldc,            
    aux_t  *aux            
	)
{
  int    i, j, p;
  double cr[ DKNN_MR * DKNN_NR ];

  for ( j = 0; j < DKNN_NR; j ++ ) {
    for ( i = 0; i < DKNN_MR; i ++ ) {
      cr[ j * DKNN_MR + i ] = 0.0;
    }
  }

  for ( p = 0; p < k; p ++ ) {
    for ( j = 0; j < DKNN_NR; j ++ ) {
      for ( i = 0; i < DKNN_MR; i ++ ) {
        cr[ j * DKNN_MR + i ] += a[ i ] * b[ j ];
      }
    }
    a += DKNN_MR;
    b += DKNN_NR;
  }

  // c0 = beta0 * c0 + alpha0 * cr
  for ( j = 0; j < aux->n; j ++ ) {
	for ( i = 0; i < aux->m; i ++ ) {
	  c0[ j * ldc + i ] = beta0 * c0[ j * ldc + i ] + alpha0 + cr[ j * DKNN_MR + i ];
	}
  }

  // c1 = beta1 * c1 + alpha1 * cr
  if ( c1 ) {
	for ( j = 0; j < aux->n; j ++ ) {
	  for ( i = 0; i < aux->m; i ++ ) {
		c1[ j * ldc + i ] = beta1 * c1[ j * ldc + i ] + alpha1 + cr[ j * DKNN_MR + i ];
	  }
	}
  }
}
