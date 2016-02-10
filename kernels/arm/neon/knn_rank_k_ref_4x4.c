#include <math.h>
#include <gsknn.h>



void knn_rank_k_ref_s4x4(
    int    k,
    float  *a,
    float  *b,
    float  *c,
    aux_t  *aux
    )
{
  int    i, j, p, ldr;

  if ( !aux->pc ) {
	for ( i = 0; i < 4; i ++ ) {
	  for ( j = 0; j < 4; j ++ ) {
		c[ i * 4 + j ] = 0.0;
	  }
	}
  }

  for ( p = 0; p < k; p ++ ) {
	for ( i = 0; i < 4; i ++ ) {
	  for ( j = 0; j < 4; j ++ ) {
		c[ i * 4 + j ] += a[ i ] * b[ j ];
	  }
	}
	a += 4;
	b += 4;
  }
}



void knn_rank_k_abs_ref_s4x4(
    int    k,
    float  *a,
    float  *b,
    float  *c,
    aux_t  *aux
    )
{
  int    i, j, p, ldr;

  if ( !aux->pc ) {
	for ( i = 0; i < 4; i ++ ) {
	  for ( j = 0; j < 4; j ++ ) {
		c[ i * 4 + j ] = 0.0;
	  }
	}
  }

  for ( p = 0; p < k; p ++ ) {
	for ( i = 0; i < 4; i ++ ) {
	  for ( j = 0; j < 4; j ++ ) {
		c[ i * 4 + j ] += fabs( a[ i ] - b[ j ] );
	  }
	}
	a += 4;
	b += 4;
  }
}



void knn_rank_k_ref_d4x4(
    int    k,
    double *a,
    double *b,
    double *c,
    aux_t  *aux
    )
{
  int    i, j, p, ldr;

  if ( !aux->pc ) {
	for ( i = 0; i < 4; i ++ ) {
	  for ( j = 0; j < 4; j ++ ) {
		c[ i * 4 + j ] = 0.0;
	  }
	}
  }

  for ( p = 0; p < k; p ++ ) {
	for ( i = 0; i < 4; i ++ ) {
	  for ( j = 0; j < 4; j ++ ) {
		c[ i * 4 + j ] += a[ i ] * b[ j ];
	  }
	}
	a += 4;
	b += 4;
  }
}


void knn_rank_k_abs_ref_d4x4(
    int    k,
    double *a,
    double *b,
    double *c,
    aux_t  *aux
    )
{
  int    i, j, p, ldr;

  if ( !aux->pc ) {
	for ( i = 0; i < 4; i ++ ) {
	  for ( j = 0; j < 4; j ++ ) {
		c[ i * 4 + j ] = 0.0;
	  }
	}
  }

  for ( p = 0; p < k; p ++ ) {
	for ( i = 0; i < 4; i ++ ) {
	  for ( j = 0; j < 4; j ++ ) {
		c[ i * 4 + j ] += fabs( a[ i ] * b[ j ] );
	  }
	}
	a += 4;
	b += 4;
  }
}
