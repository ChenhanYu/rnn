#include <math.h>
#include <gsknn.h>



void knn_r_ref_s4x4_row(
    int    k,
    int    r,
    float  *aa,
    float  *a,
    float  *bb,
    float  *b,
    float  *c,
    aux_t  *aux,
    int    *bmap
    )
{
  int    i, j, p, ldr;
  int    *I = aux->I;
  float  *D = aux->D_s;

  ldr = r;

  if ( !aux->pc ) {
	for ( i = 0; i < 4; i ++ ) {
	  for ( j = 0; j < 4; j ++ ) {
		c[ i * 4 + j ] = 0.0;
	  }
	}
  }

  //printf( "a: %f %f %f %f\n ", a[ 0 ], a[ 1 ], a[ 2 ], a[ 3 ] );
  //printf( "b: %f %f %f %f\n ", b[ 0 ], b[ 1 ], b[ 2 ], b[ 3 ] );

  for ( p = 0; p < k; p ++ ) {
	for ( i = 0; i < 4; i ++ ) {
	  for ( j = 0; j < 4; j ++ ) {
		c[ i * 4 + j ] += a[ i ] * b[ j ];
	  }
	}
	a += 4;
	b += 4;
  }

  //printf( "c: %f %f %f %f\n ", c[ 0 ], c[ 1 ], c[ 2 ], c[ 3 ] );
  //printf( "c: %f %f %f %f\n ", c[ 4 ], c[ 5 ], c[ 6 ], c[ 7 ] );
  //printf( "c: %f %f %f %f\n ", c[ 8 ], c[ 9 ], c[ 10 ], c[ 11 ] );
  //printf( "c: %f %f %f %f\n ", c[ 12 ], c[ 13 ], c[ 14 ], c[ 15 ] );

  for ( i = 0; i < 4; i ++ ) {
	for ( j = 0; j < 4; j ++ ) {
	  c[ i * 4 + j ] *= -2.0;
	  c[ i * 4 + j ] += aa[ i ] + bb[ j ];
	}
	heapSelect_s( aux->n, r, c + i * 4, bmap, D + i * ldr, I + i * ldr );
  }
}



void knn_r_abs_ref_s4x4_row(
    int    k,
    int    r,
    float  *aa,
    float  *a,
    float  *bb,
    float  *b,
    float  *c,
    aux_t  *aux,
    int    *bmap
    )
{
  int    i, j, p, ldr;
  int    *I = aux->I;
  float  *D = aux->D_s;

  ldr = r;

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

  for ( i = 0; i < 4; i ++ ) {
	heapSelect_s( aux->n, r, c + i * 4, bmap, D + i * ldr, I + i * ldr );
  }
}



void knn_r_ref_d4x4_row(
    int    k,
    int    r,
    double *aa,
    double *a,
    double *bb,
    double *b,
    double *c,
    aux_t  *aux,
    int    *bmap
    )
{
  int    i, j, p, ldr;
  int    *I = aux->I;
  double *D = aux->D;

  ldr = r;

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

  for ( i = 0; i < 4; i ++ ) {
	for ( j = 0; j < 4; j ++ ) {
	  c[ i * 4 + j ] *= -2.0;
	  c[ i * 4 + j ] += aa[ i ] + bb[ j ];
	}
	heapSelect_d( aux->n, r, c + i * 4, bmap, D + i * ldr, I + i * ldr );
  }
}


void knn_r_abs_ref_d4x4_row(
    int    k,
    int    r,
    double *aa,
    double *a,
    double *bb,
    double *b,
    double *c,
    aux_t  *aux,
    int    *bmap
    )
{
  int    i, j, p, ldr;
  int    *I = aux->I;
  double *D = aux->D;

  ldr = r;

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

  for ( i = 0; i < 4; i ++ ) {
	heapSelect_d( aux->n, r, c + i * 4, bmap, D + i * ldr, I + i * ldr );
  }
}


