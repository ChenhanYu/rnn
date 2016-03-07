#include <stdio.h>
//#include <gsknn.h>
#include <gsknn_internal.h>
#include <gsknn_config.h>

void knn_ref_row_s(
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
  int    i, j, p;
  int    ldr = aux->ldr;
  int    *I = aux->I;
  float  *D = aux->D_s;
  float  cr[ SKNN_MR * SKNN_NR ];

  if ( aux->pc ) {
    for ( j = 0; j < SKNN_NR; j ++ ) {
      for ( i = 0; i < SKNN_MR; i ++ ) {
        cr[ i * SKNN_NR + j ] = c[ j * SKNN_MR + i ];
      }
    }
  }
  else {
    for ( i = 0; i < SKNN_MR; i ++ ) {
      for ( j = 0; j < SKNN_NR; j ++ ) {
        cr[ i * SKNN_NR + j ] = 0.0;
      }
    }
  }

  for ( p = 0; p < k; p ++ ) {
    for ( i = 0; i < SKNN_MR; i ++ ) {
      for ( j = 0; j < SKNN_NR; j ++ ) {
        cr[ i * SKNN_NR + j ] += a[ i ] * b[ j ];
      }
    }
    a += SKNN_MR;
    b += SKNN_NR;
  }

  for ( i = 0; i < SKNN_MR; i ++ ) {
    for ( j = 0; j < SKNN_NR; j ++ ) {
      cr[ i * SKNN_NR + j ] *= -2.0;
      cr[ i * SKNN_NR + j ] += aa[ i ];
      cr[ i * SKNN_NR + j ] += bb[ j ];
    }
  }

  //for ( j = 0; j < SKNN_NR; j ++ ) {
  //  printf( "%d, ", bmap[ j ] );
  //}
  //printf( "\n\n" );


  /* 
  printf( "\n" );
  for ( i = 0; i < SKNN_MR; i ++ ) {
    for ( j = 0; j < SKNN_NR; j ++ ) {
      printf( "%E, ", cr[ i * SKNN_NR + j ] );
    }
    printf( "\n" );
  }
  printf( "\n" );

  for ( j = 0; j < SKNN_NR; j ++ ) {
    printf( "%E, ", bb[ j ] );
  }
  printf( "\n" );

  for ( i = 0; i < SKNN_MR; i ++ ) {
    printf( "%E, ", aa[ i ] );
  }
  printf( "\n\n" );
  */  

  //printf( "%d, %d, %d\n", aux->m, aux->n, ldr );

  for ( i = 0; i < aux->m; i ++ ) {
    heapSelect_s( aux->n, r, cr + i * SKNN_NR, bmap, D + i * ldr, I + i * ldr ); 
  }

}


void knn_ref_row_d(
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
  int    i, j, p;
  int    ldr = aux->ldr;
  int    *I = aux->I;
  double *D = aux->D;
  double cr[ DKNN_MR * DKNN_NR ];

  if ( aux->pc ) {
    for ( j = 0; j < DKNN_NR; j ++ ) {
      for ( i = 0; i < DKNN_MR; i ++ ) {
        cr[ i * DKNN_NR + j ] = c[ j * DKNN_MR + i ];
      }
    }
  }
  else {
    for ( i = 0; i < DKNN_MR; i ++ ) {
      for ( j = 0; j < DKNN_NR; j ++ ) {
        cr[ i * DKNN_NR + j ] = 0.0;
      }
    }
  }

  for ( p = 0; p < k; p ++ ) {
    for ( i = 0; i < DKNN_MR; i ++ ) {
      for ( j = 0; j < DKNN_NR; j ++ ) {
        cr[ i * DKNN_NR + j ] += a[ i ] * b[ j ];
      }
    }
    a += DKNN_MR;
    b += DKNN_NR;
  }

  for ( i = 0; i < DKNN_MR; i ++ ) {
    for ( j = 0; j < DKNN_NR; j ++ ) {
      cr[ i * DKNN_NR + j ] *= -2.0;
      cr[ i * DKNN_NR + j ] += aa[ i ];
      cr[ i * DKNN_NR + j ] += bb[ j ];
    }
  }

  for ( i = 0; i < aux->m; i ++ ) {
    heapSelect_d( aux->n, r, cr + i * DKNN_NR, bmap, D + i * ldr, I + i * ldr ); 
  }
}
