#include <gsknn.h>
#include <gsknn_config.h>

void knn_r_ref_s8x8_row(
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
  float  *D = aux->D;
  float  cr[ SKNN_MR * SKNN_NR ];

  for ( i = 0; i <SKNN_MR; i ++ ) {
    for ( j = 0; j < SKNN_NR; j ++ ) {
      cr[ i * SKNN_NR + j ] = 0.0;
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

  // Accumulate rank-k update.
  if ( aux->pc ) {
    for ( j = 0; j < SKNN_NR; j ++ ) {
      for ( i = 0; i < SKNN_MR; i ++ ) {
        cr[ i * SKNN_NR + j ] += c[ j * SKNN_MR + i ];
      }
    }
  }

  for ( i = 0; i < SKNN_MR; i ++ ) {
    for ( j = 0; j < SKNN_NR; j ++ ) {
      cr[ i * SKNN_NR + j ] *= -2.0;
      cr[ i * SKNN_NR + j ] += aa[ i ];
      cr[ i * SKNN_NR + j ] += bb[ j ];
    }
  }

  for ( i = 0; i < aux->m; i ++ ) {
    heapSelect_s( aux->n, r, c + i * SKNN_NR, bmap, D + i * ldr, I + i * ldr ); 
  }

}
