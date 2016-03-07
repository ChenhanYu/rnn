#include <math.h>
//#include <gsknn.h>
#include <gsknn_internal.h>
#include <gsknn_config.h>

void rank_k_ref_s(
    int    k,
    float  *a,
    float  *b,
    float  *c,
    int    ldc,
    aux_t  *aux
    )
{
  int    i, j, p;
  float  cr[ SKNN_MR * SKNN_NR ];

  for ( j = 0; j < SKNN_NR; j ++ ) {
    for ( i = 0; i < SKNN_MR; i ++ ) {
      cr[ j * SKNN_MR + i ] = 0.0;
    }
  }

  for ( p = 0; p < k; p ++ ) {
    for ( j = 0; j < SKNN_NR; j ++ ) {
      for ( i = 0; i < SKNN_MR; i ++ ) {
        cr[ j * SKNN_MR + i ] += a[ i ] * b[ j ];
      }
    }
    a += SKNN_MR;
    b += SKNN_NR;
  }

  // Accumulate rank-k update.
  if ( aux->pc ) {
    for ( j = 0; j < SKNN_NR; j ++ ) {
      for ( i = 0; i < SKNN_MR; i ++ ) {
        c[ j * SKNN_MR + i ] += cr[ j * SKNN_MR + i ];
      }
    }
  }
  else {
    for ( j = 0; j < SKNN_NR; j ++ ) {
      for ( i = 0; i < SKNN_MR; i ++ ) {
        c[ j * SKNN_MR + i ] = cr[ j * SKNN_MR + i ];
      }
    }
  }
}

void rank_k_abs_ref_s(
    int    k,
    float  *a,
    float  *b,
    float  *c,
    int    ldc,
    aux_t  *aux
    )
{
  int    i, j, p;
  float  cr[ SKNN_MR * SKNN_NR ];

  for ( j = 0; j < SKNN_NR; j ++ ) {
    for ( i = 0; i <SKNN_MR; i ++ ) {
      cr[ j * SKNN_MR + i ] = 0.0;
    }
  }

  for ( p = 0; p < k; p ++ ) {
    for ( j = 0; j < SKNN_NR; j ++ ) {
      for ( i = 0; i < SKNN_MR; i ++ ) {
        cr[ j * SKNN_MR + i ] += fabs( a[ i ] - b[ j ] );
      }
    }
    a += SKNN_MR;
    b += SKNN_NR;
  }

  // Accumulate rank-k update.
  if ( aux->pc ) {
    for ( j = 0; j < SKNN_NR; j ++ ) {
      for ( i = 0; i < SKNN_MR; i ++ ) {
        c[ j * SKNN_MR + i ] += cr[ j * SKNN_MR + i ];
      }
    }
  }
  else {
    for ( j = 0; j < SKNN_NR; j ++ ) {
      for ( i = 0; i < SKNN_MR; i ++ ) {
        c[ j * SKNN_MR + i ] = cr[ j * SKNN_MR + i ];
      }
    }
  }
}

void rank_k_ref_d(
    int    k,
    double *a,
    double *b,
    double *c,
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

  // Accumulate rank-k update.
  if ( aux->pc ) {
    for ( j = 0; j < DKNN_NR; j ++ ) {
      for ( i = 0; i < DKNN_MR; i ++ ) {
        c[ j * DKNN_MR + i ] += cr[ j * DKNN_MR + i ];
      }
    }
  }
  else {
    for ( j = 0; j < DKNN_NR; j ++ ) {
      for ( i = 0; i < DKNN_MR; i ++ ) {
        c[ j * DKNN_MR + i ] = cr[ j * DKNN_MR + i ];
      }
    }
  }
}

void rank_k_abs_ref_d(
    int    k,
    double *a,
    double *b,
    double *c,
    int    ldc,
    aux_t  *aux
    )
{
  int    i, j, p;
  double cr[ SKNN_MR * SKNN_NR ];

  for ( j = 0; j < DKNN_NR; j ++ ) {
    for ( i = 0; i <DKNN_MR; i ++ ) {
      cr[ j * DKNN_MR + i ] = 0.0;
    }
  }

  for ( p = 0; p < k; p ++ ) {
    for ( j = 0; j < DKNN_NR; j ++ ) {
      for ( i = 0; i < DKNN_MR; i ++ ) {
        cr[ j * DKNN_MR + i ] += fabs( a[ i ] - b[ j ] );
      }
    }
    a += DKNN_MR;
    b += DKNN_NR;
  }

  // Accumulate rank-k update.
  if ( aux->pc ) {
    for ( j = 0; j < DKNN_NR; j ++ ) {
      for ( i = 0; i < DKNN_MR; i ++ ) {
        c[ j * DKNN_MR + i ] += cr[ j * DKNN_MR + i ];
      }
    }
  }
  else {
    for ( j = 0; j < DKNN_NR; j ++ ) {
      for ( i = 0; i < DKNN_MR; i ++ ) {
        c[ j * DKNN_MR + i ] = cr[ j * DKNN_MR + i ];
      }
    }
  }
}
