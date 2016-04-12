#include <stdio.h>
#include <immintrin.h> // AVX
//#include <gsknn.h>
#include <gsknn_internal.h>
#include <avx_types.h>

void knn_int_row_s16x6(
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
  int    i, j, ldr = aux->ldr;
  int    *I = aux->I;
  float  *D = aux->D_s;

  float  neg2 = -2.0;
  float  dzero = 0.0;
  // 16 registers.
  v8sf_t c03_0, c03_1, c03_2, c03_3, c03_4, c03_5;
  v8sf_t c47_0, c47_1, c47_2, c47_3, c47_4, c47_5;
  v8sf_t a03, a47, b0, b1;

  #include <rank_k_int_s16x6.h>

  a03.v = _mm256_broadcast_ss( &neg2 );

  c03_0.v  = _mm256_mul_ps( a03.v, c03_0.v );
  c03_1.v  = _mm256_mul_ps( a03.v, c03_1.v );
  c03_2.v  = _mm256_mul_ps( a03.v, c03_2.v );
  c03_3.v  = _mm256_mul_ps( a03.v, c03_3.v );
  c03_4.v  = _mm256_mul_ps( a03.v, c03_4.v );
  c03_5.v  = _mm256_mul_ps( a03.v, c03_5.v );

  c47_0.v  = _mm256_mul_ps( a03.v, c47_0.v );
  c47_1.v  = _mm256_mul_ps( a03.v, c47_1.v );
  c47_2.v  = _mm256_mul_ps( a03.v, c47_2.v );
  c47_3.v  = _mm256_mul_ps( a03.v, c47_3.v );
  c47_4.v  = _mm256_mul_ps( a03.v, c47_4.v );
  c47_5.v  = _mm256_mul_ps( a03.v, c47_5.v );

  a47.v = _mm256_load_ps( (float*)aa );
  c03_0.v  = _mm256_add_ps( a47.v, c03_0.v );
  c03_1.v  = _mm256_add_ps( a47.v, c03_1.v );
  c03_2.v  = _mm256_add_ps( a47.v, c03_2.v );
  c03_3.v  = _mm256_add_ps( a47.v, c03_3.v );
  c03_4.v  = _mm256_add_ps( a47.v, c03_4.v );
  c03_5.v  = _mm256_add_ps( a47.v, c03_5.v );

  a47.v = _mm256_load_ps( (float*)( aa + 8 ) );
  c47_0.v  = _mm256_add_ps( a47.v, c47_0.v );
  c47_1.v  = _mm256_add_ps( a47.v, c47_1.v );
  c47_2.v  = _mm256_add_ps( a47.v, c47_2.v );
  c47_3.v  = _mm256_add_ps( a47.v, c47_3.v );
  c47_4.v  = _mm256_add_ps( a47.v, c47_4.v );
  c47_5.v  = _mm256_add_ps( a47.v, c47_5.v );

  b0.v     = _mm256_broadcast_ss( (float*)bb );
  c03_0.v  = _mm256_add_ps( b0.v, c03_0.v );
  c47_0.v  = _mm256_add_ps( b0.v, c47_0.v );

  b1.v     = _mm256_broadcast_ss( (float*)( bb + 1 ) );
  c03_1.v  = _mm256_add_ps( b1.v, c03_1.v );
  c47_1.v  = _mm256_add_ps( b1.v, c47_1.v );

  b0.v     = _mm256_broadcast_ss( (float*)( bb + 2 ) );
  c03_2.v  = _mm256_add_ps( b0.v, c03_2.v );
  c47_2.v  = _mm256_add_ps( b0.v, c47_2.v );

  b1.v     = _mm256_broadcast_ss( (float*)( bb + 3 ) );
  c03_3.v  = _mm256_add_ps( b1.v, c03_3.v );
  c47_3.v  = _mm256_add_ps( b1.v, c47_3.v );

  b0.v     = _mm256_broadcast_ss( (float*)( bb + 4 ) );
  c03_4.v  = _mm256_add_ps( b0.v, c03_4.v );
  c47_4.v  = _mm256_add_ps( b0.v, c47_4.v );

  b1.v     = _mm256_broadcast_ss( (float*)( bb + 5 ) );
  c03_5.v  = _mm256_add_ps( b1.v, c03_5.v );
  c47_5.v  = _mm256_add_ps( b1.v, c47_5.v );


  // Store c
  for ( i = 0; i <  8; i ++ ) c[ 6 * i + 0 ] = c03_0.s[ i ];
  for ( i = 8; i < 16; i ++ ) c[ 6 * i + 0 ] = c47_0.s[ i - 8 ];

  for ( i = 0; i <  8; i ++ ) c[ 6 * i + 1 ] = c03_1.s[ i ];
  for ( i = 8; i < 16; i ++ ) c[ 6 * i + 1 ] = c47_1.s[ i - 8 ];

  for ( i = 0; i <  8; i ++ ) c[ 6 * i + 2 ] = c03_2.s[ i ];
  for ( i = 8; i < 16; i ++ ) c[ 6 * i + 2 ] = c47_2.s[ i - 8 ];

  for ( i = 0; i <  8; i ++ ) c[ 6 * i + 3 ] = c03_3.s[ i ];
  for ( i = 8; i < 16; i ++ ) c[ 6 * i + 3 ] = c47_3.s[ i - 8 ];

  for ( i = 0; i <  8; i ++ ) c[ 6 * i + 4 ] = c03_4.s[ i ];
  for ( i = 8; i < 16; i ++ ) c[ 6 * i + 4 ] = c47_4.s[ i - 8 ];

  for ( i = 0; i <  8; i ++ ) c[ 6 * i + 5 ] = c03_5.s[ i ];
  for ( i = 8; i < 16; i ++ ) c[ 6 * i + 5 ] = c47_5.s[ i - 8 ];

  for ( i = 0; i < aux->m; i ++ ) {
    heapSelect_s( aux->n, r, c + i * 6, bmap, D + i * ldr, I + i * ldr ); 
  }
}


void knn_int_row_d8x6(
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
  int    i, j, ldr = aux->ldr;
  int    *I = aux->I;
  double *D = aux->D;

  double neg2 = -2.0;
  double dzero = 0.0;
  // 16 registers.
  v4df_t c03_0, c03_1, c03_2, c03_3, c03_4, c03_5;
  v4df_t c47_0, c47_1, c47_2, c47_3, c47_4, c47_5;
  v4df_t a03, a47, b0, b1;

  #include <rank_k_int_d8x6.h>

  a03.v = _mm256_broadcast_sd( &neg2 );

  c03_0.v  = _mm256_mul_pd( a03.v, c03_0.v );
  c03_1.v  = _mm256_mul_pd( a03.v, c03_1.v );
  c03_2.v  = _mm256_mul_pd( a03.v, c03_2.v );
  c03_3.v  = _mm256_mul_pd( a03.v, c03_3.v );
  c03_4.v  = _mm256_mul_pd( a03.v, c03_4.v );
  c03_5.v  = _mm256_mul_pd( a03.v, c03_5.v );

  c47_0.v  = _mm256_mul_pd( a03.v, c47_0.v );
  c47_1.v  = _mm256_mul_pd( a03.v, c47_1.v );
  c47_2.v  = _mm256_mul_pd( a03.v, c47_2.v );
  c47_3.v  = _mm256_mul_pd( a03.v, c47_3.v );
  c47_4.v  = _mm256_mul_pd( a03.v, c47_4.v );
  c47_5.v  = _mm256_mul_pd( a03.v, c47_5.v );

  a47.v = _mm256_load_pd( (double*)aa );
  c03_0.v  = _mm256_add_pd( a47.v, c03_0.v );
  c03_1.v  = _mm256_add_pd( a47.v, c03_1.v );
  c03_2.v  = _mm256_add_pd( a47.v, c03_2.v );
  c03_3.v  = _mm256_add_pd( a47.v, c03_3.v );
  c03_4.v  = _mm256_add_pd( a47.v, c03_4.v );
  c03_5.v  = _mm256_add_pd( a47.v, c03_5.v );

  a47.v = _mm256_load_pd( (double*)( aa + 4 ) );
  c47_0.v  = _mm256_add_pd( a47.v, c47_0.v );
  c47_1.v  = _mm256_add_pd( a47.v, c47_1.v );
  c47_2.v  = _mm256_add_pd( a47.v, c47_2.v );
  c47_3.v  = _mm256_add_pd( a47.v, c47_3.v );
  c47_4.v  = _mm256_add_pd( a47.v, c47_4.v );
  c47_5.v  = _mm256_add_pd( a47.v, c47_5.v );
  
  b0.v     = _mm256_broadcast_sd( (double*)bb );
  c03_0.v  = _mm256_add_pd( b0.v, c03_0.v );
  c47_0.v  = _mm256_add_pd( b0.v, c47_0.v );

  b1.v     = _mm256_broadcast_sd( (double*)( bb + 1 ) );
  c03_1.v  = _mm256_add_pd( b1.v, c03_1.v );
  c47_1.v  = _mm256_add_pd( b1.v, c47_1.v );

  b0.v     = _mm256_broadcast_sd( (double*)( bb + 2 ) );
  c03_2.v  = _mm256_add_pd( b0.v, c03_2.v );
  c47_2.v  = _mm256_add_pd( b0.v, c47_2.v );

  b1.v     = _mm256_broadcast_sd( (double*)( bb + 3 ) );
  c03_3.v  = _mm256_add_pd( b1.v, c03_3.v );
  c47_3.v  = _mm256_add_pd( b1.v, c47_3.v );

  b0.v     = _mm256_broadcast_sd( (double*)( bb + 4 ) );
  c03_4.v  = _mm256_add_pd( b0.v, c03_4.v );
  c47_4.v  = _mm256_add_pd( b0.v, c47_4.v );

  b1.v     = _mm256_broadcast_sd( (double*)( bb + 5 ) );
  c03_5.v  = _mm256_add_pd( b1.v, c03_5.v );
  c47_5.v  = _mm256_add_pd( b1.v, c47_5.v );


  // Store c
  for ( i = 0; i < 4; i ++ ) c[ 6 * i + 0 ] = c03_0.d[ i ];
  for ( i = 4; i < 8; i ++ ) c[ 6 * i + 0 ] = c47_0.d[ i - 4 ];

  for ( i = 0; i < 4; i ++ ) c[ 6 * i + 1 ] = c03_1.d[ i ];
  for ( i = 4; i < 8; i ++ ) c[ 6 * i + 1 ] = c47_1.d[ i - 4 ];

  for ( i = 0; i < 4; i ++ ) c[ 6 * i + 2 ] = c03_2.d[ i ];
  for ( i = 4; i < 8; i ++ ) c[ 6 * i + 2 ] = c47_2.d[ i - 4 ];

  for ( i = 0; i < 4; i ++ ) c[ 6 * i + 3 ] = c03_3.d[ i ];
  for ( i = 4; i < 8; i ++ ) c[ 6 * i + 3 ] = c47_3.d[ i - 4 ];

  for ( i = 0; i < 4; i ++ ) c[ 6 * i + 4 ] = c03_4.d[ i ];
  for ( i = 4; i < 8; i ++ ) c[ 6 * i + 4 ] = c47_4.d[ i - 4 ];

  for ( i = 0; i < 4; i ++ ) c[ 6 * i + 5 ] = c03_5.d[ i ];
  for ( i = 4; i < 8; i ++ ) c[ 6 * i + 5 ] = c47_5.d[ i - 4 ];


  for ( i = 0; i < aux->m; i ++ ) {
    heapSelect_d( aux->n, r, c + i * 6, bmap, D + i * ldr, I + i * ldr ); 
  }
}
