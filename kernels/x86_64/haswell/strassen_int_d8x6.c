#include <stdio.h>
#include <immintrin.h> // AVX
//#include <gsknn.h>
#include <gsknn_internal.h>
#include <avx_types.h>

void strassen_int_d8x6(
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
  int    i;
  // 16 registers.
  v4df_t c03_0, c03_1, c03_2, c03_3, c03_4, c03_5;
  v4df_t c47_0, c47_1, c47_2, c47_3, c47_4, c47_5;
  v4df_t a03, a47, b0, b1;

  #include <strassen_int_d8x6.h>


  // Accumulate c0
    b0.v    = _mm256_broadcast_sd( &beta0 );
    b1.v    = _mm256_broadcast_sd( &alpha0 );
    
    a03.v   = _mm256_load_pd( (double*)( c0      ) );
    if ( !aux->pc )
    a03.v   = _mm256_mul_pd( a03.v, b0.v );
    a03.v   = _mm256_fmadd_pd( c03_0.v, b1.v, a03.v );
    _mm256_store_pd( c0     , a03.v );

    a47.v   = _mm256_load_pd( (double*)( c0 +  4 ) );
    if ( !aux->pc )
    a47.v   = _mm256_mul_pd( a47.v, b0.v );
    a47.v   = _mm256_fmadd_pd( c47_0.v, b1.v, a47.v );
    _mm256_store_pd( c0 +  4, a47.v );

    a03.v   = _mm256_load_pd( (double*)( c0 + 1 * ldc ) );
    if ( !aux->pc )
    a03.v   = _mm256_mul_pd( a03.v, b0.v );
    a03.v   = _mm256_fmadd_pd( c03_1.v, b1.v, a03.v );
    _mm256_store_pd( c0 + 1 * ldc, a03.v );

    a47.v   = _mm256_load_pd( (double*)( c0 + 1 * ldc + 4 ) );
    if ( !aux->pc )
    a47.v   = _mm256_mul_pd( a47.v, b0.v );
    a47.v   = _mm256_fmadd_pd( c47_1.v, b1.v, a47.v );
    _mm256_store_pd( c0 + 1 * ldc + 4, a47.v );

    a03.v   = _mm256_load_pd( (double*)( c0 + 2 * ldc ) );
    if ( !aux->pc )
    a03.v   = _mm256_mul_pd( a03.v, b0.v );
    a03.v   = _mm256_fmadd_pd( c03_2.v, b1.v, a03.v );
    _mm256_store_pd( c0 + 2 * ldc, a03.v );

    a47.v   = _mm256_load_pd( (double*)( c0 + 2 * ldc + 4 ) );
    if ( !aux->pc )
    a47.v   = _mm256_mul_pd( a47.v, b0.v );
    a47.v   = _mm256_fmadd_pd( c47_2.v, b1.v, a47.v );
    _mm256_store_pd( c0 + 2 * ldc + 4, a47.v );

    a03.v   = _mm256_load_pd( (double*)( c0 + 3 * ldc ) );
    if ( !aux->pc )
    a03.v   = _mm256_mul_pd( a03.v, b0.v );
    a03.v   = _mm256_fmadd_pd( c03_3.v, b1.v, a03.v );
    _mm256_store_pd( c0 + 3 * ldc, a03.v );

    a47.v   = _mm256_load_pd( (double*)( c0 + 3 * ldc + 4 ) );
    if ( !aux->pc )
    a47.v   = _mm256_mul_pd( a47.v, b0.v );
    a47.v   = _mm256_fmadd_pd( c47_3.v, b1.v, a47.v );
    _mm256_store_pd( c0 + 3 * ldc + 4, a47.v );

    a03.v   = _mm256_load_pd( (double*)( c0 + 4 * ldc ) );
    if ( !aux->pc )
    a03.v   = _mm256_mul_pd( a03.v, b0.v );
    a03.v   = _mm256_fmadd_pd( c03_4.v, b1.v, a03.v );
    _mm256_store_pd( c0 + 4 * ldc, a03.v );

    a47.v   = _mm256_load_pd( (double*)( c0 + 4 * ldc + 4 ) );
    if ( !aux->pc )
    a47.v   = _mm256_mul_pd( a47.v, b0.v );
    a47.v   = _mm256_fmadd_pd( c47_4.v, b1.v, a47.v );
    _mm256_store_pd( c0 + 4 * ldc + 4, a47.v );

    a03.v   = _mm256_load_pd( (double*)( c0 + 5 * ldc ) );
    if ( !aux->pc )
    a03.v   = _mm256_mul_pd( a03.v, b0.v );
    a03.v   = _mm256_fmadd_pd( c03_5.v, b1.v, a03.v );
    _mm256_store_pd( c0 + 5 * ldc, a03.v );

    a47.v   = _mm256_load_pd( (double*)( c0 + 5 * ldc + 4 ) );
    if ( !aux->pc )
    a47.v   = _mm256_mul_pd( a47.v, b0.v );
    a47.v   = _mm256_fmadd_pd( c47_5.v, b1.v, a47.v );
    _mm256_store_pd( c0 + 5 * ldc + 4, a47.v );


  // Accumulate c1
  if ( c1 ) {
  //if ( beta0 != 0.0 ) {
    b0.v    = _mm256_broadcast_sd( &beta1 );
    b1.v    = _mm256_broadcast_sd( &alpha1 );
    
    a03.v   = _mm256_load_pd( (double*)( c1      ) );
    if ( !aux->pc )
    a03.v   = _mm256_mul_pd( a03.v, b0.v );
    a03.v   = _mm256_fmadd_pd( c03_0.v, b1.v, a03.v );
    _mm256_store_pd( c1     , a03.v );

    a47.v   = _mm256_load_pd( (double*)( c1 +  4 ) );
    if ( !aux->pc )
    a47.v   = _mm256_mul_pd( a47.v, b0.v );
    a47.v   = _mm256_fmadd_pd( c47_0.v, b1.v, a47.v );
    _mm256_store_pd( c1 +  4, a47.v );

    a03.v   = _mm256_load_pd( (double*)( c1 + 1 * ldc ) );
    if ( !aux->pc )
    a03.v   = _mm256_mul_pd( a03.v, b0.v );
    a03.v   = _mm256_fmadd_pd( c03_1.v, b1.v, a03.v );
    _mm256_store_pd( c1 + 1 * ldc, a03.v );

    a47.v   = _mm256_load_pd( (double*)( c1 + 1 * ldc + 4 ) );
    if ( !aux->pc )
    a47.v   = _mm256_mul_pd( a47.v, b0.v );
    a47.v   = _mm256_fmadd_pd( c47_1.v, b1.v, a47.v );
    _mm256_store_pd( c1 + 1 * ldc + 4, a47.v );

    a03.v   = _mm256_load_pd( (double*)( c1 + 2 * ldc ) );
    if ( !aux->pc )
    a03.v   = _mm256_mul_pd( a03.v, b0.v );
    a03.v   = _mm256_fmadd_pd( c03_2.v, b1.v, a03.v );
    _mm256_store_pd( c1 + 2 * ldc, a03.v );

    a47.v   = _mm256_load_pd( (double*)( c1 + 2 * ldc + 4 ) );
    if ( !aux->pc )
    a47.v   = _mm256_mul_pd( a47.v, b0.v );
    a47.v   = _mm256_fmadd_pd( c47_2.v, b1.v, a47.v );
    _mm256_store_pd( c1 + 2 * ldc + 4, a47.v );

    a03.v   = _mm256_load_pd( (double*)( c1 + 3 * ldc ) );
    if ( !aux->pc )
    a03.v   = _mm256_mul_pd( a03.v, b0.v );
    a03.v   = _mm256_fmadd_pd( c03_3.v, b1.v, a03.v );
    _mm256_store_pd( c1 + 3 * ldc, a03.v );

    a47.v   = _mm256_load_pd( (double*)( c1 + 3 * ldc + 4 ) );
    if ( !aux->pc )
    a47.v   = _mm256_mul_pd( a47.v, b0.v );
    a47.v   = _mm256_fmadd_pd( c47_3.v, b1.v, a47.v );
    _mm256_store_pd( c1 + 3 * ldc + 4, a47.v );

    a03.v   = _mm256_load_pd( (double*)( c1 + 4 * ldc ) );
    if ( !aux->pc )
    a03.v   = _mm256_mul_pd( a03.v, b0.v );
    a03.v   = _mm256_fmadd_pd( c03_4.v, b1.v, a03.v );
    _mm256_store_pd( c1 + 4 * ldc, a03.v );

    a47.v   = _mm256_load_pd( (double*)( c1 + 4 * ldc + 4 ) );
    if ( !aux->pc )
    a47.v   = _mm256_mul_pd( a47.v, b0.v );
    a47.v   = _mm256_fmadd_pd( c47_4.v, b1.v, a47.v );
    _mm256_store_pd( c1 + 4 * ldc + 4, a47.v );

    a03.v   = _mm256_load_pd( (double*)( c1 + 5 * ldc ) );
    if ( !aux->pc )
    a03.v   = _mm256_mul_pd( a03.v, b0.v );
    a03.v   = _mm256_fmadd_pd( c03_5.v, b1.v, a03.v );
    _mm256_store_pd( c1 + 5 * ldc, a03.v );

    a47.v   = _mm256_load_pd( (double*)( c1 + 5 * ldc + 4 ) );
    if ( !aux->pc )
    a47.v   = _mm256_mul_pd( a47.v, b0.v );
    a47.v   = _mm256_fmadd_pd( c47_5.v, b1.v, a47.v );
    _mm256_store_pd( c1 + 5 * ldc + 4, a47.v );
  }
}
