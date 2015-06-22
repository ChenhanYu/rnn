#include <immintrin.h> // AVX
#include <rnn.h>

typedef union {
  double d;
  unsigned long long u;
} udouble_t;


void rnn_r1_int_d8x4(
    int    k,
    double alpha,
    double *aa,
    double *a,
    double *bb,
    double *b,
    aux_t  *aux,
    int    *amap,
    int    *I,
    double *D
    )
{
  int    i;
  double neg2 = -2.0;
  double dzero = 0.0;
  //C's output
  v4df_t c03_0, c03_1, c03_2, c03_3;
  v4df_t c47_0, c47_1, c47_2, c47_3;
  v4df_t tmpc03_0, tmpc03_1, tmpc03_2, tmpc03_3;
  v4df_t tmpc47_0, tmpc47_1, tmpc47_2, tmpc47_3;
  v4df_t c_tmp;

  v4df_t a03, a47;
  v4df_t A03, A47; // prefetched A 

  v4df_t b0, b1, b2, b3;
  v4df_t B0; // prefetched B



  v4df_t tmp;


  v4df_t aa_tmp, bb_tmp;

  v4li_t index, temp_cmp32, temp_brd32;
  v4df_t data, temp_cmp64;


  int k_iter = k / 2;
  int k_left = k % 2;

  __asm__ volatile( "prefetcht0 0(%0)    \n\t" : :"r"( a ) );
  __asm__ volatile( "prefetcht2 0(%0)    \n\t" : :"r"( aux->b_next ) );


  c03_0.v = _mm256_setzero_pd();
  c03_1.v = _mm256_setzero_pd();
  c03_2.v = _mm256_setzero_pd();
  c03_3.v = _mm256_setzero_pd();
  c47_0.v = _mm256_setzero_pd();
  c47_1.v = _mm256_setzero_pd();
  c47_2.v = _mm256_setzero_pd();
  c47_3.v = _mm256_setzero_pd();

  //printf("k: %d\n", k);

  //printf("come inside the micro kernel, k_iter: %d\n", k_iter);

  // Load a03
  a03.v = _mm256_load_pd(      (double*)a         );
  // Load a47
  a47.v = _mm256_load_pd(      (double*)( a + 4 ) );
  // Load (b0,b1,b2,b3)
  b0.v  = _mm256_load_pd(      (double*)b         );

  for ( i = 0; i < k_iter; ++i ) {
    __asm__ volatile( "prefetcht0 192(%0)    \n\t" : :"r"(a) );

    // Preload A03
    A03.v = _mm256_load_pd(      (double*)( a + 8 ) );

    c_tmp.v = _mm256_mul_pd( a03.v  , b0.v    );
    c03_0.v = _mm256_add_pd( c_tmp.v, c03_0.v );
    c_tmp.v = _mm256_mul_pd( a47.v  , b0.v    );
    c47_0.v = _mm256_add_pd( c_tmp.v, c47_0.v );

    // Preload A47
    A47.v = _mm256_load_pd(      (double*)( a + 12 ) );

    // Shuffle b ( 1, 0, 3, 2 )
    b1.v  = _mm256_shuffle_pd( b0.v, b0.v, 0x5 );

	//a03_1.v = _mm256_shuffle_pd( a0

    c_tmp.v = _mm256_mul_pd( a03.v  , b1.v    );
    c03_1.v = _mm256_add_pd( c_tmp.v, c03_1.v );
    c_tmp.v = _mm256_mul_pd( a47.v  , b1.v    );
    c47_1.v = _mm256_add_pd( c_tmp.v, c47_1.v );

    // Permute b ( 3, 2, 1, 0 )
    b2.v  = _mm256_permute2f128_pd( b1.v, b1.v, 0x1 );

    // Preload B0
    B0.v  = _mm256_load_pd(      (double*)( b + 4 ) );

    c_tmp.v = _mm256_mul_pd( a03.v  , b2.v    );
    c03_2.v = _mm256_add_pd( c_tmp.v, c03_2.v );
    c_tmp.v = _mm256_mul_pd( a47.v  , b2.v    );
    c47_2.v = _mm256_add_pd( c_tmp.v, c47_2.v );

    // Shuffle b ( 3, 2, 1, 0 )
    b3.v  = _mm256_shuffle_pd( b2.v, b2.v, 0x5 );

    c_tmp.v = _mm256_mul_pd( a03.v  , b3.v    );
    c03_3.v = _mm256_add_pd( c_tmp.v, c03_3.v );
    c_tmp.v = _mm256_mul_pd( a47.v  , b3.v    );
    c47_3.v = _mm256_add_pd( c_tmp.v, c47_3.v );


    // Iteration #1
    __asm__ volatile( "prefetcht0 512(%0)    \n\t" : :"r"(a) );

    // Preload a03 ( next iteration )
    a03.v = _mm256_load_pd(      (double*)( a + 16 ) );

    c_tmp.v = _mm256_mul_pd( A03.v  , B0.v    );
    c03_0.v = _mm256_add_pd( c_tmp.v, c03_0.v );

    b1.v  = _mm256_shuffle_pd( B0.v, B0.v, 0x5 );

    c_tmp.v = _mm256_mul_pd( A47.v  , B0.v    );
    c47_0.v = _mm256_add_pd( c_tmp.v, c47_0.v );
    c_tmp.v = _mm256_mul_pd( A03.v  , b1.v    );
    c03_1.v = _mm256_add_pd( c_tmp.v, c03_1.v );

    // Preload a47 ( next iteration )
    a47.v = _mm256_load_pd(      (double*)( a + 20 ) );

    // Permute b ( 3, 2, 1, 0 )
    b2.v  = _mm256_permute2f128_pd( b1.v, b1.v, 0x1 );

    c_tmp.v = _mm256_mul_pd( A47.v  , b1.v    );
    c47_1.v = _mm256_add_pd( c_tmp.v, c47_1.v );
    c_tmp.v = _mm256_mul_pd( A03.v  , b2.v    );
    c03_2.v = _mm256_add_pd( c_tmp.v, c03_2.v );

    // Shuffle b ( 3, 2, 1, 0 )
    b3.v  = _mm256_shuffle_pd( b2.v, b2.v, 0x5 );

    c_tmp.v = _mm256_mul_pd( A47.v  , b2.v    );
    c47_2.v = _mm256_add_pd( c_tmp.v, c47_2.v );

    // Load b0 ( next iteration )
    b0.v  = _mm256_load_pd(      (double*)( b + 8 ) );

    c_tmp.v = _mm256_mul_pd( A03.v  , b3.v    );
    c03_3.v = _mm256_add_pd( c_tmp.v, c03_3.v );
    c_tmp.v = _mm256_mul_pd( A47.v  , b3.v    );
    c47_3.v = _mm256_add_pd( c_tmp.v, c47_3.v );

    a += 16;
    b += 8;
  }

  //printf(" Come inside k_left: %d\n", k_left);
  for ( i = 0; i < k_left; ++i ) {
    a03.v = _mm256_load_pd(      (double*)a         );
    //printf( "a03 = %lf, %lf, %lf, %lf\n", a03.d[0], a03.d[1], a03.d[2], a03.d[3] );

    a47.v = _mm256_load_pd(      (double*)( a + 4 ) );
    //printf( "a47 = %lf, %lf, %lf, %lf\n", a47.d[0], a47.d[1], a47.d[2], a47.d[3] );

    b0.v  = _mm256_load_pd(      (double*)b         );
    //printf( "b0  = %lf, %lf, %lf, %lf\n", b0.d[0], b0.d[1], b0.d[2], b0.d[3] );

    c_tmp.v = _mm256_mul_pd( a03.v  , b0.v    );
    c03_0.v = _mm256_add_pd( c_tmp.v, c03_0.v );
    c_tmp.v = _mm256_mul_pd( a47.v  , b0.v    );
    c47_0.v = _mm256_add_pd( c_tmp.v, c47_0.v );

    // Shuffle b ( 1, 0, 3, 2 )
    b1.v  = _mm256_shuffle_pd( b0.v, b0.v, 0x5 );

    c_tmp.v = _mm256_mul_pd( a03.v  , b1.v    );
    c03_1.v = _mm256_add_pd( c_tmp.v, c03_1.v );
    c_tmp.v = _mm256_mul_pd( a47.v  , b1.v    );
    c47_1.v = _mm256_add_pd( c_tmp.v, c47_1.v );

    // Permute b ( 3, 2, 1, 0 )
    b2.v  = _mm256_permute2f128_pd( b1.v, b1.v, 0x1 );

    c_tmp.v = _mm256_mul_pd( a03.v  , b2.v    );
    c03_2.v = _mm256_add_pd( c_tmp.v, c03_2.v );
    c_tmp.v = _mm256_mul_pd( a47.v  , b2.v    );
    c47_2.v = _mm256_add_pd( c_tmp.v, c47_2.v );

    // Shuffle b ( 3, 2, 1, 0 )
    b3.v  = _mm256_shuffle_pd( b2.v, b2.v, 0x5 );

    c_tmp.v = _mm256_mul_pd( a03.v  , b3.v    );
    c03_3.v = _mm256_add_pd( c_tmp.v, c03_3.v );
    c_tmp.v = _mm256_mul_pd( a47.v  , b3.v    );
    c47_3.v = _mm256_add_pd( c_tmp.v, c47_3.v );

    a += 8;
    b += 4;
  }
 

  // Prefetch aa and bb
  __asm__ volatile( "prefetcht0 0(%0)    \n\t" : :"r"( aa ) );
  __asm__ volatile( "prefetcht0 0(%0)    \n\t" : :"r"( bb ) );


  tmpc03_0.v = _mm256_blend_pd( c03_0.v, c03_1.v, 0x6 );
  tmpc03_1.v = _mm256_blend_pd( c03_1.v, c03_0.v, 0x6 );
  
  tmpc03_2.v = _mm256_blend_pd( c03_2.v, c03_3.v, 0x6 );
  tmpc03_3.v = _mm256_blend_pd( c03_3.v, c03_2.v, 0x6 );

  tmpc47_0.v = _mm256_blend_pd( c47_0.v, c47_1.v, 0x6 );
  tmpc47_1.v = _mm256_blend_pd( c47_1.v, c47_0.v, 0x6 );

  tmpc47_2.v = _mm256_blend_pd( c47_2.v, c47_3.v, 0x6 );
  tmpc47_3.v = _mm256_blend_pd( c47_3.v, c47_2.v, 0x6 );

  c03_0.v    = _mm256_permute2f128_pd( tmpc03_0.v, tmpc03_2.v, 0x30 );
  c03_3.v    = _mm256_permute2f128_pd( tmpc03_2.v, tmpc03_0.v, 0x30 );

  c03_1.v    = _mm256_permute2f128_pd( tmpc03_1.v, tmpc03_3.v, 0x30 );
  c03_2.v    = _mm256_permute2f128_pd( tmpc03_3.v, tmpc03_1.v, 0x30 );

  c47_0.v    = _mm256_permute2f128_pd( tmpc47_0.v, tmpc47_2.v, 0x30 );
  c47_3.v    = _mm256_permute2f128_pd( tmpc47_2.v, tmpc47_0.v, 0x30 );

  c47_1.v    = _mm256_permute2f128_pd( tmpc47_1.v, tmpc47_3.v, 0x30 );
  c47_2.v    = _mm256_permute2f128_pd( tmpc47_3.v, tmpc47_1.v, 0x30 );

  //printf( "Before transpose\n" );
  //printf( "%lf, %lf, %lf, %lf\n", c03_0.d[0], c03_1.d[0], c03_2.d[0], c03_3.d[0] );
  //printf( "%lf, %lf, %lf, %lf\n", c03_0.d[1], c03_1.d[1], c03_2.d[1], c03_3.d[1] );
  //printf( "%lf, %lf, %lf, %lf\n", c03_0.d[2], c03_1.d[2], c03_2.d[2], c03_3.d[2] );
  //printf( "%lf, %lf, %lf, %lf\n", c03_0.d[3], c03_1.d[3], c03_2.d[3], c03_3.d[3] );
  //printf( "%lf, %lf, %lf, %lf\n", c47_0.d[0], c47_1.d[0], c47_2.d[0], c47_3.d[0] );
  //printf( "%lf, %lf, %lf, %lf\n", c47_0.d[1], c47_1.d[1], c47_2.d[1], c47_3.d[1] );
  //printf( "%lf, %lf, %lf, %lf\n", c47_0.d[2], c47_1.d[2], c47_2.d[2], c47_3.d[2] );
  //printf( "%lf, %lf, %lf, %lf\n", c47_0.d[3], c47_1.d[3], c47_2.d[3], c47_3.d[3] );

 

  //for ( i = 0; i < k; i++ ) {
  //  a03.v = _mm256_load_pd(      (double*)a         );
  //  a47.v = _mm256_load_pd(      (double*)( a + 4 ) );
  //  b0.v  = _mm256_broadcast_sd( (double*)b         );
  //  b1.v  = _mm256_broadcast_sd( (double*)( b + 1 ) );
  //  b2.v  = _mm256_broadcast_sd( (double*)( b + 2 ) );
  //  b3.v  = _mm256_broadcast_sd( (double*)( b + 3 ) );

  //  a += DKS_MR;
  //  b += DKS_NR;

  //  c_tmp.v = _mm256_mul_pd( a03.v  , b0.v    );
  //  c03_0.v = _mm256_add_pd( c_tmp.v, c03_0.v );
  //  c_tmp.v = _mm256_mul_pd( a03.v  , b1.v    );
  //  c03_1.v = _mm256_add_pd( c_tmp.v, c03_1.v );
  //  c_tmp.v = _mm256_mul_pd( a03.v  , b2.v    );
  //  c03_2.v = _mm256_add_pd( c_tmp.v, c03_2.v );
  //  c_tmp.v = _mm256_mul_pd( a03.v  , b3.v    );
  //  c03_3.v = _mm256_add_pd( c_tmp.v, c03_3.v );

  //  c_tmp.v = _mm256_mul_pd( a47.v  , b0.v    );
  //  c47_0.v = _mm256_add_pd( c_tmp.v, c47_0.v );
  //  c_tmp.v = _mm256_mul_pd( a47.v  , b1.v    );
  //  c47_1.v = _mm256_add_pd( c_tmp.v, c47_1.v );
  //  c_tmp.v = _mm256_mul_pd( a47.v  , b2.v    );
  //  c47_2.v = _mm256_add_pd( c_tmp.v, c47_2.v );
  //  c_tmp.v = _mm256_mul_pd( a47.v  , b3.v    );
  //  c47_3.v = _mm256_add_pd( c_tmp.v, c47_3.v );
  //}
 
  __asm__ volatile( "prefetcht0 0(%0)    \n\t" : :"r"( aux->I ) );
  __asm__ volatile( "prefetcht0 0(%0)    \n\t" : :"r"( aux->D ) );



  aa_tmp.v = _mm256_broadcast_sd( &neg2 );
  //c03_0.v  = _mm256_mul_pd( aa_tmp.v, c03_0.v );
  //c03_1.v  = _mm256_mul_pd( aa_tmp.v, c03_1.v );
  //c03_2.v  = _mm256_mul_pd( aa_tmp.v, c03_2.v );
  //c03_3.v  = _mm256_mul_pd( aa_tmp.v, c03_3.v );
  //c47_0.v  = _mm256_mul_pd( aa_tmp.v, c47_0.v );
  //c47_1.v  = _mm256_mul_pd( aa_tmp.v, c47_1.v );
  //c47_2.v  = _mm256_mul_pd( aa_tmp.v, c47_2.v );
  //c47_3.v  = _mm256_mul_pd( aa_tmp.v, c47_3.v );
  //
  c03_0.v  = _mm256_mul_pd( aa_tmp.v, c03_0.v );
  c03_1.v  = _mm256_mul_pd( aa_tmp.v, c03_1.v );
  c03_2.v  = _mm256_mul_pd( aa_tmp.v, c03_2.v );
  c03_3.v  = _mm256_mul_pd( aa_tmp.v, c03_3.v );
  c47_0.v  = _mm256_mul_pd( aa_tmp.v, c47_0.v );
  c47_1.v  = _mm256_mul_pd( aa_tmp.v, c47_1.v );
  c47_2.v  = _mm256_mul_pd( aa_tmp.v, c47_2.v );
  c47_3.v  = _mm256_mul_pd( aa_tmp.v, c47_3.v );


  //printf( "scale -2 \n" );
  //printf( "%lf, %lf, %lf, %lf\n", c03_0.d[0], c03_1.d[0], c03_2.d[0], c03_3.d[0] );
  //printf( "%lf, %lf, %lf, %lf\n", c03_0.d[1], c03_1.d[1], c03_2.d[1], c03_3.d[1] );
  //printf( "%lf, %lf, %lf, %lf\n", c03_0.d[2], c03_1.d[2], c03_2.d[2], c03_3.d[2] );
  //printf( "%lf, %lf, %lf, %lf\n", c03_0.d[3], c03_1.d[3], c03_2.d[3], c03_3.d[3] );
  //printf( "%lf, %lf, %lf, %lf\n", c47_0.d[0], c47_1.d[0], c47_2.d[0], c47_3.d[0] );
  //printf( "%lf, %lf, %lf, %lf\n", c47_0.d[1], c47_1.d[1], c47_2.d[1], c47_3.d[1] );
  //printf( "%lf, %lf, %lf, %lf\n", c47_0.d[2], c47_1.d[2], c47_2.d[2], c47_3.d[2] );
  //printf( "%lf, %lf, %lf, %lf\n", c47_0.d[3], c47_1.d[3], c47_2.d[3], c47_3.d[3] );


  aa_tmp.v = _mm256_load_pd( (double*)aa );
  c03_0.v  = _mm256_add_pd( aa_tmp.v, c03_0.v );
  c03_1.v  = _mm256_add_pd( aa_tmp.v, c03_1.v );
  c03_2.v  = _mm256_add_pd( aa_tmp.v, c03_2.v );
  c03_3.v  = _mm256_add_pd( aa_tmp.v, c03_3.v );

  //printf( "aa03 = %lf, %lf, %lf, %lf\n", aa_tmp.d[0], aa_tmp.d[1], aa_tmp.d[2], aa_tmp.d[3] );
  //printf( "bb03 = %lf, %lf, %lf, %lf\n", bb[ 0 ], bb[ 1 ], bb[ 2 ], bb[ 3 ] );

  aa_tmp.v = _mm256_load_pd( (double*)( aa + 4 ) );
  c47_0.v  = _mm256_add_pd( aa_tmp.v, c47_0.v );
  c47_1.v  = _mm256_add_pd( aa_tmp.v, c47_1.v );
  c47_2.v  = _mm256_add_pd( aa_tmp.v, c47_2.v );
  c47_3.v  = _mm256_add_pd( aa_tmp.v, c47_3.v );
  

  //printf( "add a^2\n" );
  //printf( "%lf, %lf, %lf, %lf\n", c03_0.d[0], c03_1.d[0], c03_2.d[0], c03_3.d[0] );
  //printf( "%lf, %lf, %lf, %lf\n", c03_0.d[1], c03_1.d[1], c03_2.d[1], c03_3.d[1] );
  //printf( "%lf, %lf, %lf, %lf\n", c03_0.d[2], c03_1.d[2], c03_2.d[2], c03_3.d[2] );
  //printf( "%lf, %lf, %lf, %lf\n", c03_0.d[3], c03_1.d[3], c03_2.d[3], c03_3.d[3] );
  //printf( "%lf, %lf, %lf, %lf\n", c47_0.d[0], c47_1.d[0], c47_2.d[0], c47_3.d[0] );
  //printf( "%lf, %lf, %lf, %lf\n", c47_0.d[1], c47_1.d[1], c47_2.d[1], c47_3.d[1] );
  //printf( "%lf, %lf, %lf, %lf\n", c47_0.d[2], c47_1.d[2], c47_2.d[2], c47_3.d[2] );
  //printf( "%lf, %lf, %lf, %lf\n", c47_0.d[3], c47_1.d[3], c47_2.d[3], c47_3.d[3] );




  bb_tmp.v = _mm256_broadcast_sd( (double*)bb );
  c03_0.v  = _mm256_add_pd( bb_tmp.v, c03_0.v );
  c47_0.v  = _mm256_add_pd( bb_tmp.v, c47_0.v );

  bb_tmp.v = _mm256_broadcast_sd( (double*)( bb + 1 ) );
  c03_1.v  = _mm256_add_pd( bb_tmp.v, c03_1.v );
  c47_1.v  = _mm256_add_pd( bb_tmp.v, c47_1.v );

  bb_tmp.v = _mm256_broadcast_sd( (double*)( bb + 2 ) );
  c03_2.v  = _mm256_add_pd( bb_tmp.v, c03_2.v );
  c47_2.v  = _mm256_add_pd( bb_tmp.v, c47_2.v );

  bb_tmp.v = _mm256_broadcast_sd( (double*)( bb + 3 ) );
  c03_3.v  = _mm256_add_pd( bb_tmp.v, c03_3.v );
  c47_3.v  = _mm256_add_pd( bb_tmp.v, c47_3.v );

  //printf( "add b^2\n" );
  //printf( "%lf, %lf, %lf, %lf\n", c03_0.d[0], c03_1.d[0], c03_2.d[0], c03_3.d[0] );
  //printf( "%lf, %lf, %lf, %lf\n", c03_0.d[1], c03_1.d[1], c03_2.d[1], c03_3.d[1] );
  //printf( "%lf, %lf, %lf, %lf\n", c03_0.d[2], c03_1.d[2], c03_2.d[2], c03_3.d[2] );
  //printf( "%lf, %lf, %lf, %lf\n", c03_0.d[3], c03_1.d[3], c03_2.d[3], c03_3.d[3] );
  //printf( "%lf, %lf, %lf, %lf\n", c47_0.d[0], c47_1.d[0], c47_2.d[0], c47_3.d[0] );
  //printf( "%lf, %lf, %lf, %lf\n", c47_0.d[1], c47_1.d[1], c47_2.d[1], c47_3.d[1] );
  //printf( "%lf, %lf, %lf, %lf\n", c47_0.d[2], c47_1.d[2], c47_2.d[2], c47_3.d[2] );
  //printf( "%lf, %lf, %lf, %lf\n", c47_0.d[3], c47_1.d[3], c47_2.d[3], c47_3.d[3] );


  // Transpose c03/c47 _0, _1, _2, _3 to be the row vector
  tmpc03_0.v = _mm256_shuffle_pd( c03_0.v, c03_1.v, 0x0 );
  tmpc03_1.v = _mm256_shuffle_pd( c03_0.v, c03_1.v, 0xF );

  tmpc03_2.v = _mm256_shuffle_pd( c03_2.v, c03_3.v, 0x0 );
  tmpc03_3.v = _mm256_shuffle_pd( c03_2.v, c03_3.v, 0xF );

  tmpc47_0.v = _mm256_shuffle_pd( c47_0.v, c47_1.v, 0x0 );
  tmpc47_1.v = _mm256_shuffle_pd( c47_0.v, c47_1.v, 0xF );

  tmpc47_2.v = _mm256_shuffle_pd( c47_2.v, c47_3.v, 0x0 );
  tmpc47_3.v = _mm256_shuffle_pd( c47_2.v, c47_3.v, 0xF );

  //printf( "Middle Stage\n" );
  //printf( "%lf, %lf, %lf, %lf\n", tmpc03_0.d[0], tmpc03_1.d[0], tmpc03_2.d[0], tmpc03_3.d[0] );
  //printf( "%lf, %lf, %lf, %lf\n", tmpc03_0.d[1], tmpc03_1.d[1], tmpc03_2.d[1], tmpc03_3.d[1] );
  //printf( "%lf, %lf, %lf, %lf\n", tmpc03_0.d[2], tmpc03_1.d[2], tmpc03_2.d[2], tmpc03_3.d[2] );
  //printf( "%lf, %lf, %lf, %lf\n", tmpc03_0.d[3], tmpc03_1.d[3], tmpc03_2.d[3], tmpc03_3.d[3] );
  //printf( "%lf, %lf, %lf, %lf\n", tmpc47_0.d[0], tmpc47_1.d[0], tmpc47_2.d[0], tmpc47_3.d[0] );
  //printf( "%lf, %lf, %lf, %lf\n", tmpc47_0.d[1], tmpc47_1.d[1], tmpc47_2.d[1], tmpc47_3.d[1] );
  //printf( "%lf, %lf, %lf, %lf\n", tmpc47_0.d[2], tmpc47_1.d[2], tmpc47_2.d[2], tmpc47_3.d[2] );
  //printf( "%lf, %lf, %lf, %lf\n", tmpc47_0.d[3], tmpc47_1.d[3], tmpc47_2.d[3], tmpc47_3.d[3] );


  c03_0.v    = _mm256_permute2f128_pd( tmpc03_0.v, tmpc03_2.v, 0x20 );
  c03_2.v    = _mm256_permute2f128_pd( tmpc03_0.v, tmpc03_2.v, 0x31 );

  c03_1.v    = _mm256_permute2f128_pd( tmpc03_1.v, tmpc03_3.v, 0x20 );
  c03_3.v    = _mm256_permute2f128_pd( tmpc03_1.v, tmpc03_3.v, 0x31 );

  c47_0.v    = _mm256_permute2f128_pd( tmpc47_0.v, tmpc47_2.v, 0x20 );
  c47_2.v    = _mm256_permute2f128_pd( tmpc47_0.v, tmpc47_2.v, 0x31 );

  c47_1.v    = _mm256_permute2f128_pd( tmpc47_1.v, tmpc47_3.v, 0x20 );
  c47_3.v    = _mm256_permute2f128_pd( tmpc47_1.v, tmpc47_3.v, 0x31 );

  //printf( "c03/c47 \n" );
  //printf( "%lf, %lf, %lf, %lf\n", c03_0.d[0], c03_1.d[0], c03_2.d[0], c03_3.d[0] );
  //printf( "%lf, %lf, %lf, %lf\n", c03_0.d[1], c03_1.d[1], c03_2.d[1], c03_3.d[1] );
  //printf( "%lf, %lf, %lf, %lf\n", c03_0.d[2], c03_1.d[2], c03_2.d[2], c03_3.d[2] );
  //printf( "%lf, %lf, %lf, %lf\n", c03_0.d[3], c03_1.d[3], c03_2.d[3], c03_3.d[3] );
  //printf( "%lf, %lf, %lf, %lf\n", c47_0.d[0], c47_1.d[0], c47_2.d[0], c47_3.d[0] );
  //printf( "%lf, %lf, %lf, %lf\n", c47_0.d[1], c47_1.d[1], c47_2.d[1], c47_3.d[1] );
  //printf( "%lf, %lf, %lf, %lf\n", c47_0.d[2], c47_1.d[2], c47_2.d[2], c47_3.d[2] );
  //printf( "%lf, %lf, %lf, %lf\n", c47_0.d[3], c47_1.d[3], c47_2.d[3], c47_3.d[3] );

  // Check if there is any illegle value 
  tmp.v    = _mm256_broadcast_sd( &dzero );
  c03_0.v  = _mm256_max_pd( tmp.v, c03_0.v );
  c03_1.v  = _mm256_max_pd( tmp.v, c03_1.v );
  c03_2.v  = _mm256_max_pd( tmp.v, c03_2.v );
  c03_3.v  = _mm256_max_pd( tmp.v, c03_3.v );
  c47_0.v  = _mm256_max_pd( tmp.v, c47_0.v );
  c47_1.v  = _mm256_max_pd( tmp.v, c47_1.v );
  c47_2.v  = _mm256_max_pd( tmp.v, c47_2.v );
  c47_3.v  = _mm256_max_pd( tmp.v, c47_3.v );


  //1. Change the vector into row vector way...
  //ct03_0, ct03_1, ct03_2, ct03_3, ct47_4, ct47_5, ct47_6, ct47_7
  //This is down after C has been calculated.

  //2. Load I into a 256-bit vector ii, (notice alignment! using loadu)
  //   load D into a 256-bit vector d, (notice alignment! using loadu)
  //   Load I into index, using SSE3
  index.v = _mm_loadu_si128(            I          );
  //   Load D into data
  data.v  = _mm256_loadu_pd(      (double*)D       );



  //printf( "%d, %d, %d, %d\n", index.d[0], index.d[1], index.d[2], index.d[3] );
  //printf( "%lf, %lf, %lf, %lf\n", data.d[0], data.d[1], data.d[2], data.d[3] );

  //.........initialize D = inf................!!!!!!!
  //using inf? exception? signal???

  //3. Compare D with the row vector ct03_0, ct03_1, ct03_2, ct03_3, ct47_4, ct47_5, ct47_6, ct47_7, the cmp result is imm8.
  //Broadcast amap[ic] into aa (e.g. ct47_5 -> ic =5)
  //ii = blend(ii, aa, imm8)
 
  //using 11 instead of 1, thus no signaling..
  temp_cmp64.v = _mm256_cmp_pd( c03_0.v, data.v, 1 );

  //printf("temp_cmp64:\n");
  //udouble_t temp;
  //for (i = 0 ; i < 4; i ++ ) {
  //  temp.d = temp_cmp64.d[0];
  //  printf(" %lx,", temp.u );
  //}
  //printf("\n");

  //temp_cmp32.v = _mm256_cvttpd_epi32(temp_cmp64.v);
  temp_cmp32.v = _mm_castps_si128(_mm256_cvtpd_ps(temp_cmp64.v));

  //printf("temp_cmp32:\n");
  //printf( "%x, %x, %x, %x\n", temp_cmp32.d[0], temp_cmp32.d[1], temp_cmp32.d[2], temp_cmp32.d[3] );

  //temp_brd32.v = _mm_broadcastd_epi32( amap[0] );
  //We still have some problems with the address of amap or the function _mm_set1_epi32
  //printf("amap:\n");
  //for (i = 0; i < 8; i ++) {
  //  printf("%d\t", amap[i]);
  //}
  //printf("\n");
  temp_brd32.v = _mm_set1_epi32( amap[0] );

  //printf("temp_brd32:\n");
  //printf( "%d, %d, %d, %d\n", temp_brd32.d[0], temp_brd32.d[1], temp_brd32.d[2], temp_brd32.d[3] );


  //__mm_broadcast_ss?
  data.v = _mm256_blendv_pd( data.v, c03_0.v, temp_cmp64.v);
  //printf("c03_0.v:\n");
  //printf( "%lf, %lf, %lf, %lf\n", c03_0.d[0], c03_0.d[1], c03_0.d[2], c03_0.d[3] );
  //printf("data.v:\n");
  //printf( "%lf, %lf, %lf, %lf\n", data.d[0], data.d[1], data.d[2], data.d[3] );


  //index.v = _mm_castps_si128 (_mm_blendv_ps( _mm_castsi128_ps(temp_brd32.v), _mm_castsi128_ps(index.v), _mm_castsi128_ps(temp_cmp32.v)));
  index.v = _mm_blendv_epi8(index.v, temp_brd32.v, temp_cmp32.v);

  //printf("index.v:\n");
  //printf( "%d, %d, %d, %d\n", index.d[0], index.d[1], index.d[2], index.d[3] );




  temp_cmp64.v = _mm256_cmp_pd( c03_1.v, data.v, 1 );
  temp_cmp32.v = _mm_castps_si128(_mm256_cvtpd_ps(temp_cmp64.v));
  temp_brd32.v = _mm_set1_epi32( amap[1] );

  data.v = _mm256_blendv_pd( data.v, c03_1.v, temp_cmp64.v);
  index.v = _mm_blendv_epi8( index.v, temp_brd32.v, temp_cmp32.v);

  temp_cmp64.v = _mm256_cmp_pd( c03_2.v, data.v, 1 );
  temp_cmp32.v = _mm_castps_si128(_mm256_cvtpd_ps(temp_cmp64.v));
  temp_brd32.v = _mm_set1_epi32( amap[2] );
  data.v = _mm256_blendv_pd( data.v, c03_2.v, temp_cmp64.v);
  index.v = _mm_blendv_epi8( index.v, temp_brd32.v, temp_cmp32.v);

  temp_cmp64.v = _mm256_cmp_pd( c03_3.v, data.v, 1 );
  temp_cmp32.v = _mm_castps_si128(_mm256_cvtpd_ps(temp_cmp64.v));
  temp_brd32.v = _mm_set1_epi32( amap[3] );
  data.v = _mm256_blendv_pd( data.v, c03_3.v, temp_cmp64.v);
  index.v = _mm_blendv_epi8( index.v, temp_brd32.v, temp_cmp32.v);


  temp_cmp64.v = _mm256_cmp_pd( c47_0.v, data.v, 1 );
  temp_cmp32.v = _mm_castps_si128(_mm256_cvtpd_ps(temp_cmp64.v));
  temp_brd32.v = _mm_set1_epi32( amap[4] );
  data.v = _mm256_blendv_pd( data.v, c47_0.v, temp_cmp64.v);
  index.v = _mm_blendv_epi8(index.v, temp_brd32.v, temp_cmp32.v);

  temp_cmp64.v = _mm256_cmp_pd( c47_1.v, data.v, 1 );
  temp_cmp32.v = _mm_castps_si128(_mm256_cvtpd_ps(temp_cmp64.v));
  temp_brd32.v = _mm_set1_epi32( amap[5] );
  data.v = _mm256_blendv_pd( data.v, c47_1.v, temp_cmp64.v);
  index.v = _mm_blendv_epi8( index.v, temp_brd32.v,  temp_cmp32.v);

  temp_cmp64.v = _mm256_cmp_pd( c47_2.v, data.v, 1 );
  temp_cmp32.v = _mm_castps_si128(_mm256_cvtpd_ps(temp_cmp64.v));
  temp_brd32.v = _mm_set1_epi32( amap[6] );
  data.v = _mm256_blendv_pd( data.v, c47_2.v, temp_cmp64.v);
  index.v = _mm_blendv_epi8( index.v, temp_brd32.v, temp_cmp32.v);

  temp_cmp64.v = _mm256_cmp_pd( c47_3.v, data.v, 1 );
  temp_cmp32.v = _mm_castps_si128(_mm256_cvtpd_ps(temp_cmp64.v));
  temp_brd32.v = _mm_set1_epi32( amap[7] );
  data.v = _mm256_blendv_pd( data.v, c47_3.v, temp_cmp64.v);
  index.v = _mm_blendv_epi8( index.v, temp_brd32.v, temp_cmp32.v);



  //printf("final vector result:\n");
  //printf( "%d, %d, %d, %d\n", index.d[0], index.d[1], index.d[2], index.d[3] );
  //printf( "%lf, %lf, %lf, %lf\n", data.d[0], data.d[1], data.d[2], data.d[3] );




  //4. store the 128-bit vector into I (notice alignment! using loadu)
  //   store the 256-bit vector into D (notice alignment! using loadu)
  _mm_storeu_si128( (__m128i *)I, index.v );
  _mm256_storeu_pd( (double *)D, data.v );


}





