#include <immintrin.h> // AVX
#include <rnn.h>

/*
 * Loop over a and b, and perform a length 8 ( 4 + 4 ) merging by using the
 * same algorithm in rnn_merge_int_d8().
 *
 * In each iteration,
 *
 * y0, y1 are input keys, and y2, y3 are input indexes.
 *
 * The first part of the key output will be stored in y4, and the second part
 * will be store in y1 again. Thus, at the begining of each iteration we have
 * to load a new vector ( part fo a or b ) to y0, and a new vector ( part of
 * amap and bmap ) to y2. At the end of each iteration, we need to stor back
 * y0 and y2 to D and I and increase the index by 4.
 *
 *
 * */

void rnn_merge_int_dn_var2(
    double *a,
    double *amap,
    double *b,
    double *bmap,
    double *D,
    double *I,
    int    na,
    int    nb
    )
{
  v4df_t y0,  y1,  y2,  y3;
  v4df_t y4,  y5,  y6,  y7;
  v4df_t y15;

  int    a_ptr = 0;
  int    b_ptr = 0;
  int    d_ptr = 0;


  // Prepare y0 and y2
  // TODO: boundary check
  if ( a[ a_ptr ] < b[ b_ptr ] ) {
    __asm__ volatile( "prefetcht0 32(%0)    \n\t" : :"r"( a + a_ptr ) );
    __asm__ volatile( "prefetcht0 32(%0)    \n\t" : :"r"( amap + b_ptr ) );
    y0.v     = _mm256_load_pd( a      );
    y2.v     = _mm256_load_pd( amap   );
    a_ptr += 4;
  }
  else {
    __asm__ volatile( "prefetcht0 32(%0)    \n\t" : :"r"( b + b_ptr) );
    __asm__ volatile( "prefetcht0 32(%0)    \n\t" : :"r"( bmap + b_ptr ) );
    y0.v     = _mm256_load_pd( b      );
    y2.v     = _mm256_load_pd( bmap   );
    b_ptr += 4;
  }  

  // Loop until one of the array is fully merged.
  while ( a_ptr < na || b_ptr < nb ) {

    // normal case
    if ( a_ptr < na && b_ptr < nb ) {
      // Check which key is smaller.
      if ( a[ a_ptr ] < b[ b_ptr ] ) {
        __asm__ volatile( "prefetcht0 32(%0)    \n\t" : :"r"( a + a_ptr ) );
        __asm__ volatile( "prefetcht0 32(%0)    \n\t" : :"r"( amap + a_ptr ) );
        y1.v     = _mm256_load_pd( a    + a_ptr );
        y3.v     = _mm256_load_pd( amap + a_ptr );
        a_ptr += 4;
      }
      else {
        __asm__ volatile( "prefetcht0 32(%0)    \n\t" : :"r"( b + b_ptr) );
        __asm__ volatile( "prefetcht0 32(%0)    \n\t" : :"r"( bmap + b_ptr ) );
        y1.v     = _mm256_load_pd( b    + b_ptr );
        y3.v     = _mm256_load_pd( bmap + b_ptr );
        b_ptr += 4;
      }
    }
    else {
      // tail case
      if ( a_ptr >= na ) {
        __asm__ volatile( "prefetcht0 32(%0)    \n\t" : :"r"( a + a_ptr ) );
        __asm__ volatile( "prefetcht0 32(%0)    \n\t" : :"r"( amap + a_ptr ) );
        y1.v     = _mm256_load_pd( b    + b_ptr );
        y3.v     = _mm256_load_pd( bmap + b_ptr );
        b_ptr += 4;
      }
      else {
        __asm__ volatile( "prefetcht0 32(%0)    \n\t" : :"r"( b + b_ptr) );
        __asm__ volatile( "prefetcht0 32(%0)    \n\t" : :"r"( bmap + b_ptr ) );
        y1.v     = _mm256_load_pd( a    + a_ptr );
        y3.v     = _mm256_load_pd( amap + a_ptr );
        a_ptr += 4;
      }
    }



    // First reverse y1 and y3
    y1.v     = _mm256_permute_pd( y1.v, 0x5 );
    y3.v     = _mm256_permute_pd( y3.v, 0x5 );
    y1.v     = _mm256_permute2f128_pd( y1.v, y1.v, 0x1 );
    y3.v     = _mm256_permute2f128_pd( y3.v, y3.v, 0x1 );

    //printf( "reverse:\n" );
    //printf( "y1 = %5.2lf, %5.2lf, %5.2lf, %5.2lf\n",
    //    y1.d[ 0 ], y1.d[ 1 ], y1.d[ 2 ], y1.d[ 3 ] );

    __asm__ volatile( "prefetcht0 0(%0)    \n\t" : :"r"( D + d_ptr) );
    // Level 1 comparison
    y15.v    = _mm256_cmp_pd( y0.v, y1.v, 30 );
    y4.v     = _mm256_blendv_pd( y0.v, y1.v, y15.v ); // L1
    y5.v     = _mm256_blendv_pd( y1.v, y0.v, y15.v ); // H1
    y6.v     = _mm256_blendv_pd( y2.v, y3.v, y15.v ); //
    y7.v     = _mm256_blendv_pd( y3.v, y2.v, y15.v ); //
    // Shuffle
    y0.v     = _mm256_permute2f128_pd( y4.v, y5.v, 0x30 ); // L1p
    y1.v     = _mm256_permute2f128_pd( y4.v, y5.v, 0x21 ); // H1p
    y2.v     = _mm256_permute2f128_pd( y6.v, y7.v, 0x30 ); //
    y3.v     = _mm256_permute2f128_pd( y6.v, y7.v, 0x21 ); //

    __asm__ volatile( "prefetcht0 0(%0)    \n\t" : :"r"( I + d_ptr) );
    // Level 2 comparison
    y15.v    = _mm256_cmp_pd( y0.v, y1.v, 30 );
    y4.v     = _mm256_blendv_pd( y0.v, y1.v, y15.v ); // L2
    y5.v     = _mm256_blendv_pd( y1.v, y0.v, y15.v ); // H2
    y6.v     = _mm256_blendv_pd( y2.v, y3.v, y15.v ); //
    y7.v     = _mm256_blendv_pd( y3.v, y2.v, y15.v ); //
    // Shuffle
    y0.v     = _mm256_shuffle_pd( y4.v, y5.v, 10 ); // L2p
    y1.v     = _mm256_shuffle_pd( y4.v, y5.v, 5 );  // H2p
    y2.v     = _mm256_shuffle_pd( y6.v, y7.v, 10 ); //
    y3.v     = _mm256_shuffle_pd( y6.v, y7.v, 5 );  //

    // Level 2 comparison
    y15.v    = _mm256_cmp_pd( y0.v, y1.v, 30 );
    y4.v     = _mm256_blendv_pd( y0.v, y1.v, y15.v ); // L3
    y5.v     = _mm256_blendv_pd( y1.v, y0.v, y15.v ); // H3
    y6.v     = _mm256_blendv_pd( y2.v, y3.v, y15.v ); //
    y7.v     = _mm256_blendv_pd( y3.v, y2.v, y15.v ); //
    // Shuffle
    y0.v     = _mm256_unpacklo_pd( y4.v, y5.v );
    y1.v     = _mm256_unpackhi_pd( y4.v, y5.v );
    y2.v     = _mm256_unpacklo_pd( y6.v, y7.v );
    y3.v     = _mm256_unpackhi_pd( y6.v, y7.v );
    // The second part of the output will be in y0 and y2.
    y4.v     = _mm256_permute2f128_pd( y0.v, y1.v, 0x20 );
    y0.v     = _mm256_permute2f128_pd( y0.v, y1.v, 0x31 );
    y6.v     = _mm256_permute2f128_pd( y2.v, y3.v, 0x20 );
    y2.v     = _mm256_permute2f128_pd( y2.v, y3.v, 0x31 );

    _mm256_store_pd( D + d_ptr, y4.v );
    _mm256_store_pd( I + d_ptr, y6.v );

    d_ptr += 4;
  }



  _mm256_store_pd( D + d_ptr, y0.v );
  _mm256_store_pd( I + d_ptr, y2.v );
}
