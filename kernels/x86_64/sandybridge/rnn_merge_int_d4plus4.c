#include <immintrin.h> // AVX
#include <rnn.h>

void rnn_merge_int_d4plus4(
	double *a,
	double *amap,
	double *b,
	double *bmap,
    double *d,
    double *i
    )
{
  v4df_t  y0,  y1,  y2,  y3;
  v4df_t  y4,  y5,  y6,  y7;
  v4df_t  y15;

  y0.v     = _mm256_load_pd( a     );
  y1.v     = _mm256_load_pd( b     );
  y2.v     = _mm256_load_pd( amap  );
  y3.v     = _mm256_load_pd( bmap  );

  
  // First reverse y1 and y3
  y1.v     = _mm256_permute_pd( y1.v, 0x5 );
  y3.v     = _mm256_permute_pd( y3.v, 0x5 );
  y1.v     = _mm256_permute2f128_pd( y1.v, y1.v, 0x1 );
  y3.v     = _mm256_permute2f128_pd( y3.v, y3.v, 0x1 );


  //printf( "reverse:\n" );
  //printf( "y1 = %5.2lf, %5.2lf, %5.2lf, %5.2lf\n",
  //    y1.d[ 0 ], y1.d[ 1 ], y1.d[ 2 ], y1.d[ 3 ] );


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

  y0.v     = _mm256_unpacklo_pd( y4.v, y5.v );
  y1.v     = _mm256_unpackhi_pd( y4.v, y5.v );
  y2.v     = _mm256_unpacklo_pd( y6.v, y7.v );
  y3.v     = _mm256_unpackhi_pd( y6.v, y7.v );

  y4.v     = _mm256_permute2f128_pd( y0.v, y1.v, 0x20 );
  y5.v     = _mm256_permute2f128_pd( y0.v, y1.v, 0x31 );
  y6.v     = _mm256_permute2f128_pd( y2.v, y3.v, 0x20 );
  y7.v     = _mm256_permute2f128_pd( y2.v, y3.v, 0x31 );



  _mm256_store_pd( d     , y4.v );
  _mm256_store_pd( d +  4, y5.v );
  //_mm256_store_pd( a     , y4.v );
  //_mm256_store_pd( b     , y5.v );
  //_mm256_store_pd( amap  , y6.v );
  //_mm256_store_pd( bmap  , y7.v );
  _mm256_store_pd( i     , y6.v );
  _mm256_store_pd( i +  4, y7.v );
}
