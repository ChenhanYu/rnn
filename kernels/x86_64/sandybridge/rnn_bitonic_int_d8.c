#include <immintrin.h> // AVX
#include <rnn.h>

void rnn_bitonic_int_d8(
    int    *i,
    double *d
    )
{
  double done = 1.0;
  //v4df_t c03, c47, c_perm, rank, one;
  //v4li_t r03, r_tmp; 

  v4df_t d03, d47, d_perm, rank, one;
  v4li_t i03, i47, r03, r_tmp; 
 

  v4df_t l03, h03, lh03;


  one.v     = _mm256_broadcast_sd( &done );



  // d0, d1, d2, d3
  d03.v     = _mm256_load_pd( d );
  // i0, i1, i2, i3
  i03.v     = _mm_load_si128( i );


  printf( "d0123 = %5.2lf, %5.2lf, %5.2lf, %5.2lf\n",
      d03.d[ 0 ], d03.d[ 1 ], d03.d[ 2 ], d03.d[ 3 ] );
  printf( "i0123 = %d, %d, %d, %d\n",
      i03.d[ 0 ], i03.d[ 1 ], i03.d[ 2 ], i03.d[ 3 ] );


  // ==============================================================
  // level#1
  // ==============================================================
  // d2, d3, d0, d1
  d_perm.v  = _mm256_permute2f128_pd( d03.v, d03.v, 0x1 );
  rank.v    = _mm256_cmp_pd( d03.v, d_perm.v, 30 );
  // l0, l1, l0, l1 = min( d0, d2 ), min( d1, d3 ), min( d0, d2 ), min( d1, d3 )
  l03.v     = _mm256_blendv_pd( d03.v, d_perm.v, rank.v );
  // h0, h1, h0, h1 = max( d0, d2 ), max( d1, d3 ), max( d0, d2 ), max( d1, d3 )
  h03.v     = _mm256_blendv_pd( d_perm.v, d03.v, rank.v );


  printf( "l03 = %5.2lf, %5.2lf, %5.2lf, %5.2lf\n",
      l03.d[ 0 ], l03.d[ 1 ], l03.d[ 2 ], l03.d[ 3 ] );
  printf( "h03 = %5.2lf, %5.2lf, %5.2lf, %5.2lf\n",
      h03.d[ 0 ], h03.d[ 1 ], h03.d[ 2 ], h03.d[ 3 ] );


  // l0 = min( d0, d2 )
  // l1 = min( d1, d3 )
  // h0 = max( d0, d2 )
  // h1 = max( d1, d3 )
  // l0 < h0
  // l1 < h1


  // ==============================================================
  // level#2
  // ==============================================================
  // l0 = min( d0, d2 )
  // l1 = min( d1, d3 )
  // h0 = max( d0, d2 )
  // h1 = max( d1, d3 )
  // l0 < h0
  // l1 < h1
  // ==============================================================
  // l0, l1, h0, h1
  lh03.v    = _mm256_blend_pd( l03.v, h03.v, 12 );
  // l1, l0, h1, h0
  d_perm.v  = _mm256_permute_pd( lh03.v, 0x5 );
  // l0 > l1, l1 > l0, h0 > h1, h1 > h0
  rank.v    = _mm256_cmp_pd( lh03.v, d_perm.v, 30 );

  // min( l0, l1 ), min( l0, l1 ), min( h0, h1 ), min( h0, h1 )
  l03.v     = _mm256_blendv_pd( lh03.v, d_perm.v, rank.v );
  // max( l0, l1 ), max( l0, l1 ), max( h0, h1 ), max( h0, h1 )
  h03.v     = _mm256_blendv_pd( d_perm.v, lh03.v, rank.v );


  printf( "lh03 = %5.2lf, %5.2lf, %5.2lf, %5.2lf\n",
      lh03.d[ 0 ], lh03.d[ 1 ], lh03.d[ 2 ], lh03.d[ 3 ] );


  // ==============================================================
  // level#3
  // ==============================================================
  // l0 = min( l0, l1 ) = min( d0, d1, d2, d3 )
  // l2 = min( h0, h1 ) = min( max( d0, d2 ), max( d1, d3 ) )
  // h0 = max( l0, l1 ) = max( min( d0, d2 ), min( d1, d3 ) ) 
  // h2 = max( h0, h1 ) = max( d0, d1, d2, d3 )

  











  //printf( "r03 = %d, %d, %d, %d\n",
  //    r03.d[ 0 ], r03.d[ 1 ], r03.d[ 2 ], r03.d[ 3 ] );


  // permute d0, d1, i0, i1 and d2, d3, i2, i3 if nessary.
  d03.v     = _mm256_permutevar_pd( d03.v, _mm256_castpd_si256( rank.v ) );
  rank.v    = _mm256_and_pd( rank.v, one.v );
  r03.v     = _mm256_cvtpd_epi32( rank.v );
  //i03.v     = _mm_castps_si128( _mm_permutevar_ps( _mm_castsi128_ps( i03.v ), r03.v ) );


  printf( "d0123 = %5.2lf, %5.2lf, %5.2lf, %5.2lf\n",
      d03.d[ 0 ], d03.d[ 1 ], d03.d[ 2 ], d03.d[ 3 ] );








  printf( "r03 = %d, %d, %d, %d\n",
      r03.d[ 0 ], r03.d[ 1 ], r03.d[ 2 ], r03.d[ 3 ] );
  printf( "i0123 = %d, %d, %d, %d\n",
      i03.d[ 0 ], i03.d[ 1 ], i03.d[ 2 ], i03.d[ 3 ] );











//  // c0, c1, c2, c3
//  c03.v     = _mm256_load_pd( c );
//  one.v     = _mm256_broadcast_sd( &done );
//
//  printf( "c0123 = %5.2lf, %5.2lf, %5.2lf, %5.2lf\n",
//      c03.d[ 0 ], c03.d[ 1 ], c03.d[ 2 ], c03.d[ 3 ] );
//
//  // c1, c0, c3, c2
//  c_perm.v  = _mm256_permute_pd( c03.v, 0x5 );
//  rank.v    = _mm256_cmp_pd( c03.v, c_perm.v, 30 );
//  rank.v    = _mm256_and_pd( rank.v, one.v );
//  r03.v     = _mm256_cvtpd_epi32( rank.v );
//
//  printf( "c1032 = %5.2lf, %5.2lf, %5.2lf, %5.2lf\n",
//      c_perm.d[ 0 ], c_perm.d[ 1 ], c_perm.d[ 2 ], c_perm.d[ 3 ] );
//  printf( "rank  = %d, %d, %d, %d\n", r03.d[ 0 ], r03.d[ 1 ], r03.d[ 2 ], r03.d[ 3 ] );
//
//  // c3, c2, c1, c0
//  c_perm.v  = _mm256_permute2f128_pd( c_perm.v, c_perm.v, 0x1 );
//  rank.v    = _mm256_cmp_pd( c03.v, c_perm.v, 30 );
//  rank.v    = _mm256_and_pd( rank.v, one.v );
//  r_tmp.v   = _mm256_cvtpd_epi32( rank.v );
//  r03.v     = _mm_add_epi32( r03.v, r_tmp.v );
//
//  printf( "c3210 = %5.2lf, %5.2lf, %5.2lf, %5.2lf\n",
//      c_perm.d[ 0 ], c_perm.d[ 1 ], c_perm.d[ 2 ], c_perm.d[ 3 ] );
//  printf( "rank  = %d, %d, %d, %d\n", r03.d[ 0 ], r03.d[ 1 ], r03.d[ 2 ], r03.d[ 3 ] );
//
//
//  // c2, c3, c0, c1
//  c_perm.v  = _mm256_permute_pd( c_perm.v, 0x5 );
//  rank.v    = _mm256_cmp_pd( c03.v, c_perm.v, 30 );
//  rank.v    = _mm256_and_pd( rank.v, one.v );
//  r_tmp.v   = _mm256_cvtpd_epi32( rank.v );
//  r03.v     = _mm_add_epi32( r03.v, r_tmp.v );
//
//  printf( "c2301 = %5.2lf, %5.2lf, %5.2lf, %5.2lf\n",
//      c_perm.d[ 0 ], c_perm.d[ 1 ], c_perm.d[ 2 ], c_perm.d[ 3 ] );
//  printf( "rank  = %d, %d, %d, %d\n", r03.d[ 0 ], r03.d[ 1 ], r03.d[ 2 ], r03.d[ 3 ] );



}
