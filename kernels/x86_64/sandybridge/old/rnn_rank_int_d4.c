#include <immintrin.h> // AVX
#include <rnn.h>

void rnn_rank_int_d4(
    double *c
    )
{
  double done = 1.0;
  v4df_t c03, c_perm, rank, one;
  v4li_t r03, r_tmp; 
  
  // c0, c1, c2, c3
  c03.v     = _mm256_load_pd( c );
  one.v     = _mm256_broadcast_sd( &done );

  printf( "c0123 = %5.2lf, %5.2lf, %5.2lf, %5.2lf\n",
      c03.d[ 0 ], c03.d[ 1 ], c03.d[ 2 ], c03.d[ 3 ] );

  // c1, c0, c3, c2
  c_perm.v  = _mm256_permute_pd( c03.v, 0x5 );
  rank.v    = _mm256_cmp_pd( c03.v, c_perm.v, 30 );
  rank.v    = _mm256_and_pd( rank.v, one.v );
  r03.v     = _mm256_cvtpd_epi32( rank.v );

  printf( "c1032 = %5.2lf, %5.2lf, %5.2lf, %5.2lf\n",
      c_perm.d[ 0 ], c_perm.d[ 1 ], c_perm.d[ 2 ], c_perm.d[ 3 ] );
  printf( "rank  = %d, %d, %d, %d\n", r03.d[ 0 ], r03.d[ 1 ], r03.d[ 2 ], r03.d[ 3 ] );

  // c3, c2, c1, c0
  c_perm.v  = _mm256_permute2f128_pd( c_perm.v, c_perm.v, 0x1 );
  rank.v    = _mm256_cmp_pd( c03.v, c_perm.v, 30 );
  rank.v    = _mm256_and_pd( rank.v, one.v );
  r_tmp.v   = _mm256_cvtpd_epi32( rank.v );
  r03.v     = _mm_add_epi32( r03.v, r_tmp.v );

  printf( "c3210 = %5.2lf, %5.2lf, %5.2lf, %5.2lf\n",
      c_perm.d[ 0 ], c_perm.d[ 1 ], c_perm.d[ 2 ], c_perm.d[ 3 ] );
  printf( "rank  = %d, %d, %d, %d\n", r03.d[ 0 ], r03.d[ 1 ], r03.d[ 2 ], r03.d[ 3 ] );


  // c2, c3, c0, c1
  c_perm.v  = _mm256_permute_pd( c_perm.v, 0x5 );
  rank.v    = _mm256_cmp_pd( c03.v, c_perm.v, 30 );
  rank.v    = _mm256_and_pd( rank.v, one.v );
  r_tmp.v   = _mm256_cvtpd_epi32( rank.v );
  r03.v     = _mm_add_epi32( r03.v, r_tmp.v );

  printf( "c2301 = %5.2lf, %5.2lf, %5.2lf, %5.2lf\n",
      c_perm.d[ 0 ], c_perm.d[ 1 ], c_perm.d[ 2 ], c_perm.d[ 3 ] );
  printf( "rank  = %d, %d, %d, %d\n", r03.d[ 0 ], r03.d[ 1 ], r03.d[ 2 ], r03.d[ 3 ] );



}
