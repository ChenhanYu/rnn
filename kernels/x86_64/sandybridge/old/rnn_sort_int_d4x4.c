#include <immintrin.h> // AVX
#include <rnn.h>

/*
 * Inputs: 
 *       ymm0 = m i e a (keys) 
 *       ymm1 = n j f b (keys) 
 *       ymm2 = o k g c (keys) 
 *       ymm3 = p l h d (keys) 
 * 
 *       ymm4 = M I E A (pointers) 
 *       ymm5 = N J F B (pointers) 
 *       ymm6 = O K G C (pointers) 
 *       ymm7 = P L H D (pointers) 
 * 
 * Outputs: 
 *       ymm0 = d c b a (sorted keys) 
 *       ymm1 = h g f e (sorted keys) 
 *       ymm2 = l k j i (sorted keys) 
 *       ymm3 = p o n m (sorted keys) 
 * 
 *       ymm4 = D C B A (sorted pointers) 
 *       ymm5 = H G F E (sorted pointers) 
 *       ymm6 = L K J I (sorted pointers) 
 *       ymm7 = P O N M (sorted pointers)
 */


void rnn_sort_int_d4x4(
    double *i,
    double *d
    )
{
  v4df_t  y0,  y1,  y2,  y3;
  v4df_t  y4,  y5,  y6,  y7;
  v4df_t  y8,  y9, y10, y11;
  v4df_t y12, y13, y14, y15;

  y0.v     = _mm256_load_pd( d      );
  y1.v     = _mm256_load_pd( d +  4 );
  y2.v     = _mm256_load_pd( d +  8 );
  y3.v     = _mm256_load_pd( d + 12 );

  y4.v     = _mm256_load_pd( i      );
  y5.v     = _mm256_load_pd( i +  4 );
  y6.v     = _mm256_load_pd( i +  8 );
  y7.v     = _mm256_load_pd( i + 12 );


  y15.v    = _mm256_cmp_pd( y0.v, y2.v, 30 );        // y15 = mask( a > c ), clocks = 1     
  y14.v    = _mm256_cmp_pd( y1.v, y3.v, 30 );        // y14 = mask( b > d ), clocks = 2

  // comp( a, c )
  y8.v     = _mm256_blendv_pd( y0.v, y2.v, y15.v );  // y8 = a,              clocks = 4
  y0.v     = _mm256_blendv_pd( y2.v, y0.v, y15.v );  // y0 = c,              clocks = 5
  // comp( A, C )
  y2.v     = _mm256_blendv_pd( y4.v, y6.v, y15.v );  // y2 = A,              clocks = 6
  y4.v     = _mm256_blendv_pd( y6.v, y4.v, y15.v );  // y4 = C,              clocks = 7


  // comp( b, d )
  y6.v     = _mm256_blendv_pd( y1.v, y3.v, y14.v );  // y6 = b,              clocks = 8
  y1.v     = _mm256_blendv_pd( y3.v, y1.v, y14.v );  // y1 = d,              clocks = 9
  y15.v    = _mm256_cmp_pd( y8.v, y6.v, 30 );        // y15 = mask( a > b ), ??
  // comp( B, D )
  y3.v     = _mm256_blendv_pd( y5.v, y7.v, y14.v );  // y3 = B               clocks = 10
  y14.v    = _mm256_cmp_pd( y0.v, y1.v, 30 );        // y14 = mask( c > d ), ??
  y5.v     = _mm256_blendv_pd( y7.v, y5.v, y14.v );  // y5 = D               clocks = 11


  // comp( a, b )
  y7.v     = _mm256_blendv_pd( y8.v, y6.v, y15.v );  // y7 = a,              clocks = 12
  y8.v     = _mm256_blendv_pd( y6.v, y8.v, y15.v );  // y8 = b,              clocks = 13
  // comp( A, B )
  y6.v     = _mm256_blendv_pd( y2.v, y3.v, y15.v );  // y6 = A,              clocks = 14
  y2.v     = _mm256_blendv_pd( y3.v, y2.v, y15.v );  // y2 = B,              clocks = 15


  // comp( c, d )
  y3.v     = _mm256_blendv_pd( y0.v, y1.v, y14.v );  // y3 = c,              clocks = 16
  y0.v     = _mm256_blendv_pd( y1.v, y0.v, y14.v );  // y0 = d,              clocks = 17
  y15.v    = _mm256_cmp_pd( y8.v, y3.v, 30 );        // y15 = mask( b > c ), ??
  // comp( C, D )
  y1.v     = _mm256_blendv_pd( y4.v, y5.v, y14.v );  // y1 = C,              clocks = 18
  y4.v     = _mm256_blendv_pd( y5.v, y4.v, y14.v );  // y4 = D,              clocks = 19

  // comp( b, c )
  y5.v     = _mm256_blendv_pd( y8.v, y3.v, y15.v );  // y5 = b,              clocks = 20
  y8.v     = _mm256_blendv_pd( y3.v, y8.v, y15.v );  // y8 = c,              clocks = 21
  // comp( B, C )
  y3.v     = _mm256_blendv_pd( y2.v, y1.v, y15.v );  // y3 = B,              clocks = 22
  y2.v     = _mm256_blendv_pd( y1.v, y2.v, y15.v );  // y2 = C,              clocks = 23


  /*
   * ymm7 = m i e a (keys)
   * ymm5 = n j f b (keys)
   * ymm8 = o k g c (keys)
   * ymm0 = p l h d (keys)
   * 
   * ymm6 = M I E A (pointers)
   * ymm3 = N J F B (pointers)
   * ymm2 = O K G C (pointers)
   * ymm4 = P L H D (pointers)
   *
   */

  y15.v    = _mm256_permute2f128_pd( y7.v, y8.v, 0x20 );
  y14.v    = _mm256_permute2f128_pd( y5.v, y0.v, 0x20 );
  y13.v    = _mm256_permute2f128_pd( y7.v, y8.v, 0x31 );
  y12.v    = _mm256_permute2f128_pd( y5.v, y0.v, 0x31 );

  y11.v    = _mm256_permute2f128_pd( y6.v, y2.v, 0x20 );
  y10.v    = _mm256_permute2f128_pd( y3.v, y4.v, 0x20 );
  y9.v     = _mm256_permute2f128_pd( y6.v, y2.v, 0x31 );
  y8.v     = _mm256_permute2f128_pd( y3.v, y4.v, 0x31 );

  y0.v     = _mm256_unpacklo_pd( y15.v, y14.v );
  y1.v     = _mm256_unpackhi_pd( y15.v, y14.v );
  y2.v     = _mm256_unpacklo_pd( y13.v, y12.v );
  y3.v     = _mm256_unpackhi_pd( y13.v, y12.v );

  y4.v     = _mm256_unpacklo_pd( y11.v, y10.v );
  y5.v     = _mm256_unpackhi_pd( y11.v, y10.v );
  y6.v     = _mm256_unpacklo_pd( y9.v, y8.v );
  y7.v     = _mm256_unpackhi_pd( y9.v, y8.v );


  _mm256_store_pd( d     , y0.v );
  _mm256_store_pd( d +  4, y1.v );
  _mm256_store_pd( d +  8, y2.v );
  _mm256_store_pd( d + 12, y3.v );

  _mm256_store_pd( i     , y4.v );
  _mm256_store_pd( i +  4, y5.v );
  _mm256_store_pd( i +  8, y6.v );
  _mm256_store_pd( i + 12, y7.v );

}
