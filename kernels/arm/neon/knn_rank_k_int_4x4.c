#include <math.h>
#include <gsknn.h>

#include <arm_neon.h>


void knn_rank_k_int_s4x4(
    int    k,
    float  *a,
    float  *b,
    float  *c,
    int    ldc,
    aux_t  *aux
    )
{
  int    i, j, p, ldr;

  float32x4_t av1, av2, av3, av4;
  float32x4_t bv1, bv2, bv3, bv4;
  float32x4_t cv0, cv1, cv2, cv3;

  int    k_iter = k / 4;
  int    k_left = k % 4;

  if ( !aux->pc ) {
    cv0 = vmovq_n_f32( 0.0 );  
    cv1 = vmovq_n_f32( 0.0 );  
    cv2 = vmovq_n_f32( 0.0 );  
    cv3 = vmovq_n_f32( 0.0 );  
  }
  else {
    cv0 = vld1q_f32( c +  0 );
    cv1 = vld1q_f32( c +  4 );
    cv2 = vld1q_f32( c +  8 );
    cv3 = vld1q_f32( c + 12 );
  }

  //printf( "a: %f %f %f %f\n ", a[ 0 ], a[ 1 ], a[ 2 ], a[ 3 ] );
  //printf( "b: %f %f %f %f\n ", b[ 0 ], b[ 1 ], b[ 2 ], b[ 3 ] );



  // Rank-k update. Manually unlooped by 4.
  for ( p = 0; p < k_iter; p ++ ) {

    av1 = vld1q_f32( a ); 

    __builtin_prefetch( a + 224 );
    __builtin_prefetch( b + 224 );

    bv1 = vld1q_f32( b );

    cv0 = vmlaq_lane_f32( cv0, bv1, vget_low_f32(av1), 0 );
    cv1 = vmlaq_lane_f32( cv1, bv1, vget_low_f32(av1), 1 );
    cv2 = vmlaq_lane_f32( cv2, bv1, vget_high_f32(av1), 0 );
    cv3 = vmlaq_lane_f32( cv3, bv1, vget_high_f32(av1), 1 );

    av2 = vld1q_f32( a + 4 ); 
    bv2 = vld1q_f32( b + 4 );

    cv0 = vmlaq_lane_f32( cv0, bv2, vget_low_f32(av2), 0 );
    cv1 = vmlaq_lane_f32( cv1, bv2, vget_low_f32(av2), 1 );
    cv2 = vmlaq_lane_f32( cv2, bv2, vget_high_f32(av2), 0 );
    cv3 = vmlaq_lane_f32( cv3, bv2, vget_high_f32(av2), 1 );

    av3 = vld1q_f32( a + 8 ); 
    bv3 = vld1q_f32( b + 8 );

    cv0 = vmlaq_lane_f32( cv0, bv3, vget_low_f32(av3), 0 );
    cv1 = vmlaq_lane_f32( cv1, bv3, vget_low_f32(av3), 1 );
    cv2 = vmlaq_lane_f32( cv2, bv3, vget_high_f32(av3), 0 );
    cv3 = vmlaq_lane_f32( cv3, bv3, vget_high_f32(av3), 1 );

    av4 = vld1q_f32( a + 12 ); 
    bv4 = vld1q_f32( b + 12 );

    cv0 = vmlaq_lane_f32( cv0, bv4, vget_low_f32(av4), 0 );
    cv1 = vmlaq_lane_f32( cv1, bv4, vget_low_f32(av4), 1 );
    cv2 = vmlaq_lane_f32( cv2, bv4, vget_high_f32(av4), 0 );
    cv3 = vmlaq_lane_f32( cv3, bv4, vget_high_f32(av4), 1 );

	a += 16;
	b += 16;
  }

  // Tail case.
  for ( p = 0; p < k_left; p ++ ) {

    av1 = vld1q_f32( a ); 

    __builtin_prefetch( a + 112 );
    __builtin_prefetch( b + 112 );

    bv1 = vld1q_f32( b );

    cv0 = vmlaq_lane_f32( cv0, bv1, vget_low_f32(av1), 0 );
    cv1 = vmlaq_lane_f32( cv1, bv1, vget_low_f32(av1), 1 );
    cv2 = vmlaq_lane_f32( cv2, bv1, vget_high_f32(av1), 0 );
    cv3 = vmlaq_lane_f32( cv3, bv1, vget_high_f32(av1), 1 );

	a += 4;
	b += 4;
  }

  __builtin_prefetch( aux->b_next_s );

  //printf( "c: %f %f %f %f\n ", c[ 0 ], c[ 1 ], c[ 2 ], c[ 3 ] );
  //printf( "c: %f %f %f %f\n ", c[ 4 ], c[ 5 ], c[ 6 ], c[ 7 ] );
  //printf( "c: %f %f %f %f\n ", c[ 8 ], c[ 9 ], c[ 10 ], c[ 11 ] );
  //printf( "c: %f %f %f %f\n ", c[ 12 ], c[ 13 ], c[ 14 ], c[ 15 ] );

  vst1q_f32( c +  0, cv0 );
  vst1q_f32( c +  4, cv1 );
  vst1q_f32( c +  8, cv2 );
  vst1q_f32( c + 12, cv3 );
}
