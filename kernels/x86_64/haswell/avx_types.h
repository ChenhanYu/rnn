#ifndef __AVX_TYPES_H__
#define __AVX_TYPES_H__

typedef union {
  __m256  v;
  __m256i u;
  float  s[ 8 ];
} v8sf_t;

typedef union {
  __m256d v;
  __m256i u;
  double d[ 4 ];
} v4df_t;


typedef union {
  __m128i v;
  int d[ 4 ];
} v4li_t;

#endif // define __AVX_TYPES_H__
