#include <immintrin.h> // AVX
#include <rnn.h>

void rnn_merge_int_d4plus4(
	double *a,
	double *amap,
	double *b,
	double *bmap,
    double *d,
    double *i
    );

void rnn_merge_int_dn(
	double *a,
	double *amap,
	double *b,
	double *bmap,
    double *d,
    double *index,
	int    a_len,
	int    b_len
    )
{
  int     i = 0, j = 0, t = 0;
//  v4df_t  y0,  y1,  y2,  y3;
//  v4df_t  y4,  y5,  y6,  y7;
//  v4df_t  y15;

  if ( a[i] < b[j] ) {
	//Can we memcpy 4*sizeof(double)?
	for ( t = 0; t < 4; t++ ) {
	  d[t] = a[i];
	  index[t] = amap[i];
	  i ++;
	}
  } else {
	//Can we memcpy 4*sizeof(double)?
	for ( t = 0; t < 4; t++ ) {
	  d[t] = b[j];
	  index[t] = bmap[j];
	  j ++;
	}
  }
  
  t = 0;

  while ( i < a_len && j < b_len ) {
	if ( a[i] < b[j] ) {
	  //d[t] = a[i];
	  //index[t] = amap[i];
	  //i ++;
	  //t ++;
	  rnn_merge_int_d4plus4( d + t , index + t, a + i, amap + i, d + t, index + t );
	  i += 4;
	  t += 4;

	} else if ( a[i] >= b[j] ) {
	  //d[t] = b[j];
	  //index[t] = bmap[j];
	  //i ++;
	  //t ++;
	  rnn_merge_int_d4plus4( d + t , index + t, b + j, bmap + j, d + t , index + t );
	  j += 4;
	  t += 4;

	}
//	else {
//	  d[t] = a[i];
//	  i ++;
//	  t ++;
//	  d[t] = b[j];
//	  j ++;
//	  t ++;
//	}

  }

  while ( i < a_len ) {
	//d[ t ] = a[ i ];
	//i ++;
	//t ++;
	rnn_merge_int_d4plus4( d + t , index + t, a + i, amap + i, d + t, index + t );
	i += 4;
	t += 4;
  }

  while ( j < b_len ) {
	//d[ j ] = b[ j ];
	//j ++;
	//t ++;
	rnn_merge_int_d4plus4( d + t , index + t, b + j, bmap + j, d + t , index + t );
	j += 4;
	t += 4;
  }
}
