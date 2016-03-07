#include <stdio.h>
#include <omp.h>
#include <gsknn.h>
#include <gsknn_internal.h>
#define min( i, j ) ( (i)<(j) ? (i): (j) )

#include <gsknn_config.h>
#include <gsknn_kernel.h>

inline void packA0A1_kcxmc_d(
    int    m,
    int    k,
	int    d,
    double *XA0,
    int    *amap0,
    double *XA1,
    int    *amap1,
	double opA,
    double *packA
    )
{
  int    i, p;
  double *a_pntr0[ DKNN_MR ];
  double *a_pntr1[ DKNN_MR ];

  if ( amap1 ) { 
	for ( i = 0; i < m; i ++ ) {
	  a_pntr0[ i ] = XA0 + amap0[ i ] * d;
	  a_pntr1[ i ] = XA1 + amap1[ i ] * d;
	}

	for ( i = m; i < DKNN_MR; i ++ ) {
	  a_pntr0[ i ] = XA0 + amap0[ 0 ] * d;
	  a_pntr1[ i ] = XA1 + amap1[ 0 ] * d;
	}

	for ( p = 0; p < k; p ++ ) {
	  for ( i = 0; i < DKNN_MR; i ++ ) {
		*packA ++ = ( *a_pntr0[ i ] + opA * ( *a_pntr1[ i ] ) );
		a_pntr0[ i ] ++;
		a_pntr1[ i ] ++;
	  }
	}
  }
  else {
	for ( i = 0; i < m; i ++ ) {
	  a_pntr0[ i ] = XA0 + amap0[ i ] * d;
	}

	for ( i = m; i < DKNN_MR; i ++ ) {
	  a_pntr0[ i ] = XA0 + amap0[ 0 ] * d;
	}

	for ( p = 0; p < k; p ++ ) {
	  for ( i = 0; i < DKNN_MR; i ++ ) {
		*packA ++ = *a_pntr0[ i ] ++;
	  }
	}
  }
}

inline void packB0B1_kcxnc_d(
    int    n,
    int    k,
	int    d,
    double *XB0,
    int    *bmap0,
    double *XB1,
    int    *bmap1,
	double opB,
    double *packB
    )
{
  int    i, p;
  double *b_pntr0[ DKNN_NR ];
  double *b_pntr1[ DKNN_NR ];

  if ( bmap1 ) { 

	//printf( "packB chk0\n" );

	for ( i = 0; i < n; i ++ ) {
	  b_pntr0[ i ] = XB0 + bmap0[ i ] * d;
	  b_pntr1[ i ] = XB1 + bmap1[ i ] * d;
	}

	//printf( "packB chk1\n" );

	for ( i = n; i < DKNN_NR; i ++ ) {
	  b_pntr0[ i ] = XB0 + bmap0[ 0 ] * d;
	  b_pntr1[ i ] = XB1 + bmap1[ 0 ] * d;
	}

	//printf( "packB chk2\n" );

	for ( p = 0; p < k; p ++ ) {
	  for ( i = 0; i < DKNN_NR; i ++ ) {
		*packB ++ = ( *b_pntr0[ i ] + opB * ( *b_pntr1[ i ] ) );
		b_pntr0[ i ] ++;
		b_pntr1[ i ] ++;
	  }
	}
  }
  else {
	for ( i = 0; i < n; i ++ ) {
	  b_pntr0[ i ] = XB0 + bmap0[ i ] * d;
	}

	for ( i = n; i < DKNN_NR; i ++ ) {
	  b_pntr0[ i ] = XB0 + bmap0[ 0 ] * d;
	}

	for ( p = 0; p < k; p ++ ) {
	  for ( i = 0; i < DKNN_NR; i ++ ) {
		*packB ++ = *b_pntr0[ i ] ++;
	  }
	}
  }
}


void dstrassen_macro_kernel(
    int    m,
    int    n,
    int    k,
	double alpha0,
	double alpha1,
    double *packA,
    double *packB,
	double beta0,
    double *C0,
	double beta1,
	double *C1,
    int    ldc,
	int    pc
    )
{
  int    i, ii, j;
  double *c1ptr;
  aux_t  aux;
  
  aux.b_next = packB;
  aux.pc     = pc;

  //printf( "here, pc = %d, last = %d, ldc = %d, m = %d, n = %d, k %d\n", 
  //    pc, lastiter, ldc, m, n , k );

  for ( j = 0; j < n; j += DKNN_NR ) {
    aux.n  = min( n - j, DKNN_NR );
    for ( i = 0; i < m; i += DKNN_MR ) {
      aux.m = min( m - i, DKNN_MR );
      if ( i + DKNN_MR >= m ) {
        aux.b_next += DKNN_NR * k;
      }

	  if ( C1 ) c1ptr = C1 + j * ldc + i;
	  else      c1ptr = NULL;	  

      ( *strassen_d[ 0 ] ) (
		  k,
		  alpha0,
		  alpha1,
          &packA[ i * k ],
          &packB[ j * k ],
		  beta0,
          &C0[ j * ldc + i ],
		  beta1,
          c1ptr,
		  ldc,
		  &aux
		  );
    }
  }
}








void dstrassen(
	int    m,
	int    n,
	int    k,
	int    d,      // This is ldA and ldB.
	double alpha0,
	double alpha1,
	double *A0,
	int    *amap0,
	double *A1,
	int    *amap1,
	double opA,
	double *B0,
	int    *bmap0,
	double *B1,
	int    *bmap1,
    double opB,
	double beta0,
	double *C0,
	double beta1,
	double *C1,
	int    ldc,
	int    dopackC
	)
{
  int    i, j, p, gsknn_ic_nt;
  int    ic, ib, jc, jb, pc, pb;
  double *packA, *packB;
  char   *str;

  // Early return if possible
  if ( m == 0 || n == 0 || k == 0 ) {
    printf( "dstrassen(): early return\n" );
    return;
  }


  // sequential is the default situation
  gsknn_ic_nt = 1;


  // check the environment variable
  str = getenv( "GSKNN_IC_NT" );
  if ( str != NULL ) {
    gsknn_ic_nt = (int)strtol( str, NULL, 10 );
  }


  // Allocate packing buffers
  packA  = gsknn_malloc_aligned( DKNN_KC, ( DKNN_MC + 1 ) * gsknn_ic_nt, sizeof(double) );
  packB  = gsknn_malloc_aligned( DKNN_KC, ( DKNN_NC + 1 )              , sizeof(double) );

  for ( jc = 0; jc < n; jc += DKNN_NC ) {                  // 6-th loop
    jb = min( n - jc, DKNN_NC );
    for ( pc = 0; pc < k; pc += DKNN_KC ) {                // 5-th loop
      pb = min( k - pc, DKNN_KC );


      #pragma omp parallel for num_threads( gsknn_ic_nt )
      for ( j = 0; j < jb; j += DKNN_NR ) {
		int    *bmapptr;
		if ( bmap1 ) bmapptr = bmap1 + jc + j;
		else         bmapptr = NULL;
        packB0B1_kcxnc_d(
            min( jb - j, DKNN_NR ),
            pb,
			d,
            &B0[ pc ],
            &bmap0[ jc + j ],
            &B1[ pc ],
            bmapptr,
		    opB,
            &packB[ j * pb ]
            );
      }

      //printf( "packB0B1\n" );

      #pragma omp parallel for num_threads( gsknn_ic_nt ) private( ic, ib, i )
      for ( ic = 0; ic < m; ic += DKNN_MC ) {              // 4-th loop
        int    tid = omp_get_thread_num();
		double *c0ptr, *c1ptr;

        ib = min( m - ic, DKNN_MC );
        for ( i = 0; i < ib; i += DKNN_MR ) {
		int    *amapptr;
		if ( amap1 ) amapptr = amap1 + ic + i;
		else         amapptr = NULL;
		  packA0A1_kcxmc_d(
			  min( ib - i, DKNN_MR ),
			  pb,
			  d,
			  &A0[ pc ],
			  &amap0[ ic + i ],
			  &A1[ pc ],
			  amapptr,
			  opA,
			  &packA[ tid * DKNN_MC * pb + i * pb ]
			  );
		}

        //printf( "packA0A1\n" );

		// Macro-Kernel Here
		if ( dopackC ) {
		  c0ptr = C0 + jc * ldc + ic;
		}
		else {
		  c0ptr = C0 + jc * ldc + ic;
		  if ( C1 ) c1ptr = C1 + jc * ldc + ic;
		  else      c1ptr = NULL;
		}

		dstrassen_macro_kernel(
			ib,
			jb,
			pb,
			alpha0, alpha1,
			packA  + tid * DKNN_MC * pb,
			packB,
			beta0, c0ptr,
			beta1, c1ptr,
			ldc,
			pc
			);
	  }                                                    // End 4.th loop
	}                                                      // End 5.th loop
  }                                                        // End 6.th loop

  free( packA );
  free( packB );
}

void dstrrk(
	int    m,
	int    n,
	int    k,
	int    d,
	double *A,
	int    *amap,
	double *B,
	int    *bmap,
	double *C,
	int    ldc
	)
{
  double *A00, *A01, *A10, *A11;
  double *B00, *B01, *B10, *B11;
  double *C00, *C01, *C10, *C11;
  int    *amap0, *amap1;
  int    *bmap0, *bmap1;

  A00 = A;
  A01 = A + ( k / 2 );
  A10 = A;
  A11 = A + ( k / 2 );

  B00 = B;
  B01 = B;
  B10 = B + ( k / 2 );
  B11 = B + ( k / 2 );
  
  C00 = C;
  C01 = C + ( n / 2 ) * ldc;
  C10 = C + ( m / 2 );
  C11 = C + ( n / 2 ) * ldc + ( m / 2 );

  amap0 = amap;
  amap1 = amap + ( m / 2 );
  bmap0 = bmap;
  bmap1 = bmap + ( n / 2 );

  //printf( "before M1\n");

  // M1
  // C00 = 0 * C00 + 1 * ( A00 + A11 )( B00 + B11 )
  // C11 = 0 * C11 + 1 * ( A00 + A11 )( B00 + B11 )
  dstrassen( 
	  m / 2, n / 2, k / 2, d,
      1.0, 1.0,
	  A00, amap0, A11, amap1, 1.0,
	  B00, bmap0, B11, bmap1, 1.0,
      0.0, C00, 
	  0.0, C11, ldc, 0
	  );

  //printf( "M1\n");

  // M2
  // C10 = 0 * C10 + 1 * ( A10 + A11 )( B00 )
  // C11 = 1 * C11 + 1 * ( A10 + A11 )( B00 )
  dstrassen( 
	  m / 2, n / 2, k / 2, d,
      1.0, -1.0,
	  A10, amap1, A11, amap1, 1.0,
	  B00, bmap0, NULL, NULL, 0.0,
      0.0, C10, 
	  1.0, C11, ldc, 0
	  );

  //printf( "M2\n");

  // M3
  // C01 = 0 * C01 + 1 * ( A00 )( B01 - B11 )
  // C11 = 1 * C11 + 1 * ( A00 )( B01 - B11 )
  dstrassen( 
	  m / 2, n / 2, k / 2, d,
      1.0, 1.0,
	  A00, amap0, NULL, NULL, 0.0,
	  B01, bmap1, B11, bmap1, -1.0,
      0.0, C01, 
	  1.0, C11, ldc, 0
	  );

  //printf( "M3\n");

  // M4
  // C00 = 1 * C00 + 1 * ( A11 )( B10 - B00 )
  // C10 = 1 * C10 + 1 * ( A11 )( B10 - B00 )
  dstrassen( 
	  m / 2, n / 2, k / 2, d,
      1.0, 1.0, 
	  A11, amap1, NULL, NULL, 0.0,
	  B10, bmap0, B00, bmap0, -1.0,
      1.0, C00, 
	  1.0, C10, ldc, 0
	  );

  //printf( "M4\n");

  // M5
  // C00 = 1 * C00 - 1 * ( A00 + A01 )( B11 )
  // C01 = 1 * C01 + 1 * ( A00 + A01 )( B11 )
  dstrassen( 
	  m / 2, n / 2, k / 2, d,
      -1.0, 1.0,
	  A00, amap0, A01, amap0, 1.0,
	  B11, bmap1, NULL, NULL, 0.0,
      1.0, C00, 
	  1.0, C01, ldc, 0
	  );

  //printf( "M5\n");

  // M6
  // C11 = 1 * C11 + ( A10 - A00 )( B00 + B01 )
  dstrassen( 
	  m / 2, n / 2, k / 2, d,
      1.0, 0.0,
	  A10, amap1, A00, amap0, -1.0,
	  B00, bmap0, B01, bmap1, 1.0,
      1.0, C11, 
	  0.0, NULL, ldc, 0
	  );

  //printf( "M6\n");
  
  // M7
  // C00 = 1 * C00 + ( A01 - A11 )( B10 + B11 )
  dstrassen( 
	  m / 2, n / 2, k / 2, d,
      1.0, 0.0,
	  A01, amap0, A11, amap1, -1.0,
	  B10, bmap0, B11, bmap1, 1.0,
      1.0, C00, 
	  0.0, NULL, ldc, 0
	  );

  //printf( "M7\n");

  // Edge case here.
}


void dstrrk_packC(
	int    m,
	int    n,
	int    k,
	int    d,
	double *A,
	int    *amap,
	double *B,
	int    *bmap,
	double *C,
	int    ldc
	)
{
  double *A00, *A01, *A10, *A11;
  double *B00, *B01, *B10, *B11;
  double *C00, *C01, *C10, *C11;
  int    *amap0, *amap1;
  int    *bmap0, *bmap1;
  int    ldpackC, npackC;

  A00 = A;
  A01 = A + ( k / 2 );
  A10 = A;
  A11 = A + ( k / 2 );

  B00 = B;
  B01 = B;
  B10 = B + ( k / 2 );
  B11 = B + ( k / 2 );
  
  npackC  = ( n / ( 2 * DKNN_NR ) + 1 ) * DKNN_NR;
  ldpackC = ( m / ( 2 * DKNN_MR ) + 1 ) * DKNN_MR;

  C00 = gsknn_malloc_aligned( ldpackC, npackC, sizeof(double) );
  C01 = gsknn_malloc_aligned( ldpackC, npackC, sizeof(double) );
  C10 = gsknn_malloc_aligned( ldpackC, npackC, sizeof(double) );
  C11 = gsknn_malloc_aligned( ldpackC, npackC, sizeof(double) );

  amap0 = amap;
  amap1 = amap + ( m / 2 );
  bmap0 = bmap;
  bmap1 = bmap + ( n / 2 );

  //printf( "before M1\n");

  // M1
  // C00 = 0 * C00 + 1 * ( A00 + A11 )( B00 + B11 )
  // C11 = 0 * C11 + 1 * ( A00 + A11 )( B00 + B11 )
  dstrassen( 
	  m / 2, n / 2, k / 2, d,
      1.0, 1.0,
	  A00, amap0, A11, amap1, 1.0,
	  B00, bmap0, B11, bmap1, 1.0,
      0.0, C00, 
	  0.0, C11, ldpackC, 1
	  );

  //printf( "M1\n");

  // M2
  // C10 = 0 * C10 + 1 * ( A10 + A11 )( B00 )
  // C11 = 1 * C11 + 1 * ( A10 + A11 )( B00 )
  dstrassen( 
	  m / 2, n / 2, k / 2, d,
      1.0, -1.0,
	  A10, amap1, A11, amap1, 1.0,
	  B00, bmap0, NULL, NULL, 0.0,
      0.0, C10, 
	  1.0, C11, ldpackC, 1
	  );

  //printf( "M2\n");

  // M3
  // C01 = 0 * C01 + 1 * ( A00 )( B01 - B11 )
  // C11 = 1 * C11 + 1 * ( A00 )( B01 - B11 )
  dstrassen( 
	  m / 2, n / 2, k / 2, d,
      1.0, 1.0,
	  A00, amap0, NULL, NULL, 0.0,
	  B01, bmap1, B11, bmap1, -1.0,
      0.0, C01, 
	  1.0, C11, ldpackC, 1
	  );

  //printf( "M3\n");

  // M4
  // C00 = 1 * C00 + 1 * ( A11 )( B10 - B00 )
  // C10 = 1 * C10 + 1 * ( A11 )( B10 - B00 )
  dstrassen( 
	  m / 2, n / 2, k / 2, d,
      1.0, 1.0, 
	  A11, amap1, NULL, NULL, 0.0,
	  B10, bmap0, B00, bmap0, -1.0,
      1.0, C00, 
	  1.0, C10, ldpackC, 1
	  );

  //printf( "M4\n");

  // M5
  // C00 = 1 * C00 - 1 * ( A00 + A01 )( B11 )
  // C01 = 1 * C01 + 1 * ( A00 + A01 )( B11 )
  dstrassen( 
	  m / 2, n / 2, k / 2, d,
      -1.0, 1.0,
	  A00, amap0, A01, amap0, 1.0,
	  B11, bmap1, NULL, NULL, 0.0,
      1.0, C00, 
	  1.0, C01, ldpackC, 1
	  );

  //printf( "M5\n");

  // M6
  // C11 = 1 * C11 + ( A10 - A00 )( B00 + B01 )
  dstrassen( 
	  m / 2, n / 2, k / 2, d,
      1.0, 0.0,
	  A10, amap1, A00, amap0, -1.0,
	  B00, bmap0, B01, bmap1, 1.0,
      1.0, C11, 
	  0.0, NULL, ldpackC, 1
	  );

  //printf( "M6\n");
  
  // M7
  // C00 = 1 * C00 + ( A01 - A11 )( B10 + B11 )
  dstrassen( 
	  m / 2, n / 2, k / 2, d,
      1.0, 0.0,
	  A01, amap0, A11, amap1, -1.0,
	  B10, bmap0, B11, bmap1, 1.0,
      1.0, C00, 
	  0.0, NULL, ldpackC, 1
	  );

  //printf( "M7\n");

  // Edge case here.

  free( C00 );
  free( C01 );
  free( C10 );
  free( C11 );
}
