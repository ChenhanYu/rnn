  int k_iter = k / 2;
  int k_left = k % 2;

  __asm__ volatile( "prefetcht0 0(%0)    \n\t" : :"r"( a ) );
  __asm__ volatile( "prefetcht2 0(%0)    \n\t" : :"r"( aux->b_next ) );
  __asm__ volatile( "prefetcht0 192(%0)  \n\t" : :"r"( c0 ) );
  if ( c1 ) {
    __asm__ volatile( "prefetcht0 192(%0)  \n\t" : :"r"( c1 ) );
  }

  c03_0.v = _mm256_setzero_pd();
  c03_1.v = _mm256_setzero_pd();
  c03_2.v = _mm256_setzero_pd();
  c03_3.v = _mm256_setzero_pd();
  c03_4.v = _mm256_setzero_pd();
  c03_5.v = _mm256_setzero_pd();

  c47_0.v = _mm256_setzero_pd();
  c47_1.v = _mm256_setzero_pd();
  c47_2.v = _mm256_setzero_pd();
  c47_3.v = _mm256_setzero_pd();
  c47_4.v = _mm256_setzero_pd();
  c47_5.v = _mm256_setzero_pd();

  // Load a03, a47, b0
  a03.v = _mm256_load_pd( (double*)  a        );
  a47.v = _mm256_load_pd( (double*)( a +  4 ) );

  for ( i = 0; i < k_iter; ++ i ) {

	// Iteration #0
    __asm__ volatile( "prefetcht0 192(%0)    \n\t" : :"r"(a) );

    b0.v    = _mm256_broadcast_sd( b      );
    b1.v    = _mm256_broadcast_sd( b +  1 );
    c03_0.v = _mm256_fmadd_pd( a03.v, b0.v, c03_0.v );
    c47_0.v = _mm256_fmadd_pd( a47.v, b0.v, c47_0.v );
    c03_1.v = _mm256_fmadd_pd( a03.v, b1.v, c03_1.v );
    c47_1.v = _mm256_fmadd_pd( a47.v, b1.v, c47_1.v );

    b0.v    = _mm256_broadcast_sd( b +  2 );
    b1.v    = _mm256_broadcast_sd( b +  3 );
    c03_2.v = _mm256_fmadd_pd( a03.v, b0.v, c03_2.v );
    c47_2.v = _mm256_fmadd_pd( a47.v, b0.v, c47_2.v );
    c03_3.v = _mm256_fmadd_pd( a03.v, b1.v, c03_3.v );
    c47_3.v = _mm256_fmadd_pd( a47.v, b1.v, c47_3.v );

    b0.v    = _mm256_broadcast_sd( b +  4 );
    b1.v    = _mm256_broadcast_sd( b +  5 );
    c03_4.v = _mm256_fmadd_pd( a03.v, b0.v, c03_4.v );
    c47_4.v = _mm256_fmadd_pd( a47.v, b0.v, c47_4.v );
    c03_5.v = _mm256_fmadd_pd( a03.v, b1.v, c03_5.v );
    c47_5.v = _mm256_fmadd_pd( a47.v, b1.v, c47_5.v );

	a03.v = _mm256_load_pd( (double*)( a +  8 ) );
	a47.v = _mm256_load_pd( (double*)( a + 12 ) );

	// Iteration #1
	__asm__ volatile( "prefetcht0 512(%0)    \n\t" : :"r"(a) );

    b0.v    = _mm256_broadcast_sd( b +  6 );
    b1.v    = _mm256_broadcast_sd( b +  7 );
    c03_0.v = _mm256_fmadd_pd( a03.v, b0.v, c03_0.v );
    c47_0.v = _mm256_fmadd_pd( a47.v, b0.v, c47_0.v );
    c03_1.v = _mm256_fmadd_pd( a03.v, b1.v, c03_1.v );
    c47_1.v = _mm256_fmadd_pd( a47.v, b1.v, c47_1.v );

    b0.v    = _mm256_broadcast_sd( b +  8 );
    b1.v    = _mm256_broadcast_sd( b +  9 );
    c03_2.v = _mm256_fmadd_pd( a03.v, b0.v, c03_2.v );
    c47_2.v = _mm256_fmadd_pd( a47.v, b0.v, c47_2.v );
    c03_3.v = _mm256_fmadd_pd( a03.v, b1.v, c03_3.v );
    c47_3.v = _mm256_fmadd_pd( a47.v, b1.v, c47_3.v );

    b0.v    = _mm256_broadcast_sd( b + 10 );
    b1.v    = _mm256_broadcast_sd( b + 11 );
    c03_4.v = _mm256_fmadd_pd( a03.v, b0.v, c03_4.v );
    c47_4.v = _mm256_fmadd_pd( a47.v, b0.v, c47_4.v );
    c03_5.v = _mm256_fmadd_pd( a03.v, b1.v, c03_5.v );
    c47_5.v = _mm256_fmadd_pd( a47.v, b1.v, c47_5.v );

	a03.v = _mm256_load_pd( (double*)( a + 16 ) );
	a47.v = _mm256_load_pd( (double*)( a + 20 ) );

	a += 16;
	b += 12;
  }


  for ( i = 0; i < k_left; ++ i ) {

	// Iteration #0
    __asm__ volatile( "prefetcht0 192(%0)    \n\t" : :"r"(a) );

    b0.v    = _mm256_broadcast_sd( b      );
    b1.v    = _mm256_broadcast_sd( b +  1 );
    c03_0.v = _mm256_fmadd_pd( a03.v, b0.v, c03_0.v );
    c47_0.v = _mm256_fmadd_pd( a47.v, b0.v, c47_0.v );
    c03_1.v = _mm256_fmadd_pd( a03.v, b1.v, c03_1.v );
    c47_1.v = _mm256_fmadd_pd( a47.v, b1.v, c47_1.v );

    b0.v    = _mm256_broadcast_sd( b +  2 );
    b1.v    = _mm256_broadcast_sd( b +  3 );
    c03_2.v = _mm256_fmadd_pd( a03.v, b0.v, c03_2.v );
    c47_2.v = _mm256_fmadd_pd( a47.v, b0.v, c47_2.v );
    c03_3.v = _mm256_fmadd_pd( a03.v, b1.v, c03_3.v );
    c47_3.v = _mm256_fmadd_pd( a47.v, b1.v, c47_3.v );

    b0.v    = _mm256_broadcast_sd( b +  4 );
    b1.v    = _mm256_broadcast_sd( b +  5 );
    c03_4.v = _mm256_fmadd_pd( a03.v, b0.v, c03_4.v );
    c47_4.v = _mm256_fmadd_pd( a47.v, b0.v, c47_4.v );
    c03_5.v = _mm256_fmadd_pd( a03.v, b1.v, c03_5.v );
    c47_5.v = _mm256_fmadd_pd( a47.v, b1.v, c47_5.v );

	a03.v = _mm256_load_pd( (double*)( a +  8 ) );
	a47.v = _mm256_load_pd( (double*)( a + 12 ) );

    a += 8;
    b += 6;
  }
