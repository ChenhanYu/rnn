  int k_iter = k / 2;
  int k_left = k % 2;

  __asm__ volatile( "prefetcht0 0(%0)    \n\t" : :"r"( a ) );
  __asm__ volatile( "prefetcht2 0(%0)    \n\t" : :"r"( aux->b_next ) );
  __asm__ volatile( "prefetcht0 192(%0)  \n\t" : :"r"( c ) );

  c03_0.v = _mm256_setzero_ps();
  c03_1.v = _mm256_setzero_ps();
  c03_2.v = _mm256_setzero_ps();
  c03_3.v = _mm256_setzero_ps();
  c03_4.v = _mm256_setzero_ps();
  c03_5.v = _mm256_setzero_ps();

  c47_0.v = _mm256_setzero_ps();
  c47_1.v = _mm256_setzero_ps();
  c47_2.v = _mm256_setzero_ps();
  c47_3.v = _mm256_setzero_ps();
  c47_4.v = _mm256_setzero_ps();
  c47_5.v = _mm256_setzero_ps();

  // Load a03, a47, b0
  a03.v = _mm256_load_ps( (float*)  a        );
  a47.v = _mm256_load_ps( (float*)( a +  8 ) );

  for ( i = 0; i < k_iter; ++ i ) {

	// Iteration #0
    __asm__ volatile( "prefetcht0 192(%0)    \n\t" : :"r"(a) );

    b0.v    = _mm256_broadcast_ss( b      );
    b1.v    = _mm256_broadcast_ss( b +  1 );
    c03_0.v = _mm256_fmadd_ps( a03.v, b0.v, c03_0.v );
    c47_0.v = _mm256_fmadd_ps( a47.v, b0.v, c47_0.v );
    c03_1.v = _mm256_fmadd_ps( a03.v, b1.v, c03_1.v );
    c47_1.v = _mm256_fmadd_ps( a47.v, b1.v, c47_1.v );

    b0.v    = _mm256_broadcast_ss( b +  2 );
    b1.v    = _mm256_broadcast_ss( b +  3 );
    c03_2.v = _mm256_fmadd_ps( a03.v, b0.v, c03_2.v );
    c47_2.v = _mm256_fmadd_ps( a47.v, b0.v, c47_2.v );
    c03_3.v = _mm256_fmadd_ps( a03.v, b1.v, c03_3.v );
    c47_3.v = _mm256_fmadd_ps( a47.v, b1.v, c47_3.v );

    b0.v    = _mm256_broadcast_ss( b +  4 );
    b1.v    = _mm256_broadcast_ss( b +  5 );
    c03_4.v = _mm256_fmadd_ps( a03.v, b0.v, c03_4.v );
    c47_4.v = _mm256_fmadd_ps( a47.v, b0.v, c47_4.v );
    c03_5.v = _mm256_fmadd_ps( a03.v, b1.v, c03_5.v );
    c47_5.v = _mm256_fmadd_ps( a47.v, b1.v, c47_5.v );

	a03.v = _mm256_load_ps( (float*)( a + 16 ) );
	a47.v = _mm256_load_ps( (float*)( a + 24 ) );

	// Iteration #1
	__asm__ volatile( "prefetcht0 512(%0)    \n\t" : :"r"(a) );

    b0.v    = _mm256_broadcast_ss( b +  6 );
    b1.v    = _mm256_broadcast_ss( b +  7 );
    c03_0.v = _mm256_fmadd_ps( a03.v, b0.v, c03_0.v );
    c47_0.v = _mm256_fmadd_ps( a47.v, b0.v, c47_0.v );
    c03_1.v = _mm256_fmadd_ps( a03.v, b1.v, c03_1.v );
    c47_1.v = _mm256_fmadd_ps( a47.v, b1.v, c47_1.v );

    b0.v    = _mm256_broadcast_ss( b +  8 );
    b1.v    = _mm256_broadcast_ss( b +  9 );
    c03_2.v = _mm256_fmadd_ps( a03.v, b0.v, c03_2.v );
    c47_2.v = _mm256_fmadd_ps( a47.v, b0.v, c47_2.v );
    c03_3.v = _mm256_fmadd_ps( a03.v, b1.v, c03_3.v );
    c47_3.v = _mm256_fmadd_ps( a47.v, b1.v, c47_3.v );

    b0.v    = _mm256_broadcast_ss( b + 10 );
    b1.v    = _mm256_broadcast_ss( b + 11 );
    c03_4.v = _mm256_fmadd_ps( a03.v, b0.v, c03_4.v );
    c47_4.v = _mm256_fmadd_ps( a47.v, b0.v, c47_4.v );
    c03_5.v = _mm256_fmadd_ps( a03.v, b1.v, c03_5.v );
    c47_5.v = _mm256_fmadd_ps( a47.v, b1.v, c47_5.v );

	a03.v = _mm256_load_ps( (float*)( a + 32 ) );
	a47.v = _mm256_load_ps( (float*)( a + 40 ) );

	a += 32;
	b += 12;
  }


  for ( i = 0; i < k_left; ++ i ) {

	// Iteration #0
    __asm__ volatile( "prefetcht0 192(%0)    \n\t" : :"r"(a) );

    b0.v    = _mm256_broadcast_ss( b      );
    b1.v    = _mm256_broadcast_ss( b +  1 );
    c03_0.v = _mm256_fmadd_ps( a03.v, b0.v, c03_0.v );
    c47_0.v = _mm256_fmadd_ps( a47.v, b0.v, c47_0.v );
    c03_1.v = _mm256_fmadd_ps( a03.v, b1.v, c03_1.v );
    c47_1.v = _mm256_fmadd_ps( a47.v, b1.v, c47_1.v );

    b0.v    = _mm256_broadcast_ss( b +  2 );
    b1.v    = _mm256_broadcast_ss( b +  3 );
    c03_2.v = _mm256_fmadd_ps( a03.v, b0.v, c03_2.v );
    c47_2.v = _mm256_fmadd_ps( a47.v, b0.v, c47_2.v );
    c03_3.v = _mm256_fmadd_ps( a03.v, b1.v, c03_3.v );
    c47_3.v = _mm256_fmadd_ps( a47.v, b1.v, c47_3.v );

    b0.v    = _mm256_broadcast_ss( b +  4 );
    b1.v    = _mm256_broadcast_ss( b +  5 );
    c03_4.v = _mm256_fmadd_ps( a03.v, b0.v, c03_4.v );
    c47_4.v = _mm256_fmadd_ps( a47.v, b0.v, c47_4.v );
    c03_5.v = _mm256_fmadd_ps( a03.v, b1.v, c03_5.v );
    c47_5.v = _mm256_fmadd_ps( a47.v, b1.v, c47_5.v );

	a03.v = _mm256_load_ps( (float*)( a + 16 ) );
	a47.v = _mm256_load_ps( (float*)( a + 24 ) );

    a += 16;
    b += 6;
  }

  // Accumulate
  if ( aux->pc ) {
    a03.v   = _mm256_load_ps( (float*)( c      ) );
    c03_0.v = _mm256_add_ps( a03.v, c03_0.v );
    a47.v   = _mm256_load_ps( (float*)( c + 8  ) );
    c47_0.v = _mm256_add_ps( a47.v, c47_0.v );

    a03.v   = _mm256_load_ps( (float*)( c + 16 ) );
    c03_1.v = _mm256_add_ps( a03.v, c03_1.v );
    a47.v   = _mm256_load_ps( (float*)( c + 24 ) );
    c47_1.v = _mm256_add_ps( a47.v, c47_1.v );

    a03.v   = _mm256_load_ps( (float*)( c + 32 ) );
    c03_2.v = _mm256_add_ps( a03.v, c03_2.v );
    a47.v   = _mm256_load_ps( (float*)( c + 40 ) );
    c47_2.v = _mm256_add_ps( a47.v, c47_2.v );

    a03.v   = _mm256_load_ps( (float*)( c + 48 ) );
    c03_3.v = _mm256_add_ps( a03.v, c03_3.v );
    a47.v   = _mm256_load_ps( (float*)( c + 56 ) );
    c47_3.v = _mm256_add_ps( a47.v, c47_3.v );

    a03.v   = _mm256_load_ps( (float*)( c + 64 ) );
    c03_4.v = _mm256_add_ps( a03.v, c03_4.v );
    a47.v   = _mm256_load_ps( (float*)( c + 72 ) );
    c47_4.v = _mm256_add_ps( a47.v, c47_4.v );

    a03.v   = _mm256_load_ps( (float*)( c + 80 ) );
    c03_5.v = _mm256_add_ps( a03.v, c03_5.v );
    a47.v   = _mm256_load_ps( (float*)( c + 88 ) );
    c47_5.v = _mm256_add_ps( a47.v, c47_5.v );
  }
