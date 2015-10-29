  // Collect As from XA and XB.
  beg = omp_get_wtime();
  #pragma omp parallel for private( p )
  for ( i = 0; i < m; i ++ ) {
    for ( p = 0; p < k; p ++ ) {
      As[ i * k + p ] = XA[ alpha[ i ] * k + p ];
    }
  }
  #pragma omp parallel for private( p )
  for ( j = 0; j < n; j ++ ) {
    for ( p = 0; p < k; p ++ ) {
      Bs[ j * k + p ] = XB[ beta[ j ] * k + p ];
    }
  }
  time_collect = omp_get_wtime() - beg;


  // Compute the inner-product term.
  beg = omp_get_wtime();
#ifdef USE_BLAS
  dgemm( "T", "N", &m, &n, &k, &fneg2,
        As, &k, Bs, &k, &fzero, Cs, &m );
#else
  #pragma omp parallel for private( i, p )
  for ( j = 0; j < n; j ++ ) {
    for ( i = 0; i < m; i ++ ) {
      Cs[ j * m + i ] = 0.0;
      for ( p = 0; p < k; p ++ ) {
        Cs[ j * m + i ] += As[ i * k + p ] * Bs[ j * k + p ];
      }
    }
  }
#endif
  time_dgemm = omp_get_wtime() - beg;

  /*
  // 1-norm
  #pragma omp parallel for private( i, p )
  for ( j = 0; j < n; j ++ ) {
    for ( i = 0; i < m; i ++ ) {
      Cs[ j * m + i ] = 0.0;
      for ( p = 0; p < k; p ++ ) {
        Cs[ j * m + i ] += fabs( As[ i * k + p ] - Bs[ j * k + p ] );
      }
    }
  }
  */

  time_dgemm = omp_get_wtime() - beg;

  beg = omp_get_wtime();
  #pragma omp parallel for private( i )
  for ( j = 0; j < n; j ++ ) {
    for ( i = 0; i < m; i ++ ) {
#ifdef USE_BLAS
#else
      Cs[ j * m + i ] *= -2.0;
#endif
      Cs[ j * m + i ] += XA2[ alpha[ i ] ];
      Cs[ j * m + i ] += XB2[ beta[ j ] ];
    }
  }
  time_square = omp_get_wtime() - beg;
