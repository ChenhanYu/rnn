#include <omp.h>
#include <rnn.h>
#define min( i, j ) ( (i)<(j) ? (i): (j) )

inline void packA_kcxmc(
    int    m,
    int    k,
    double *XA,
    int    ldXA,
    int    *amap,
    double *packA
    )
{
  int    i, p;
  double *a_pntr[ DRNN_MR ];

  for ( i = 0; i < m; i ++ ) {
    a_pntr[ i ] = XA + ldXA * amap[ i ];
  }

  for ( i = m; i < DRNN_MR; i ++ ) {
    a_pntr[ i ] = XA + ldXA * amap[ 0 ];
  }

  for ( p = 0; p < k; p ++ ) {
    for ( i = 0; i < DRNN_MR; i ++ ) {
      *packA ++ = *a_pntr[ i ] ++;
    }
  }
}


// Pack B from 4 different columns of XB
inline void packB_kcxnc(
    int    n,
    int    k,
    double *XB,
    int    ldXB, // ldXB is the original k
    int    *bmap,
    double *packB
    )
{
  int    j, p; 
  double *b_pntr[ DRNN_NR ];

  for ( j = 0; j < n; j ++ ) {
    b_pntr[ j ] = XB + ldXB * bmap[ j ];
  }

  for ( j = n; j < DRNN_NR; j ++ ) {
    b_pntr[ j ] = XB + ldXB * bmap[ 0 ];
  }

  for ( p = 0; p < k; p ++ ) {
    for ( j = 0; j < DRNN_NR; j ++ ) {
      *packB ++ = *b_pntr[ j ] ++;
    }
  }
}




/*
 * --------------------------------------------------------------------------
 * @brief  This macro-kernel contains the 3.rd and the 2.nd loop of the
 *         rank-k update.
 * 
 * @param  m        Number of target points
 * @param  n        Number of source points
 * @param  k        Data point dimension
 * @param  *packA   Packed target points coordinates
 * @param  *packB   Packed source points coordinates
 * @param  *packC   Packed accumulated rank-k update
 * @param  ldc      Leading dimension of packC
 * @param  pc       5.th loop counter, used to indicate whether this is the
 *                  first iteration
 * --------------------------------------------------------------------------
 */
void rank_k_macro_kernel(
    int    m,
    int    n,
    int    k,
    double *packA,
    double *packB,
    double *packC,
    int    ldc,
    int    pc
    )
{
  int    i, j;
  aux_t  aux;

  aux.pc     = pc;
  aux.b_next = packB;

  for ( j = 0; j < n; j += DRNN_NR ) {
    for ( i = 0; i < m; i += DRNN_MR ) {
      if ( i + DRNN_MR >= m ) {
        aux.b_next += DRNN_NR * k;
      }
      rnn_rank_k_asm_d8x4(
          k,
          &packA[ i * k ],
          &packB[ j * k ],
          &packC[ j * ldc + i * DRNN_NR ], // packed
          ldc,
          &aux
          );
    }
  }
}


// This macro kernel is called in the last iteration while k > KC. 
// packC is required, and it will be discarded after this micro kernel call.
// 
// ldC is max( MC, ( ( m - 1 ) / MR + 1 ) * MR )
//
// We need to deal with the edge case here. Before every micro-kernel call,
// compute the valid aux.m and aux.n.
void dgsrnn_macro_kernel_case2(
    int    m,
    int    n,
    int    k,
    int    r,
    double *packA,
    double *packA2,
    int    *amap,
    double *packB,
    double *packB2,
    double *packC,
    int    ldc,
    int    pc,
    double *D,
    int    *I
    )
{
  int    i, j, jj;
  aux_t  aux;

  aux.pc = pc;
  aux.b_next = packB;

  for ( j = 0; j < n; j += DRNN_NR ) {
    aux.n  = min( n - j, DRNN_NR );
    for ( i = 0; i < m; i += DRNN_MR ) {
      aux.m = min( m - i, DRNN_MR );
      if ( i + DRNN_MR >= m ) {
        aux.b_next += DRNN_NR * k;
      }

      rnn_asm_d8x4(
          k,
          &packA[ i * k ],
          packA2 + i,
          &packB[ j * k ],
          packB2 + j,
          &packC[ j * ldc + i * DRNN_NR ], // packed
          &aux
          );

      for ( jj = 0; jj < aux.n; jj ++ ) {
        heap_sort( 
            aux.m, 
            r, 
            packC + j * ldc + i * DRNN_NR + 8 * jj, 
            amap + i, 
            D + r * ( j + jj ), 
            I + r * ( j + jj ) 
            ); 
      }
    }
  }
}

// This macro kernel is called if k <= KC.
/*
 * --------------------------------------------------------------------------
 * @brief  This macro-kernel contains the 3.rd and the 2.nd loop of the
 *         rank-k update, computing the square distance in the micro-kernel.
 *
 * @param  m        Number of target points
 * @param  n        Number of source points
 * @param  k        Data point dimension
 * @param  r        The desired size of the nearest neighbor
 * @param  *packA   Packed target points coordinates
 * @param  *packA2  Packed target points square distance
 * @param  *amap    Target point index map, used as a tracing id
 * @param  *packB   Packed source points coordinates
 * @param  *packB2  Packed source points square distance
 * @param  *D       Square distance of the current r-nn, arranged as several
 *                  length-r heaps or sorted lists. [ Key value ] 
 * @param  *I       Corresponing point index of D
 */ 
void dgsrnn_macro_kernel(
    int    m,
    int    n,
    int    k,
    int    r,
    double *packA,
    double *packA2,
    int    *amap,
    double *packB,
    double *packB2,
    double *D,
    int    *I
    )
{
  double c[ DRNN_MR * DRNN_NR ] __attribute__((aligned(32)));
  int    i, j, jj;
  aux_t  aux;

  aux.b_next = packB;

  for ( j = 0; j < n; j += DRNN_NR ) {
    aux.n  = min( n - j, DRNN_NR );
    aux.I  = I + j * r;
    aux.D  = D + j * r;
    for ( i = 0; i < m; i += DRNN_MR ) {
      aux.m = min( m - i, DRNN_MR );
      if ( i + DRNN_MR >= m ) {
        aux.b_next += DRNN_NR * k;
      }
      if ( r < 0 ) {
        //rnn_r1_int_d8x4(
        //    k,
        //    0.0,
        //    &packA2[ i ],
        //    &packA[ i * k ],
        //    &packB2[ j ],
        //    &packB[ j * k ],
        //    &aux,
        //    &amap[ i ],
        //    &I[ j ],
        //    &D[ j ]
        //    );
      }
      else {
        // ------------------------------------------------------------------
        // Compute the square distance
        // ------------------------------------------------------------------
        //rnn_int_d8x4(
        //    k,
        //    &packA2[ i ],
        //    &packA[ i * k ],
        //    &packB2[ j ],
        //    &packB[ j * k ],
        //    c,
        //    &aux
        //    );
        // ------------------------------------------------------------------


        // ------------------------------------------------------------------
        // Heap insertion
        // ------------------------------------------------------------------
        //for ( jj = 0; jj < aux.n; jj ++ ) {
        //  heap_sort( 
        //      aux.m, 
        //      r, 
        //      c + 8 * jj, 
        //      amap + i, 
        //      D + r * ( j + jj ), 
        //      I + r * ( j + jj ) 
        //      ); 
        //}
        // ------------------------------------------------------------------


        // ------------------------------------------------------------------
        // Combine selective square distance and the heap adjustment.
        // ------------------------------------------------------------------
        rnn_r_int_d8x4(
            k,
            r,
            packA2 + i,
            &packA[ i * k ],
            packB2 + j,
            &packB[ j * k ],
            c,
            &aux,
            amap + i,
            I + r * j,
            D + r * j
            );
        // ------------------------------------------------------------------
      }
    }
  }
}


void dgsrnn_macro_kernel_row(
    int    m,
    int    n,
    int    k,
    int    r,
    double *packA,
    double *packA2,
    double *packB,
    double *packB2,
    int    *bmap,
    double *D,
    int    *I,
    int    ldr
    )
{
  double c[ DRNN_MR * DRNN_NR ] __attribute__((aligned(32)));
  int    i, j, ii, jj, mr, nr;
  aux_t  aux;

  double beg;

  aux.pc  = 0;
  aux.ldr = ldr;

  for ( j = 0; j < n; j += DRNN_NR ) {
    aux.n = min( n - j, DRNN_NR );
    for ( i = 0; i < m; i += DRNN_MR ) {
      aux.m = min( m - i, DRNN_MR );
      aux.I = I + i * ldr;
      aux.D = D + i * ldr;
      if ( i + DRNN_MR >= m ) {
        aux.b_next += DRNN_NR * k;
      }

      // --------------------------------------------------------------------
      // Compute the square distance
      // --------------------------------------------------------------------
      //rnn_int_d8x4_var2(
      //    k,
      //    packA2 + i,
      //    &packA[ i * k ],
      //    packB2 + j,
      //    &packB[ j * k ],
      //    c,
      //    &aux
      //    );
      // --------------------------------------------------------------------

      
      // --------------------------------------------------------------------
      // Heap insertion
      // --------------------------------------------------------------------
      //for ( ii = 0; ii < aux.m; ii ++ ) {
      //  heap_sort( 
      //  //heapSelect_dheap_var2(
      //      aux.n, 
      //      r, 
      //      c + 4 * ii, 
      //      bmap + j, 
      //      D + ldr * ( i + ii ), 
      //      I + ldr * ( i + ii ) 
      //      );
      //}
      // --------------------------------------------------------------------


      // --------------------------------------------------------------------
      // Combine selective square distance and the heap adjustment.
      // --------------------------------------------------------------------
      rnn_r_int_d8x4_row(
          k,
          r,
          &packA2[ i ],
          &packA[ i * k ],
          &packB2[ j ],
          &packB[ j * k ],
          c,
          &aux,
          bmap + j
          );
      // --------------------------------------------------------------------
    }
  }
}



void dgsrnn_macro_kernel_var3(
    int    m,
    int    n,
    int    k,
    int    r,
    double *packA,
    double *packA2,
    double *packB,
    double *packB2,
    int    *bmap,
    double *D,
    int    *I,
    int    ldr
    )
{
  double c[ DRNN_MC * DRNN_NC ] __attribute__((aligned(32)));
  char   flag[ DRNN_MC * ( DRNN_NC / DRNN_NR ) ] = { 0 };
  int    i, j, ii, jj, mr, nr, ldc;
  aux_t  aux;

  aux.pc  = 0;
  aux.ldr = ldr;

  ldc = ( ( n - 1 ) / DRNN_NR + 1 ) * DRNN_NR;
  //ldr = I[ 2 ];

  //printf( "macro_kernel_var3(): ldr = %d\n", ldr );

  /*
   * For example:
   *
   * c = [   c0,  c1,  c2,  c3,    x,   x,   x,   x;
   *          x,   x,   x,   x,   c4,  c5,  c6,  c7;
   *          x,   x,   x,   x,    x,   x,   x,   x;
   *         c8,  c9, c10, c11,  c12, c13, c14, c15;  ]
   *
   * flag = [ 1, 0;
   *          0, 1;
   *          0, 0; 
   *          1, 1; ]
   * */
  // ------------------------------------------------------------------------
  // Flag initialization
  // ------------------------------------------------------------------------
  //for ( i = 0; i < DRNN_MC * ( DRNN_NC / DRNN_NR ); i ++ ) {
  //  flag[ i ] = 0;
  //}
  // ------------------------------------------------------------------------


  for ( j = 0; j < n; j += DRNN_NR ) {
    aux.n = min( n - j, DRNN_NR );
    for ( i = 0; i < m; i += DRNN_MR ) {
      aux.m = min( m - i, DRNN_MR );
      aux.I = I + i * ldr;
      aux.D = D + i * ldr;
      if ( i + DRNN_MR >= m ) {
        aux.b_next += DRNN_NR * k;
      }

      // --------------------------------------------------------------------
      // Compute the square distance. Store c selectively.
      // --------------------------------------------------------------------
      //rnn_r_int_d8x4_var3(
      //    k,
      //    r,
      //    packA2 + i,
      //    &packA[ i * k ],
      //    packB2 + j,
      //    &packB[ j * k ],
      //    &c[ i * DRNN_NC + j ],
      //    &flag[ i * ( DRNN_NC / DRNN_NR ) + ( j / DRNN_NR ) ],
      //    &aux
      //    );
      // ------------------------------------------------------------------------
      rnn_asm_d8x4_var3(
          k,
          &packA[ i * k ],
          packA2 + i,
          &packB[ j * k ],
          packB2 + j,
          &c[ i * ldc + j ],
          ldc,
          &aux
          );
    }
  }

  //printf( "here\n" );

  // ------------------------------------------------------------------------
  // Heap Adjustment
  // ------------------------------------------------------------------------
  //for ( i = 0; i < m; i ++ ) {
  //  for ( j = 0; j < n; j += DRNN_NR ) {
  //    aux.n = min( n - j, DRNN_NR );
  //    if ( flag[ i * ( DRNN_NC / DRNN_NR ) + ( j / DRNN_NR ) ] ) {
  //      heap_sort( 
  //          aux.n,
  //          r,
  //          c + i * DRNN_NC + j,
  //          bmap + j,
  //          D + i * ldr, 
  //          I + i * ldr 
  //          );
  //    }
  //  }
  //}
  // ------------------------------------------------------------------------
  // ------------------------------------------------------------------------
  for ( i = 0; i < m; i ++ ) {
    //heap_sort(
    heapSelect_dheap_var2(
    //heapSelect_int_d4(
        n, 
        r, 
        c + ldc * i, 
        bmap, 
        D + ldr * i, 
        I + ldr * i  
        );
  }
}




void dgsrnn_macro_kernel_row_large_k(
    int    m,
    int    n,
    int    k,
    int    r,
    double *packA,
    double *packA2,
    double *packB,
    double *packB2,
    int    *bmap,
    double *packC,
    int    ldc,
    int    pc,
    double *D,
    int    *I,
    int    ldr
    )
{
  double c[ DRNN_MR * DRNN_NR ] __attribute__((aligned(32)));
  int    i, ii, j;
  aux_t  aux;
  double beg;


  aux.pc     = pc;
  aux.b_next = packB;
  aux.ldr    = ldr;


  for ( j = 0; j < n; j += DRNN_NR ) {
    aux.n  = min( n - j, DRNN_NR );
    for ( i = 0; i < m; i += DRNN_MR ) {
      aux.m = min( m - i, DRNN_MR );
      aux.I = I + i * ldr;
      aux.D = D + i * ldr;
      if ( i + DRNN_MR >= m ) {
        aux.b_next += DRNN_NR * k;
      }

      // Notice that packC is packed column major, yet we need a row major
      // to perform the heap adjustment.
      //rnn_asm_d8x4_case2(
      //    k,
      //    &packA[ i * k ],
      //    packA2 + i,
      //    &packB[ j * k ],
      //    packB2 + j,
      //    &packC[ j * ldc + i * DRNN_NR ], // packed
      //    c,
      //    &aux
      //    );

      // --------------------------------------------------------------------
      // Heap insertion
      // --------------------------------------------------------------------
      //for ( ii = 0; ii < aux.m; ii ++ ) {
      //  heap_sort( 
      //  //heapSelect_d16( 
      //  //heapSelect_dheap_var2( 
      //      aux.n, 
      //      r, 
      //      c + 4 * ii, 
      //      bmap + j, 
      //      D + ldr * ( i + ii ), 
      //      I + ldr * ( i + ii ) 
      //      );
      //}
      // --------------------------------------------------------------------
      
      // --------------------------------------------------------------------
      // Combine selective square distance and the heap adjustment.
      // --------------------------------------------------------------------
      rnn_r_int_d8x4_row(
          k,
          r,
          &packA2[ i ],
          &packA[ i * k ],
          &packB2[ j ],
          &packB[ j * k ],
          &packC[ j * ldc + i * DRNN_NR ], // packed
          &aux,
          bmap + j
          );
      // --------------------------------------------------------------------
    }
  }


}


void dgsrnn_macro_kernel_var3_case2(
    int    m,
    int    n,
    int    k,
    int    r,
    double *packA,
    double *packA2,
    double *packB,
    double *packB2,
    int    *bmap,
    double *packC,
    int    ldc,
    int    pc,
    double *D,
    int    *I
    )
{
  double c[ DRNN_MC * DRNN_NC ] __attribute__((aligned(32)));

  int    i, ii, j, ldctmp, ldr;
  aux_t  aux;

  aux.pc = pc;
  aux.b_next = packB;


  ldctmp = ( ( n - 1 ) / DRNN_NR + 1 ) * DRNN_NR;
  ldr = I[ 2 ];

  for ( j = 0; j < n; j += DRNN_NR ) {
    aux.n  = min( n - j, DRNN_NR );
    for ( i = 0; i < m; i += DRNN_MR ) {
      aux.m = min( m - i, DRNN_MR );
      if ( i + DRNN_MR >= m ) {
        aux.b_next += DRNN_NR * k;
      }
      rnn_asm_d8x4_var3_case2(
          k,
          &packA[ i * k ],
          packA2 + i,
          &packB[ j * k ],
          packB2 + j,
          &packC[ j * ldc + i * DRNN_NR ], // packed
          &c[ i * ldctmp + j ],
          ldctmp,
          &aux
          );
    }
  }

  // ------------------------------------------------------------------------
  // Heap Adjustment
  // ------------------------------------------------------------------------
  //printf( "heapSelect_d16()\n" );
  for ( i = 0; i < m; i ++ ) {
    heap_sort( 
    //heapSelect_dheap_var2( 
    //heapSelect_int_d4(
        n, 
        r, 
        c + ldctmp * i, 
        bmap, 
        D + ldr * i, 
        I + ldr * i  
        );
  }
}




void dsq2nrm_macro_kernel(
    int    m,
    int    n,
    int    k,
    double *packA,
    double *packA2,
    double *packB,
    double *packB2,
    double *C,
    int    ldc,
    int    pc,
    int    lastiter
    )
{
  int    i, ii, j;
  aux_t  aux;

  aux.pc = pc;
  aux.b_next = packB;

  //printf( "here, pc = %d, last = %d, ldc = %d, m = %d, n = %d, k %d\n", 
  //    pc, lastiter, ldc, m, n , k );

  for ( j = 0; j < n; j += DRNN_NR ) {
    aux.n  = min( n - j, DRNN_NR );
    for ( i = 0; i < m; i += DRNN_MR ) {
      aux.m = min( m - i, DRNN_MR );
      if ( i + DRNN_MR >= m ) {
        aux.b_next += DRNN_NR * k;
      }

      sq2nrm_asm_d8x4(
          k,
          &packA[ i * k ],
          packA2 + i,
          &packB[ j * k ],
          packB2 + j,
          &C[ j * ldc + i ],
          (unsigned long long) ldc,
          (unsigned long long) lastiter,
          &aux
          );
    }
  }
}




// C must be aligned
void dgssq2nrm(
    int    m,
    int    n,
    int    k,
    double *XA,
    double *XA2,
    int    *amap,
    double *XB,
    double *XB2,
    int    *bmap,
    double *C,        // must be aligned
    int    ldc        // ldc must also be aligned
)
{
  int    i, j, p, rnn_ic_nt;
  int    ic, ib, jc, jb, pc, pb;
  int    ir, jr;
  double *packA, *packB, *packA2, *packB2;
  char   *str;


  // Early return if possible
  if ( m == 0 || n == 0 || k == 0 ) {
    printf( "dgsrnn(): early return\n" );
    return;
  }


  // sequential is the default situation
  rnn_ic_nt = 1;


  // check the environment variable
  str = getenv( "RNN_IC_NT" );
  if ( str != NULL ) {
    rnn_ic_nt = (int)strtol( str, NULL, 10 );
  }


  // Allocate packing buffers
  packA  = rnn_malloc_aligned( DRNN_KC, ( DRNN_MC + 1 ) * rnn_ic_nt, sizeof(double) );
  packB  = rnn_malloc_aligned( DRNN_KC, ( DRNN_NC + 1 )            , sizeof(double) );
  packA2 = rnn_malloc_aligned(       1, ( DRNN_MC + 1 ) * rnn_ic_nt, sizeof(double) );
  packB2 = rnn_malloc_aligned(       1, ( DRNN_NC + 1 )            , sizeof(double) );


  for ( jc = 0; jc < n; jc += DRNN_NC ) {                  // 6-th loop
    jb = min( n - jc, DRNN_NC );
    for ( pc = 0; pc < k; pc += DRNN_KC ) {                // 5-th loop
      pb = min( k - pc, DRNN_KC );


      #pragma omp parallel for num_threads( rnn_ic_nt ) private( jr )
      for ( j = 0; j < jb; j += DRNN_NR ) {
        if ( pc + DRNN_KC >= k ) {
          for ( jr = 0; jr < min( jb - j, DRNN_NR ); jr ++ ) {
            packB2[ j + jr ] = XB2[ bmap[ jc + j + jr ] ];
          }
        }
        packB_kcxnc(
            min( jb - j, DRNN_NR ),
            pb,
            &XB[ pc ],
            k, // should be ldXB instead
            &bmap[ jc + j ],
            &packB[ j * pb ]
            );
      }

      #pragma omp parallel for num_threads( rnn_ic_nt ) private( ic, ib, i, ir )
      for ( ic = 0; ic < m; ic += DRNN_MC ) {              // 4-th loop
        int     tid = omp_get_thread_num();

        ib = min( m - ic, DRNN_MC );
        for ( i = 0; i < ib; i += DRNN_MR ) {
          if ( pc + DRNN_KC >= k ) {
            for ( ir = 0; ir < min( ib - i, DRNN_MR ); ir ++ ) {
              packA2[ tid * DRNN_MC + i + ir ] = XA2[ amap[ ic + i + ir ] ];
            }
          }
          packA_kcxmc(
              min( ib - i, DRNN_MR ),
              pb,
              &XA[ pc ],
              k,
              &amap[ ic + i ],
              &packA[ tid * DRNN_MC * pb + i * pb ]
              );
        }


        dsq2nrm_macro_kernel(
            ib,
            jb,
            pb,
            packA  + tid * DRNN_MC * pb,
            packA2 + tid * DRNN_MC,
            packB,
            packB2,
            &C[ jc * ldc + ic ], 
            ldc,
            pc,
            ( pc + DRNN_KC >= k )
            );

      }                                                    // End 4.th loop
    }                                                      // End 5.th loop
  }                                                        // End 6.th loop

  free( packA );
  free( packB );
  free( packA2 );
  free( packB2 );
}


void dgsrnn_var3(
    int    m,
    int    n,
    int    k,
    int    r,
    double *XA,
    double *XA2,
    int    *amap,
    double *XB,
    double *XB2,
    int    *bmap,
    double *D,
    int    *I
    )
{
  int    i, j, ldr;
  int    ldc = ( ( m - 1 ) / DRNN_MR + 1 ) * DRNN_MR;
  double beg, time_heap, time_sq2nrm;


  double *C = rnn_malloc_aligned( ldc, n + 4, sizeof(double) );

  //if ( posix_memalign( (void**)&C, (size_t)DRNN_SIMD_ALIGN_SIZE, 
  //      sizeof(double) * ldc * ( n + 4 ) ) ) {
  //  printf( "test_dgsrnn_var2(): posix_memalign() failures" );
  //  exit( 1 );    
  //}

  beg = omp_get_wtime();
  dgssq2nrm(
      m,
      n,
      k,
      XA,
      XA2,
      amap,
      XB,
      XB2,
      bmap,
      C,
      ldc
      );
  time_sq2nrm = omp_get_wtime() - beg;


  ldr = I[ 2 ];

  beg = omp_get_wtime();
  #pragma omp parallel for
  for ( j = 0; j < n; j ++ ) {
    //heap_sort( m, r, &C[ j * ldc ], amap, &D[ j * r ], &I[ j * r ] );
    heapSelect_dheap_var2( m, r, &C[ j * ldc ], amap, &D[ j * ldr ], &I[ j * ldr ] );
  }
  time_heap = omp_get_wtime() - beg;

  //printf( "gsrnn sq2nrm: %5.3lf, heap: %5.3lf\n", time_sq2nrm, time_heap );

  free( C );
}





void dgsrnn(
    int    m,
    int    n,
    int    k,
    int    r,
    double *XA,
    double *XA2,
    int    *amap,
    double *XB,
    double *XB2,
    int    *bmap,
    double *D,
    int    *I
    )
{
  int    i, j, p;
  int    ic, ib, jc, jb, pc, pb;
  int    ir, jr;
  int    rnn_ic_nt;
  int    ldc, padn;
  double *packA, *packB, *packC, *packA2, *packB2;


  // Early return if possible
  if ( m == 0 || n == 0 || k == 0 || r == 0 ) {
    printf( "dgsrnn(): early return\n" );
    return;
  }


  // sequential is the default situation
  rnn_ic_nt = 1;


  // Allocate packing buffers
  packA  = rnn_malloc_aligned( DRNN_KC, ( DRNN_MC + 1 ) * rnn_ic_nt, sizeof(double) );
  packB  = rnn_malloc_aligned( DRNN_KC, ( DRNN_NC + 1 )            , sizeof(double) );
  packA2 = rnn_malloc_aligned(       1, ( DRNN_MC + 1 ) * rnn_ic_nt, sizeof(double) );
  packB2 = rnn_malloc_aligned(       1, ( DRNN_NC + 1 )            , sizeof(double) );


  if ( k > DRNN_KC ) {

    ldc  = ( ( m - 1 ) / DRNN_MR + 1 ) * DRNN_MR;
    padn = DRNN_NC;
    if ( n < DRNN_NC ) {
      padn = ( ( n - 1 ) / DRNN_NR + 1 ) * DRNN_NR;
    }

    packC  = rnn_malloc_aligned( ldc, padn, sizeof(double) );


    for ( jc = 0; jc < n; jc += DRNN_NC ) {           // 6-th loop
      jb = min( n - jc, DRNN_NC );
      for ( pc = 0; pc < k; pc += DRNN_KC ) {         // 5-th loop
        pb = min( k - pc, DRNN_KC );
        for ( j = 0; j < jb; j += DRNN_NR ) {
          if ( pc + DRNN_KC >= k ) {
            for ( jr = 0; jr < min( jb - j, DRNN_NR ); jr ++ ) {
                packB2[ j + jr ] = XB2[ bmap[ jc + j + jr ] ];
            }
          }
          packB_kcxnc(
              min( jb - j, DRNN_NR ),
              pb,
              &XB[ pc ],
              k, // should be ldXB instead
              &bmap[ jc + j ],
              &packB[ j * pb ]
              );
        }


        for ( ic = 0; ic < m; ic += DRNN_MC ) {       // 4-th loop
          int     tid = 0;

          ib = min( m - ic, DRNN_MC );
          for ( i = 0; i < ib; i += DRNN_MR ) {
            if ( pc + DRNN_KC >= k ) {
              for ( ir = 0; ir < min( ib - i, DRNN_MR ); ir ++ ) {
                packA2[ tid * DRNN_MC + i + ir ] = XA2[ amap[ ic + i + ir ] ];
              }
            }
            packA_kcxmc(
                min( ib - i, DRNN_MR ),
                pb,
                &XA[ pc ],
                k,
                &amap[ ic + i ],
                &packA[ tid * DRNN_MC * pb + i * pb ]
                );
          }

          // Check if this is the last kc interation
          if ( pc + DRNN_KC < k ) {
            //if ( r > 64 ) {
            //  rank_k_macro_kernel_col(
            //      ib,
            //      jb,
            //      pb,
            //      packA + tid * DRNN_MC * pb,
            //      packB,
            //      &packC[ ic * padn ], // packed
            //      ldc, // non-packed
            //      pc
            //      );
            //}
            //else {
              rank_k_macro_kernel(
                  ib,
                  jb,
                  pb,
                  packA + tid * DRNN_MC * pb,
                  packB,
                  &packC[ ic * padn ], // packed
                  ( ( ib - 1 ) / DRNN_MR + 1 ) * DRNN_MR, // packed
                  pc
                  );
            //}
          }
          else {
            //printf( "dgsrnn(): k > KC macro kerenel hasn't been implemented yet.\n" );
              dgsrnn_macro_kernel_case2(                      // 1~3 loops
                  ib,
                  jb,
                  pb,
                  r,
                  packA  + tid * DRNN_MC * pb,
                  packA2 + tid * DRNN_MC,
                  amap + ic,
                  packB,
                  packB2,
                  &packC[ ic * padn ],                    // packed
                  ( ( ib - 1 ) / DRNN_MR + 1 ) * DRNN_MR, // packed
                  pc,
                  D + jc * r,
                  I + jc * r
                  );
          }
        }
      }
    }


    free( packC );
  }
  else {

    for ( jc = 0; jc < n; jc += DRNN_NC ) {           // 6-th loop
      jb = min( n - jc, DRNN_NC );
      for ( pc = 0; pc < k; pc += DRNN_KC ) {         // 5-th loop
        pb = min( k - pc, DRNN_KC );
        // packB, packw, packbb
        for ( j = 0; j < jb; j += DRNN_NR ) {
          // packw and packB2
          for ( jr = 0; jr < min( jb - j, DRNN_NR ); jr ++ ) {
            packB2[ j + jr ] = XB2[ bmap[ jc + j + jr ] ];
          }
          // packB
          packB_kcxnc(
              min( jb - j, DRNN_NR ),
              pb,
              XB,
              k, // should be ldXB instead
              &bmap[ jc + j ],
              &packB[ j * pb ]
              );
        }

        for ( ic = 0; ic < m; ic += DRNN_MC ) {       // 4-th loop
          int     tid = 0;

          ib = min( m - ic, DRNN_MC );
          for ( i = 0; i < ib; i += DRNN_MR ) {
            for ( ir = 0; ir < min( ib - i, DRNN_MR ); ir ++ ) {
              packA2[ tid * DRNN_MC + i + ir ] = XA2[ amap[ ic + i + ir ] ];
            }
            packA_kcxmc(
                min( ib - i, DRNN_MR ),
                pb,
                XA,
                k,
                &amap[ ic + i ],
                &packA[ tid * DRNN_MC * pb + i * pb ]
                );
          }

          dgsrnn_macro_kernel(                      // 1~3 loops
              ib,
              jb,
              pb,
              r,
              packA  + tid * DRNN_MC * pb,
              packA2 + tid * DRNN_MC,
              amap + ic,
              packB,
              packB2,
              D + jc * r,
              I + jc * r
              );
        }
      }
    }
  }


  free( packA );
  free( packB );
  free( packA2 );
  free( packB2 );
}


void dgsrnn_var2(
    int    m,
    int    n,
    int    k,
    int    r,
    double *XA,
    double *XA2,
    int    *amap,
    double *XB,
    double *XB2,
    int    *bmap,
    double *D,
    int    *I
    )
{
  int    i, j, p, rnn_ic_nt;
  int    ic, ib, jc, jb, pc, pb;
  int    ir, jr;
  int    ldc, padn, ldr;
  double *packA, *packB, *packC, *packw, *packu, *packA2, *packB2;
  char   *str;


  // Early return if possible
  if ( m == 0 || n == 0 || k == 0 ) {
    printf( "dgsrnn(): early return\n" );
    return;
  }


  // sequential is the default situation
  rnn_ic_nt = 1;


  // check the environment variable
  str = getenv( "RNN_IC_NT" );
  if ( str != NULL ) {
    rnn_ic_nt = (int)strtol( str, NULL, 10 );
  }


  // D-array heap leading dimenstion.
  //ldr = I[ 2 ];
  ldr = r;


  // Allocate packing buffers
  packA  = rnn_malloc_aligned( DRNN_KC, ( DRNN_MC + 1 ) * rnn_ic_nt, sizeof(double) );
  packB  = rnn_malloc_aligned( DRNN_KC, ( DRNN_NC + 1 )            , sizeof(double) );
  packA2 = rnn_malloc_aligned(       1, ( DRNN_MC + 1 ) * rnn_ic_nt, sizeof(double) );
  packB2 = rnn_malloc_aligned(       1, ( DRNN_NC + 1 )            , sizeof(double) );


  if ( k > DRNN_KC ) {
    ldc  = ( ( m - 1 ) / DRNN_MR + 1 ) * DRNN_MR;
    padn = DRNN_NC;
    if ( n < DRNN_NC ) {
      padn = ( ( n - 1 ) / DRNN_NR + 1 ) * DRNN_NR;
    }


    packC  = rnn_malloc_aligned( ldc, padn, sizeof(double) );


    for ( jc = 0; jc < n; jc += DRNN_NC ) {           // 6-th loop
      jb = min( n - jc, DRNN_NC );
      for ( pc = 0; pc < k; pc += DRNN_KC ) {         // 5-th loop
        pb = min( k - pc, DRNN_KC );

        // packB, packw, packbb
        #pragma omp parallel for num_threads( rnn_ic_nt ) private( jr )
        for ( j = 0; j < jb; j += DRNN_NR ) {
          if ( pc + DRNN_KC >= k ) {
            // packw and packB2
            for ( jr = 0; jr < min( jb - j, DRNN_NR ); jr ++ ) {
                packB2[ j + jr ] = XB2[ bmap[ jc + j + jr ] ];
            }
          }
          packB_kcxnc(
              min( jb - j, DRNN_NR ),
              pb,
              &XB[ pc ],
              k, // should be ldXB instead
              &bmap[ jc + j ],
              &packB[ j * pb ]
              );
        }

        #pragma omp parallel for num_threads( rnn_ic_nt ) private( ic, ib, i, ir )
        for ( ic = 0; ic < m; ic += DRNN_MC ) {       // 4-th loop
          //int     tid = 0;
          int     tid = omp_get_thread_num();

          ib = min( m - ic, DRNN_MC );
          for ( i = 0; i < ib; i += DRNN_MR ) {
            if ( pc + DRNN_KC >= k ) {
              for ( ir = 0; ir < min( ib - i, DRNN_MR ); ir ++ ) {
                packA2[ tid * DRNN_MC + i + ir ] = XA2[ amap[ ic + i + ir ] ];
              }
            }
            packA_kcxmc(
                min( ib - i, DRNN_MR ),
                pb,
                &XA[ pc ],
                k,
                &amap[ ic + i ],
                &packA[ tid * DRNN_MC * pb + i * pb ]
                );
          }

          // Check if this is the last kc interation
          if ( pc + DRNN_KC < k ) {
            rank_k_macro_kernel(
                ib,
                jb,
                pb,
                packA + tid * DRNN_MC * pb,
                packB,
                &packC[ ic * padn ], // packed
                ( ( ib - 1 ) / DRNN_MR + 1 ) * DRNN_MR, // packed
                pc
                );
          }
          else {
            if ( r <= 2048 ) {
              dgsrnn_macro_kernel_row_large_k(                      // 1~3 loops
                  ib,
                  jb,
                  pb,
                  r,
                  packA  + tid * DRNN_MC * pb,
                  packA2 + tid * DRNN_MC,
                  packB,
                  packB2,
                  bmap   + jc,
                  &packC[ ic * padn ], // packed
                  ( ( ib - 1 ) / DRNN_MR + 1 ) * DRNN_MR, // packed
                  pc,
                  D      + ic * ldr, // D is m x ldr (d-array heap) 
                  I      + ic * ldr, // I is m x ldr (d-array heap)
                  ldr
                  );
            }
            else {
              dgsrnn_macro_kernel_var3_case2(                      // 1~3 loops
                  ib,
                  jb,
                  pb,
                  r,
                  packA  + tid * DRNN_MC * pb,
                  packA2 + tid * DRNN_MC,
                  packB,
                  packB2,
                  bmap + jc,
                  &packC[ ic * padn ], // packed
                  ( ( ib - 1 ) / DRNN_MR + 1 ) * DRNN_MR, // packed
                  pc,
                  D      + ic * ldr, // D is m x ldr (d-array heap) 
                  I      + ic * ldr  // I is m x ldr (d-array heap)
                  );
            }
          }
        }
      }
    }

    free( packC );
  }
  else {

    for ( jc = 0; jc < n; jc += DRNN_NC ) {                // 6-th loop
      jb = min( n - jc, DRNN_NC );
      for ( pc = 0; pc < k; pc += DRNN_KC ) {              // 5-th loop
        pb = min( k - pc, DRNN_KC );

        #pragma omp parallel for num_threads( rnn_ic_nt ) private( jr )
        for ( j = 0; j < jb; j += DRNN_NR ) {
          for ( jr = 0; jr < min( jb - j, DRNN_NR ); jr ++ ) {
            packB2[ j + jr ] = XB2[ bmap[ jc + j + jr ] ];
          }
          packB_kcxnc(
              min( jb - j, DRNN_NR ),
              pb,
              XB,
              k, // should be ldXB instead
              &bmap[ jc + j ],
              &packB[ j * pb ]
              );
        }

        #pragma omp parallel for num_threads( rnn_ic_nt ) private( ic, ib, i, ir )
        for ( ic = 0; ic < m; ic += DRNN_MC ) {            // 4-th loop
          int     tid = omp_get_thread_num();

          ib = min( m - ic, DRNN_MC );
          for ( i = 0; i < ib; i += DRNN_MR ) {
            for ( ir = 0; ir < min( ib - i, DRNN_MR ); ir ++ ) {
              packA2[ tid * DRNN_MC + i + ir ] = XA2[ amap[ ic + i + ir ] ];
            }
            packA_kcxmc(
                min( ib - i, DRNN_MR ),
                pb,
                XA,
                k,
                &amap[ ic + i ],
                &packA[ tid * DRNN_MC * pb + i * pb ]
                );
          }

          if ( r <= 2048 ) {
            dgsrnn_macro_kernel_row(                      // 1~3 loops
                ib,
                jb,
                pb,
                r,
                packA  + tid * DRNN_MC * pb,
                packA2 + tid * DRNN_MC,
                packB,
                packB2,
                bmap   + jc,
                D      + ic * ldr, // D is m x ldr (d-array heap) 
                I      + ic * ldr, // I is m x ldr (d-array heap)
                ldr
                );
          }
          else {
            dgsrnn_macro_kernel_var3(                      // 1~3 loops
                ib,
                jb,
                pb,
                r,
                packA  + tid * DRNN_MC * pb,
                packA2 + tid * DRNN_MC,
                packB,
                packB2,
                bmap   + jc,
                D      + ic * ldr, // D is m x ldr (d-array heap) 
                I      + ic * ldr, // I is m x ldr (d-array heap)
                ldr
                );
          }
        }                                                  // End 4-th loop
      }                                                    // End 5-th loop
    }                                                      // end 6-th loop
  }


  free( packA );
  free( packB );
  free( packA2 );
  free( packB2 );
}
