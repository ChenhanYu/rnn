#include <stdio.h>
#include <omp.h>
#include <gsknn.h>
#define min( i, j ) ( (i)<(j) ? (i): (j) )

#include <rnn_kernel.h>

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
      ( *rankk[ 0 ] ) (
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
    double *packC,
    int    ldc,
    int    pc,
    double *D,
    int    *I,
    int    ldr
    )
{
  double c[ DRNN_MC * DRNN_NC ] __attribute__((aligned(32)));
  double *cbuff = c;
  int    i, ii, j;
  aux_t  aux;

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
      if ( pc ) {
        cbuff = packC  + j * ldc + i * DRNN_NR;
      }

      // --------------------------------------------------------------------
      // Combine selective square distance and the heap adjustment.
      // --------------------------------------------------------------------
      ( *micro[ 0 ] ) (
          k,
          r,
          packA2 + i,
          packA  + i * k,
          packB2 + j,
          packB  + j * k,
          cbuff,
          &aux,
          bmap   + j
          );
      // --------------------------------------------------------------------
    }
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

  aux.pc     = pc;
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

      //sq2nrm_asm_d8x4(
      ( *sq2nrm ) (
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
  str = getenv( "GSKNN_IC_NT" );
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
    heap_t *heap
    //double *D,
    //int    *I
    )
{
  int    i, j, ldr;
  int    ldc = ( ( m - 1 ) / DRNN_MR + 1 ) * DRNN_MR;
  double beg, time_heap, time_sq2nrm;
  double *D = heap->D;
  int    *I = heap->I;
  double *C = rnn_malloc_aligned( ldc, n + 4, sizeof(double) );

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


  //ldr = I[ 2 ];
  ldr = heap->ldk;

  beg = omp_get_wtime();
  #pragma omp parallel for
  for ( j = 0; j < n; j ++ ) {
    //heap_sort( m, r, &C[ j * ldc ], amap, &D[ j * r ], &I[ j * r ] );
    heapSelect_dheap( m, r, &C[ j * ldc ], amap, &D[ j * ldr ], &I[ j * ldr ] );
  }
  time_heap = omp_get_wtime() - beg;

  //printf( "gsrnn sq2nrm: %5.3lf, heap: %5.3lf\n", time_sq2nrm, time_heap );

  free( C );
}


void dgsrnn_var1(
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
    heap_t *heap
    //double *D,
    //int    *I
    )
{
  int    i, j, p, rnn_ic_nt;
  int    ic, ib, jc, jb, pc, pb;
  int    ir, jr;
  int    ldc, padn, ldr;
  double *packA, *packB, *packC, *packw, *packu, *packA2, *packB2;
  char   *str;
  double *D = heap->D;
  int    *I = heap->I;


  // Early return if possible
  if ( m == 0 || n == 0 || k == 0 ) {
    printf( "dgsrnn(): early return\n" );
    return;
  }


  // sequential is the default situation
  rnn_ic_nt = 1;


  // check the environment variable
  str = getenv( "GSKNN_IC_NT" );
  if ( str != NULL ) {
    rnn_ic_nt = (int)strtol( str, NULL, 10 );
  }


  // D-array heap leading dimenstion.
  //ldr = I[ 2 ];
  //ldr = r;
  ldr = heap->ldk;


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
                &packC[ ic * padn ], // packed
                ( ( ib - 1 ) / DRNN_MR + 1 ) * DRNN_MR, // packed
                pc,
                D      + ic * ldr, // D is m x ldr (d-array heap) 
                I      + ic * ldr, // I is m x ldr (d-array heap)
                ldr
                );
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
              NULL,
              0,
              pc,
              D      + ic * ldr, // D is m x ldr (d-array heap) 
              I      + ic * ldr, // I is m x ldr (d-array heap)
              ldr
              );
        }                                                  // End 4-th loop
      }                                                    // End 5-th loop
    }                                                      // end 6-th loop
  }


  free( packA );
  free( packB );
  free( packA2 );
  free( packB2 );
}


#ifdef KNN_PREC_SINGLE 
void sgsrnn
#else
void dgsrnn
#endif
    (
    int    m,
    int    n,
    int    k,
    int    r,
    prec_t *XA,
    prec_t *XA2,
    int    *amap,
    prec_t *XB,
    prec_t *XB2,
    int    *bmap,
    heap_t *heap
    )
{
  int    i, j;

  if ( r > RNN_VAR_THRES ) {
#ifdef KNN_PREC_SINGLE
    sgsrnn_var3
#else
    dgsrnn_var3
#endif      
      (
        m,
        n,
        k,
        r,
        XA,
        XA2,
        amap,
        XB,
        XB2,
        bmap,
        heap
        );
  }
  else {
#ifdef KNN_PREC_SINGLE
    sgsrnn_var1
#else
    dgsrnn_var1
#endif
      (
        n,
        m,
        k,
        r,
        XB,
        XB2,
        bmap,
        XA,
        XA2,
        amap,
        heap
        );
  }
}
