#include <stdio.h>
#include <omp.h>
#include <gsknn.h>
#define min( i, j ) ( (i)<(j) ? (i): (j) )

#include <gsknn_config.h>
#include <gsknn_kernel.h>



/*
 *
 */
inline void packA_kcxmc_s(
    int    m,
    int    k,
    float  *XA,
    int    ldXA,
    int    *amap,
    float  *packA
    )
{
  int    i, p;
  float *a_pntr[ SKNN_MR ];

  for ( i = 0; i < m; i ++ ) {
    a_pntr[ i ] = XA + ldXA * amap[ i ];
  }

  for ( i = m; i < SKNN_MR; i ++ ) {
    a_pntr[ i ] = XA + ldXA * amap[ 0 ];
  }

  for ( p = 0; p < k; p ++ ) {
    for ( i = 0; i < SKNN_MR; i ++ ) {
      *packA ++ = *a_pntr[ i ] ++;
    }
  }
}


inline void packA_kcxmc_d(
    int    m,
    int    k,
    double *XA,
    int    ldXA,
    int    *amap,
    double *packA
    )
{
  int    i, p;
  double *a_pntr[ DKNN_MR ];

  for ( i = 0; i < m; i ++ ) {
    a_pntr[ i ] = XA + ldXA * amap[ i ];
  }

  for ( i = m; i < DKNN_MR; i ++ ) {
    a_pntr[ i ] = XA + ldXA * amap[ 0 ];
  }

  for ( p = 0; p < k; p ++ ) {
    for ( i = 0; i < DKNN_MR; i ++ ) {
      *packA ++ = *a_pntr[ i ] ++;
    }
  }
}


/*
 *
 */ 
inline void packB_kcxnc_s(
    int    n,
    int    k,
    float  *XB,
    int    ldXB, // ldXB is the original k
    int    *bmap,
    float  *packB
    )
{
  int    j, p; 
  float  *b_pntr[ SKNN_NR ];

  for ( j = 0; j < n; j ++ ) {
    b_pntr[ j ] = XB + ldXB * bmap[ j ];
  }

  for ( j = n; j < SKNN_NR; j ++ ) {
    b_pntr[ j ] = XB + ldXB * bmap[ 0 ];
  }

  for ( p = 0; p < k; p ++ ) {
    for ( j = 0; j < SKNN_NR; j ++ ) {
      *packB ++ = *b_pntr[ j ] ++;
    }
  }
}


inline void packB_kcxnc_d(
    int    n,
    int    k,
    double *XB,
    int    ldXB, // ldXB is the original k
    int    *bmap,
    double *packB
    )
{
  int    j, p; 
  double *b_pntr[ DKNN_NR ];

  for ( j = 0; j < n; j ++ ) {
    b_pntr[ j ] = XB + ldXB * bmap[ j ];
  }

  for ( j = n; j < DKNN_NR; j ++ ) {
    b_pntr[ j ] = XB + ldXB * bmap[ 0 ];
  }

  for ( p = 0; p < k; p ++ ) {
    for ( j = 0; j < DKNN_NR; j ++ ) {
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
void rank_k_macro_kernel_s(
    int    m,
    int    n,
    int    k,
    float  *packA,
    float  *packB,
    float  *packC,
    int    ldc,
    int    pc
    )
{
  //printf( "rank_k_macro_kernel_s(): Not implemented yet.\n" );
  int    i, j;
  aux_t  aux;

  aux.pc       = pc;
  aux.b_next_s = packB;

  for ( j = 0; j < n; j += SKNN_NR ) {
    for ( i = 0; i < m; i += SKNN_MR ) {
      if ( i + SKNN_MR >= m ) {
        aux.b_next_s += SKNN_NR * k;
      }
      ( *rankk_s[ 0 ] )(
          k,
          &packA[ i * k ],
          &packB[ j * k ],
          &packC[ j * ldc + i * SKNN_NR ], // packed
          ldc,
          &aux
          );
    }
  }
}


void rank_k_macro_kernel_d(
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

  for ( j = 0; j < n; j += DKNN_NR ) {
    for ( i = 0; i < m; i += DKNN_MR ) {
      if ( i + DKNN_MR >= m ) {
        aux.b_next += DKNN_NR * k;
      }
      ( *rankk_d[ 0 ] )(
          k,
          &packA[ i * k ],
          &packB[ j * k ],
          &packC[ j * ldc + i * DKNN_NR ], // packed
          ldc,
          &aux
          );
    }
  }
}



/*
 *
 *
 */ 
void sgsknn_macro_kernel_row(
    int    m,
    int    n,
    int    k,
    int    r,
    float  *packA,
    float  *packA2,
    float  *packB,
    float  *packB2,
    int    *bmap,
    float  *packC,
    int    ldc,
    int    pc,
    float  *D,
    int    *I,
    int    ldr
    )
{
  //printf( "sgsknn_macro_kernel_row(): Not implemented yet.\n");
  float  c[ SKNN_MC * SKNN_NC ] __attribute__((aligned(32)));
  float  *cbuff = c;
  int    i, ii, j;
  aux_t  aux;

  aux.pc       = pc;
  aux.b_next_s = packB;
  aux.ldr      = ldr;

  for ( j = 0; j < n; j += SKNN_NR ) {
    aux.n  = min( n - j, SKNN_NR );
    for ( i = 0; i < m; i += SKNN_MR ) {
      aux.m = min( m - i, SKNN_MR );
      aux.I   = I + i * ldr;
      aux.D_s = D + i * ldr;
      if ( i + SKNN_MR >= m ) {
        aux.b_next_s += SKNN_NR * k;
      }
      if ( pc ) {
        cbuff = packC  + j * ldc + i * SKNN_NR;
      }

      ( *micro_s[ 0 ] ) (
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
    }
  }
}


void dgsknn_macro_kernel_row(
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
  double c[ DKNN_MC * DKNN_NC ] __attribute__((aligned(32)));
  double *cbuff = c;
  int    i, ii, j;
  aux_t  aux;

  aux.pc     = pc;
  aux.b_next = packB;
  aux.ldr    = ldr;


  for ( j = 0; j < n; j += DKNN_NR ) {
    aux.n  = min( n - j, DKNN_NR );
    for ( i = 0; i < m; i += DKNN_MR ) {
      aux.m = min( m - i, DKNN_MR );
      aux.I = I + i * ldr;
      aux.D = D + i * ldr;
      if ( i + DKNN_MR >= m ) {
        aux.b_next += DKNN_NR * k;
      }
      if ( pc ) {
        cbuff = packC  + j * ldc + i * DKNN_NR;
      }

      // --------------------------------------------------------------------
      // Combine selective square distance and the heap adjustment.
      // --------------------------------------------------------------------
      ( *micro_d[ 0 ] ) (
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

  for ( j = 0; j < n; j += DKNN_NR ) {
    aux.n  = min( n - j, DKNN_NR );
    for ( i = 0; i < m; i += DKNN_MR ) {
      aux.m = min( m - i, DKNN_MR );
      if ( i + DKNN_MR >= m ) {
        aux.b_next += DKNN_NR * k;
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
  int    i, j, p, gsknn_ic_nt;
  int    ic, ib, jc, jb, pc, pb;
  int    ir, jr;
  double *packA, *packB, *packA2, *packB2;
  char   *str;


  // Early return if possible
  if ( m == 0 || n == 0 || k == 0 ) {
    printf( "dgsknn(): early return\n" );
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
  packB  = gsknn_malloc_aligned( DKNN_KC, ( DKNN_NC + 1 )            , sizeof(double) );
  packA2 = gsknn_malloc_aligned(       1, ( DKNN_MC + 1 ) * gsknn_ic_nt, sizeof(double) );
  packB2 = gsknn_malloc_aligned(       1, ( DKNN_NC + 1 )            , sizeof(double) );


  for ( jc = 0; jc < n; jc += DKNN_NC ) {                  // 6-th loop
    jb = min( n - jc, DKNN_NC );
    for ( pc = 0; pc < k; pc += DKNN_KC ) {                // 5-th loop
      pb = min( k - pc, DKNN_KC );


      #pragma omp parallel for num_threads( gsknn_ic_nt ) private( jr )
      for ( j = 0; j < jb; j += DKNN_NR ) {
        if ( pc + DKNN_KC >= k ) {
          for ( jr = 0; jr < min( jb - j, DKNN_NR ); jr ++ ) {
            packB2[ j + jr ] = XB2[ bmap[ jc + j + jr ] ];
          }
        }
        packB_kcxnc_d(
            min( jb - j, DKNN_NR ),
            pb,
            &XB[ pc ],
            k, // should be ldXB instead
            &bmap[ jc + j ],
            &packB[ j * pb ]
            );
      }

      #pragma omp parallel for num_threads( gsknn_ic_nt ) private( ic, ib, i, ir )
      for ( ic = 0; ic < m; ic += DKNN_MC ) {              // 4-th loop
        int     tid = omp_get_thread_num();

        ib = min( m - ic, DKNN_MC );
        for ( i = 0; i < ib; i += DKNN_MR ) {
          if ( pc + DKNN_KC >= k ) {
            for ( ir = 0; ir < min( ib - i, DKNN_MR ); ir ++ ) {
              packA2[ tid * DKNN_MC + i + ir ] = XA2[ amap[ ic + i + ir ] ];
            }
          }
          packA_kcxmc_d(
              min( ib - i, DKNN_MR ),
              pb,
              &XA[ pc ],
              k,
              &amap[ ic + i ],
              &packA[ tid * DKNN_MC * pb + i * pb ]
              );
        }


        dsq2nrm_macro_kernel(
            ib,
            jb,
            pb,
            packA  + tid * DKNN_MC * pb,
            packA2 + tid * DKNN_MC,
            packB,
            packB2,
            &C[ jc * ldc + ic ], 
            ldc,
            pc,
            ( pc + DKNN_KC >= k )
            );

      }                                                    // End 4.th loop
    }                                                      // End 5.th loop
  }                                                        // End 6.th loop

  free( packA );
  free( packB );
  free( packA2 );
  free( packB2 );
}


void dgsknn_var3(
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
  int    ldc = ( ( m - 1 ) / DKNN_MR + 1 ) * DKNN_MR;
  double beg, time_heap, time_sq2nrm;
  double *D = heap->D;
  int    *I = heap->I;
  double *C = gsknn_malloc_aligned( ldc, n + 4, sizeof(double) );

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
    //heapSelect_dheap( m, r, &C[ j * ldc ], amap, &D[ j * ldr ], &I[ j * ldr ] );
    ( *kselect ) (
        m, 
        r, 
        &C[ j * ldc ], 
        amap, 
        &D[ j * ldr ], 
        &I[ j * ldr ] 
        );
  }
  time_heap = omp_get_wtime() - beg;

  //printf( "gsknn sq2nrm: %5.3lf, heap: %5.3lf\n", time_sq2nrm, time_heap );

  free( C );
}


/*
 *
 */ 
void sgsknn_var1(
    int    m,
    int    n,
    int    k,
    int    r,
    float  *XA,
    float  *XA2,
    int    *amap,
    float  *XB,
    float  *XB2,
    int    *bmap,
    heap_t *heap
    )
{
  //printf( "sgsknn_var1(): Not implemented yet.\n" );
  int    i, j, p, gsknn_ic_nt;
  int    ic, ib, jc, jb, pc, pb;
  int    ir, jr;
  int    ldc, padn, ldr;
  float  *packA, *packB, *packC, *packw, *packu, *packA2, *packB2;
  char   *str;
  float   *D = heap->D_s;
  int    *I = heap->I;

  // Early return if possible
  if ( m == 0 || n == 0 || k == 0 ) {
    printf( "sgsknn_var1(): early return\n" );
    return;
  }


  // sequential is the default situation
  gsknn_ic_nt = 1;


  // check the environment variable
  str = getenv( "GSKNN_IC_NT" );
  if ( str != NULL ) {
    gsknn_ic_nt = (int)strtol( str, NULL, 10 );
  }


  // D-array heap leading dimenstion.
  ldr = heap->ldk;


  // Allocate packing buffers
  packA  = (float*)gsknn_malloc_aligned( SKNN_KC, ( SKNN_MC + 1 ) * gsknn_ic_nt, sizeof(float) );
  packB  = (float*)gsknn_malloc_aligned( SKNN_KC, ( SKNN_NC + 1 )              , sizeof(float) );
  packA2 = (float*)gsknn_malloc_aligned(       1, ( SKNN_MC + 1 ) * gsknn_ic_nt, sizeof(float) );
  packB2 = (float*)gsknn_malloc_aligned(       1, ( SKNN_NC + 1 )              , sizeof(float) );

  if ( k > SKNN_KC ) {
    ldc  = ( ( m - 1 ) / SKNN_MR + 1 ) * SKNN_MR;
    padn = SKNN_NC;
    if ( n < SKNN_NC ) {
      padn = ( ( n - 1 ) / SKNN_NR + 1 ) * SKNN_NR;
    }


    packC = (float*)gsknn_malloc_aligned( ldc, padn, sizeof(float) );

    for ( jc = 0; jc < n; jc += SKNN_NC ) {           // 6-th loop
      jb = min( n - jc, SKNN_NC );
      for ( pc = 0; pc < k; pc += SKNN_KC ) {         // 5-th loop
        pb = min( k - pc, SKNN_KC );

        // packB, packw, packbb
        #pragma omp parallel for num_threads( gsknn_ic_nt ) private( jr )
        for ( j = 0; j < jb; j += SKNN_NR ) {
          if ( pc + SKNN_KC >= k ) {
            // packw and packB2
            for ( jr = 0; jr < min( jb - j, SKNN_NR ); jr ++ ) {
                packB2[ j + jr ] = XB2[ bmap[ jc + j + jr ] ];
            }
          }
          packB_kcxnc_s(
              min( jb - j, SKNN_NR ),
              pb,
              &XB[ pc ],
              k, // should be ldXB instead
              &bmap[ jc + j ],
              &packB[ j * pb ]
              );
        }

        #pragma omp parallel for num_threads( gsknn_ic_nt ) private( ic, ib, i, ir )
        for ( ic = 0; ic < m; ic += SKNN_MC ) {       // 4-th loop
          //int     tid = 0;
          int     tid = omp_get_thread_num();

          ib = min( m - ic, SKNN_MC );
          for ( i = 0; i < ib; i += SKNN_MR ) {
            if ( pc + SKNN_KC >= k ) {
              for ( ir = 0; ir < min( ib - i, SKNN_MR ); ir ++ ) {
                packA2[ tid * SKNN_MC + i + ir ] = XA2[ amap[ ic + i + ir ] ];
              }
            }
            packA_kcxmc_s(
                min( ib - i, SKNN_MR ),
                pb,
                &XA[ pc ],
                k,
                &amap[ ic + i ],
                &packA[ tid * SKNN_MC * pb + i * pb ]
                );
          }

          // Check if this is the last kc interation
          if ( pc + SKNN_KC < k ) {
            rank_k_macro_kernel_s(
                ib,
                jb,
                pb,
                packA + tid * SKNN_MC * pb,
                packB,
                &packC[ ic * padn ], // packed
                ( ( ib - 1 ) / SKNN_MR + 1 ) * SKNN_MR, // packed
                pc
                );
          }
          else {
            sgsknn_macro_kernel_row(                      // 1~3 loops
                ib,
                jb,
                pb,
                r,
                packA  + tid * SKNN_MC * pb,
                packA2 + tid * SKNN_MC,
                packB,
                packB2,
                bmap   + jc,
                &packC[ ic * padn ], // packed
                ( ( ib - 1 ) / SKNN_MR + 1 ) * SKNN_MR, // packed
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

    for ( jc = 0; jc < n; jc += SKNN_NC ) {                // 6-th loop
      jb = min( n - jc, SKNN_NC );
      for ( pc = 0; pc < k; pc += SKNN_KC ) {              // 5-th loop
        pb = min( k - pc, SKNN_KC );

        #pragma omp parallel for num_threads( gsknn_ic_nt ) private( jr )
        for ( j = 0; j < jb; j += SKNN_NR ) {
          for ( jr = 0; jr < min( jb - j, SKNN_NR ); jr ++ ) {
            packB2[ j + jr ] = XB2[ bmap[ jc + j + jr ] ];
          }
          packB_kcxnc_s(
              min( jb - j, SKNN_NR ),
              pb,
              XB,
              k, // should be ldXB instead
              &bmap[ jc + j ],
              &packB[ j * pb ]
              );
        }

        #pragma omp parallel for num_threads( gsknn_ic_nt ) private( ic, ib, i, ir )
        for ( ic = 0; ic < m; ic += SKNN_MC ) {            // 4-th loop
          int     tid = omp_get_thread_num();

          ib = min( m - ic, SKNN_MC );
          for ( i = 0; i < ib; i += SKNN_MR ) {
            for ( ir = 0; ir < min( ib - i, SKNN_MR ); ir ++ ) {
              packA2[ tid * SKNN_MC + i + ir ] = XA2[ amap[ ic + i + ir ] ];
            }
            packA_kcxmc_s(
                min( ib - i, SKNN_MR ),
                pb,
                XA,
                k,
                &amap[ ic + i ],
                &packA[ tid * SKNN_MC * pb + i * pb ]
                );
          }

          sgsknn_macro_kernel_row(                      // 1~3 loops
              ib,
              jb,
              pb,
              r,
              packA  + tid * SKNN_MC * pb,
              packA2 + tid * SKNN_MC,
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


void dgsknn_var1(
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
  int    i, j, p, gsknn_ic_nt;
  int    ic, ib, jc, jb, pc, pb;
  int    ir, jr;
  int    ldc, padn, ldr;
  double *packA, *packB, *packC, *packw, *packu, *packA2, *packB2;
  char   *str;
  double *D = heap->D;
  int    *I = heap->I;


  // Early return if possible
  if ( m == 0 || n == 0 || k == 0 ) {
    printf( "dgsknn(): early return\n" );
    return;
  }


  // sequential is the default situation
  gsknn_ic_nt = 1;


  // check the environment variable
  str = getenv( "GSKNN_IC_NT" );
  if ( str != NULL ) {
    gsknn_ic_nt = (int)strtol( str, NULL, 10 );
  }


  // D-array heap leading dimenstion.
  //ldr = I[ 2 ];
  //ldr = r;
  ldr = heap->ldk;


  // Allocate packing buffers
  packA  = gsknn_malloc_aligned( DKNN_KC, ( DKNN_MC + 1 ) * gsknn_ic_nt, sizeof(double) );
  packB  = gsknn_malloc_aligned( DKNN_KC, ( DKNN_NC + 1 )            , sizeof(double) );
  packA2 = gsknn_malloc_aligned(       1, ( DKNN_MC + 1 ) * gsknn_ic_nt, sizeof(double) );
  packB2 = gsknn_malloc_aligned(       1, ( DKNN_NC + 1 )            , sizeof(double) );


  if ( k > DKNN_KC ) {
    ldc  = ( ( m - 1 ) / DKNN_MR + 1 ) * DKNN_MR;
    padn = DKNN_NC;
    if ( n < DKNN_NC ) {
      padn = ( ( n - 1 ) / DKNN_NR + 1 ) * DKNN_NR;
    }


    packC  = gsknn_malloc_aligned( ldc, padn, sizeof(double) );


    for ( jc = 0; jc < n; jc += DKNN_NC ) {           // 6-th loop
      jb = min( n - jc, DKNN_NC );
      for ( pc = 0; pc < k; pc += DKNN_KC ) {         // 5-th loop
        pb = min( k - pc, DKNN_KC );

        // packB, packw, packbb
        #pragma omp parallel for num_threads( gsknn_ic_nt ) private( jr )
        for ( j = 0; j < jb; j += DKNN_NR ) {
          if ( pc + DKNN_KC >= k ) {
            // packw and packB2
            for ( jr = 0; jr < min( jb - j, DKNN_NR ); jr ++ ) {
                packB2[ j + jr ] = XB2[ bmap[ jc + j + jr ] ];
            }
          }
          packB_kcxnc_d(
              min( jb - j, DKNN_NR ),
              pb,
              &XB[ pc ],
              k, // should be ldXB instead
              &bmap[ jc + j ],
              &packB[ j * pb ]
              );
        }

        #pragma omp parallel for num_threads( gsknn_ic_nt ) private( ic, ib, i, ir )
        for ( ic = 0; ic < m; ic += DKNN_MC ) {       // 4-th loop
          //int     tid = 0;
          int     tid = omp_get_thread_num();

          ib = min( m - ic, DKNN_MC );
          for ( i = 0; i < ib; i += DKNN_MR ) {
            if ( pc + DKNN_KC >= k ) {
              for ( ir = 0; ir < min( ib - i, DKNN_MR ); ir ++ ) {
                packA2[ tid * DKNN_MC + i + ir ] = XA2[ amap[ ic + i + ir ] ];
              }
            }
            packA_kcxmc_d(
                min( ib - i, DKNN_MR ),
                pb,
                &XA[ pc ],
                k,
                &amap[ ic + i ],
                &packA[ tid * DKNN_MC * pb + i * pb ]
                );
          }

          // Check if this is the last kc interation
          if ( pc + DKNN_KC < k ) {
            rank_k_macro_kernel_d(
                ib,
                jb,
                pb,
                packA + tid * DKNN_MC * pb,
                packB,
                &packC[ ic * padn ], // packed
                ( ( ib - 1 ) / DKNN_MR + 1 ) * DKNN_MR, // packed
                pc
                );
          }
          else {
            dgsknn_macro_kernel_row(                      // 1~3 loops
                ib,
                jb,
                pb,
                r,
                packA  + tid * DKNN_MC * pb,
                packA2 + tid * DKNN_MC,
                packB,
                packB2,
                bmap   + jc,
                &packC[ ic * padn ], // packed
                ( ( ib - 1 ) / DKNN_MR + 1 ) * DKNN_MR, // packed
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

    for ( jc = 0; jc < n; jc += DKNN_NC ) {                // 6-th loop
      jb = min( n - jc, DKNN_NC );
      for ( pc = 0; pc < k; pc += DKNN_KC ) {              // 5-th loop
        pb = min( k - pc, DKNN_KC );

        #pragma omp parallel for num_threads( gsknn_ic_nt ) private( jr )
        for ( j = 0; j < jb; j += DKNN_NR ) {
          for ( jr = 0; jr < min( jb - j, DKNN_NR ); jr ++ ) {
            packB2[ j + jr ] = XB2[ bmap[ jc + j + jr ] ];
          }
          packB_kcxnc_d(
              min( jb - j, DKNN_NR ),
              pb,
              XB,
              k, // should be ldXB instead
              &bmap[ jc + j ],
              &packB[ j * pb ]
              );
        }

        #pragma omp parallel for num_threads( gsknn_ic_nt ) private( ic, ib, i, ir )
        for ( ic = 0; ic < m; ic += DKNN_MC ) {            // 4-th loop
          int     tid = omp_get_thread_num();

          ib = min( m - ic, DKNN_MC );
          for ( i = 0; i < ib; i += DKNN_MR ) {
            for ( ir = 0; ir < min( ib - i, DKNN_MR ); ir ++ ) {
              packA2[ tid * DKNN_MC + i + ir ] = XA2[ amap[ ic + i + ir ] ];
            }
            packA_kcxmc_d(
                min( ib - i, DKNN_MR ),
                pb,
                XA,
                k,
                &amap[ ic + i ],
                &packA[ tid * DKNN_MC * pb + i * pb ]
                );
          }

          dgsknn_macro_kernel_row(                      // 1~3 loops
              ib,
              jb,
              pb,
              r,
              packA  + tid * DKNN_MC * pb,
              packA2 + tid * DKNN_MC,
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


void sgsknn(
    int    m,
    int    n,
    int    k,
    int    r,
    float  *XA,
    float  *XA2,
    int    *amap,
    float  *XB,
    float  *XB2,
    int    *bmap,
    heap_t *heap
    )
{
  sgsknn_var1(
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


void dgsknn(
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
    )
{
  if ( r > KNN_VAR_THRES ) {
    dgsknn_var3(
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
    dgsknn_var1(
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
