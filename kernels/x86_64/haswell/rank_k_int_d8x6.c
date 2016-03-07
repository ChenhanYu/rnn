/*
 * This file is modified and redistribued from 
 * 
 * BLIS
 * An object-based framework for developing high-performance BLAS-like
 * libraries.
 *
 * Copyright (C) 2014, The University of Texas at Austin
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met:
 *  - Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  - Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the
 *    distribution.
 *  - Neither the name of The University of Texas at Austin nor the names
 *    of its contributors may be used to endorse or promote products
 *    derived from this software without specific prior written
 *    permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 *
 *
 * rank_k_asm_d8x6.c
 * 
 * Mofidifier:
 * Chenhan D. Yu - Department of Computer Science, 
 *                 The University of Texas at Austin
 *
 *
 * Purpose: 
 *
 *
 * Todo:
 *
 *
 * Modification:
 * 
 *
 *
 * */

#include <stdio.h>
#include <immintrin.h> // AVX
//#include <gsknn.h>
#include <gsknn_internal.h>
#include <avx_types.h>

void rank_k_asm_s16x6(
    int    k,
    float  *a,
    float  *b,
    float  *c,
    int    ldc,
    aux_t  *aux
    )
{
  printf( "rank_k_asm_s16x6 not yet implemented.\n" );
}

void rank_k_int_d8x6(
    int    k,
    double *a,
    double *b,
    double *c,
    int    ldc,
    aux_t  *aux
    )
{
  int    i;
  // 16 registers.
  v4df_t c03_0, c03_1, c03_2, c03_3, c03_4, c03_5;
  v4df_t c47_0, c47_1, c47_2, c47_3, c47_4, c47_5;
  v4df_t a03, a47, b0, b1;

  #include <rank_k_int_d8x6.h>

  // Store c
  _mm256_store_pd( c      , c03_0.v );
  _mm256_store_pd( c +  4 , c47_0.v );

  _mm256_store_pd( c +  8 , c03_1.v );
  _mm256_store_pd( c + 12 , c47_1.v );

  _mm256_store_pd( c + 16 , c03_2.v );
  _mm256_store_pd( c + 20 , c47_2.v );

  _mm256_store_pd( c + 24 , c03_3.v );
  _mm256_store_pd( c + 28 , c47_3.v );

  _mm256_store_pd( c + 32 , c03_4.v );
  _mm256_store_pd( c + 36 , c47_4.v );

  _mm256_store_pd( c + 40 , c03_5.v );
  _mm256_store_pd( c + 44 , c47_5.v );
}
