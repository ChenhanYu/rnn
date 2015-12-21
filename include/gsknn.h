/*
 * --------------------------------------------------------------------------
 * GSKNN (General Stride K-Nearest Neighbors)
 * --------------------------------------------------------------------------
 * Copyright (C) 2015, The University of Texas at Austin
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 *
 *
 * gsknn.h
 *
 * Chenhan D. Yu - Department of Computer Science,
 *                 The University of Texas at Austin
 *
 *
 * Purpose:
 * this header file contains all function prototypes.
 *
 * Todo:
 *
 *
 * Modification:
 *
 * 
 * */

#ifdef __cplusplus
extern "C" {
#endif

#include <math.h>
#include <immintrin.h>

#define KNN_VAR_THRES 512
#define KNN_HEAP_OFFSET 3

typedef unsigned long long dim_t;

typedef enum {
  KNN_2NORM,
  KNN_1NORM
} knn_type;

typedef enum {
  KNN_DOUBLE,
  KNN_SINGLE
} knn_prec;

struct aux_s {
  double *b_next;
  float  *b_next_s;
  int    *I;
  double *D;
  float  *D_s;
  int    ldr;
  char   *flag;
  int    pc;
  int    m;
  int    n;
};
typedef struct aux_s aux_t;

struct heap_s {
  int    m;
  int    k;
  int    ldk;
  int    d;
  double ro;
  double *D;
  float  ro_s;
  float  *D_s;
  int    *I;
  knn_type type;
  knn_prec prec;
};
typedef struct heap_s heap_t;

void sgsknn(
    int    m,
    int    n,
    int    k,
    int    r,
    float  *XA,
    float  *XA2,
    int    *alpha,
    float  *XB,
    float  *XB2,
    int    *beta,
    heap_t *heap
    );

void dgsknn(
    int    m,
    int    n,
    int    k,
    int    r,
    double *XA,
    double *XA2,
    int    *alpha,
    double *XB,
    double *XB2,
    int    *beta,
    heap_t *heap
    );

void sgsknn_var1(
    int    m,
    int    n,
    int    k,
    int    r,
    float  *XA,
    float  *XA2,
    int    *alpha,
    float  *XB,
    float  *XB2,
    int    *beta,
    //heap_t *heap
    float  *D,
    int    *I
    );

void dgsknn_var1(
    int    m,
    int    n,
    int    k,
    int    r,
    double *XA,
    double *XA2,
    int    *alpha,
    double *XB,
    double *XB2,
    int    *beta,
    //heap_t *heap
    double *D,
    int    *I
    );

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
    );

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
    double *C,
    int    ldc
);

double *gsknn_malloc_aligned(
    int    m,
    int    n,
    int    size
    );

heap_t *heapCreate_s(
    int    m,
    int    k,
    float  ro
    );

heap_t *heapCreate_d(
    int    m,
    int    k,
    double ro
    );

void heapFree_s(
    heap_t *heap
    );

void heapFree_d(
    heap_t *heap
    );

heap_t *heapAttach_s(
    int    m,
    int    k,
    float  *D,
    int    *I
    );

heap_t *heapAttach_d(
    int    m,
    int    k,
    double *D,
    int    *I
    );

void heapSelect_s(
    int    m,
    int    r,
    float  *x, 
    int    *alpha, 
    float  *D,
    int    *I
    );

void heapSelect_d(
    int    m,
    int    r,
    double *x, 
    int    *alpha, 
    double *D,
    int    *I
    );

void heapSelect_dheap(
    int    m,
    int    k,
    double *key,
    int    *val,
    double *D,
    int    *I
    );

void swap_float( 
    float  *x, 
    int    i, 
    int    j 
    );

void swap_double( 
    double *x, 
    int    i, 
    int    j 
    );

void swap_int( 
    int    *I, 
    int    i, 
    int    j 
    );

void bubbleSort_s(
    int    n,
    float  *D,
    int    *I
    );

void bubbleSort_d(
    int    n,
    double *D,
    int    *I
    );

#ifdef __cplusplus
}
#endif
