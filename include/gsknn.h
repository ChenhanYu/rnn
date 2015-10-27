#include <math.h>
#include <immintrin.h>

#ifdef KNN_PREC_SINGLE
#define prec_t float
#else
#define prec_t double
#endif

#define KNN_VAR_THRES 512
#define KNN_HEAP_OFFSET 3

typedef unsigned long long dim_t;

typedef enum {
  KNN_2NORM,
  KNN_1NORM
} knn_type;

struct aux_s {
  double *b_next;
  int    *I;
  double *D;
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
  int    *I;
  knn_type type;
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
    heap_t *heap
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
    heap_t *heap
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

heap_t *gsknn_heapCreate(
    int    m,
    int    k,
    double ro
    );

heap_t *gsknn_heapAttach(
    int    m,
    int    k,
    double *D,
    int    *I
    );

void HeapAdjust_s(
    float  *D, 
    int    s, 
    int    n, 
    int    *I
    );

void HeapAdjust_d(
    double *D, 
    int    s, 
    int    n, 
    int    *I
    );

void heap_sort_s(
    int    m,
    int    r,
    float  *x, 
    int    *alpha, 
    float  *D,
    int    *I
    );

void heap_sort_d(
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
