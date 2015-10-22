#include <math.h>
#include <immintrin.h>

#ifdef KNN_PREC_SINGLE
#define prec_t float
#else
#define prec_t double
#endif

#define RNN_NUM_THD_MC 1
#define RNN_VAR_THRES 512

#define DRNN_SIMD_ALIGN_SIZE 32
#define DRNN_MC 104
#define DRNN_NC 2048
#define DRNN_KC 256
#define DRNN_MR 8
#define DRNN_NR 4

typedef unsigned long long dim_t;

typedef enum {
  RNN_2NORM,
  RNN_1NORM
} rnn_type;

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
  rnn_type type;
};

typedef struct heap_s heap_t;

void dgsrnn(
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

void dgsrnn_var1(
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

double *rnn_malloc_aligned(
    int    m,
    int    n,
    int    size
    );

heap_t *rnn_heapCreate(
    int    m,
    int    k,
    double ro
    );

heap_t *rnn_heapAttach(
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
    double *x, 
    int    *alpha, 
    double *D,
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
