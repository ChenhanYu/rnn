#include <math.h>
#include <immintrin.h> // AVX

#define RNN_NUM_THD_MC 1
#define RNN_VAR_THRES 512

#define DRNN_SIMD_ALIGN_SIZE 32
#define DRNN_MC 104
#define DRNN_NC 2048
#define DRNN_KC 256
#define DRNN_MR 8
#define DRNN_NR 4

typedef unsigned long long dim_t;

typedef union {
  __m256d v;
  __m256i u;
  double d[ 4 ];
} v4df_t;


typedef union {
  __m128i v;
  int d[ 4 ];
} v4li_t;

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

void dgsrnn_var2(
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
    double *D,
    int    *I
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
    double *D,
    int    *I
    );

void dgsrnn_ref(
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
    double *D,
    int    *I
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

void HeapAdjust(
    double *D, 
    int    s, 
    int    n, 
    int    *I
    );

void heap_sort(
    int    m,
    int    r,
    double *x, 
    int    *alpha, 
    double *D,
    int    *I
    );
