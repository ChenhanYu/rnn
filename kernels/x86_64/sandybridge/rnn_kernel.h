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

void rnn_rank_k_asm_d8x4(
    int    k,
    double* a,
    double* b,
    double* c,
    int    ldc,
    aux_t  *aux
    );

void rnn_rank_k_abs_int_d8x4(
    int    k,
    double *a,
    double *b,
    double *c,
    int    ldc,
    aux_t  *aux
    );

void rnn_r_int_d8x4_row(
    int    k,
    int    r,
    double *aa,
    double *a,
    double *bb,
    double *b,
    double *c,
    aux_t  *aux,
    int    *bmap
    );

void rnn_r_1norm_int_d8x4_row(
    int    k,
    int    r,
    double *aa,
    double *a,
    double *bb,
    double *b,
    double *c,
    aux_t  *aux,
    int    *bmap
    );

void sq2nrm_asm_d8x4(
    int    k,
    double *a,
    double *aa,
    double *b,
    double *bb,
    double *c,
    unsigned long long ldc,
    unsigned long long last,
    aux_t  *aux
    );

void (*micro[ 2 ]) (
    int    k,
    int    r,
    double *aa,
    double *a,
    double *bb,
    double *b,
    double *c,
    aux_t  *aux,
    int    *bmap
    ) = {
  rnn_r_int_d8x4_row,
  rnn_r_1norm_int_d8x4_row
};

void (*rankk[ 2 ]) (
    int    k,
    double *a,
    double *b,
    double *c,
    int    ldc,
    aux_t  *aux
    ) = {
  rnn_rank_k_asm_d8x4,
  rnn_rank_k_abs_int_d8x4
};

void (*sq2nrm) (
    int    k,
    double *a,
    double *aa,
    double *b,
    double *bb,
    double *c,
    unsigned long long ldc,
    unsigned long long last,
    aux_t  *aux
    ) = {
  sq2nrm_asm_d8x4
};
