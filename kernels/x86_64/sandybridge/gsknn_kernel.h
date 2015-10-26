#ifndef __RNN_KERNEL_H__
#define __RNN_KERNEL_H__

void knn_rank_k_ref_s8x8(
    int    k,
    float  *a,
    float  *b,
    float  *c,
    int    ldc,
    aux_t  *aux
    );

void knn_rank_k_abs_ref_s8x8(
    int    k,
    float  *a,
    float  *b,
    float  *c,
    int    ldc,
    aux_t  *aux
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

void gsknn_heapselect_int_d4(
    int    m,
    int    k,
    double *key,
    int    *val,
    double *D,
    int    *I
    );

void (*kselect)(
    int    m,
    int    k,
    double *key,
    int    *val,
    double *D,
    int    *I
    ) ={
  gsknn_heapselect_int_d4
};

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


#ifdef KNN_PREC_SINGLE
void (*rankk_s[ 2 ]) (
    int    k,
    float  *a,
    float  *b,
    float  *c,
    int    ldc,
    aux_t  *aux
    ) = {
  knn_rank_k_ref_s8x8
};
#else
void (*rankk_d[ 2 ]) (
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
#endif


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

#endif // define __RNN_KERNEL_H__
