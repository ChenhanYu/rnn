#ifndef __RNN_KERNEL_H__
#define __RNN_KERNEL_H__


// Single precision
void knn_rank_k_ref_s4x4(
    int    k,
    float  *a,
    float  *b,
    float  *c,
    int    ldc,
    aux_t  *aux
    );

void knn_rank_k_int_s4x4(
    int    k,
    float  *a,
    float  *b,
    float  *c,
    int    ldc,
    aux_t  *aux
    );

void knn_rank_k_abs_ref_s4x4(
    int    k,
    float  *a,
    float  *b,
    float  *c,
    int    ldc,
    aux_t  *aux
    );

void knn_r_ref_s4x4_row(
    int    k,
    int    r,
    float  *aa,
    float  *a,
    float  *bb,
    float  *b,
    float  *c,
    aux_t  *aux,
    int    *bmap
    );

void knn_r_int_s4x4_row(
    int    k,
    int    r,
    float  *aa,
    float  *a,
    float  *bb,
    float  *b,
    float  *c,
    aux_t  *aux,
    int    *bmap
    );

void knn_r_abs_ref_s4x4_row(
    int    k,
    int    r,
    float  *aa,
    float  *a,
    float  *bb,
    float  *b,
    float  *c,
    aux_t  *aux,
    int    *bmap
    );



// Double precision
void knn_rank_k_ref_d4x4(
    int    k,
    double* a,
    double* b,
    double* c,
    int    ldc,
    aux_t  *aux
    );

void knn_rank_k_abs_ref_d4x4(
    int    k,
    double *a,
    double *b,
    double *c,
    int    ldc,
    aux_t  *aux
    );

void knn_r_ref_d4x4_row(
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

void knn_r_abs_ref_d4x4_row(
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



// Not yet implemented
void sq2nrm_ref_d4x4(
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

void knn_heapselect_ref_d4(
    int    m,
    int    k,
    double *key,
    int    *val,
    double *D,
    int    *I
    );

void (*kselect[ 1 ])(
    int    m,
    int    k,
    double *key,
    int    *val,
    double *D,
    int    *I
    ) ={
  knn_heapselect_ref_d4
};

void (*micro_s[ 2 ]) (
    int    k,
    int    r,
    float  *aa,
    float  *a,
    float  *bb,
    float  *b,
    float  *c,
    aux_t  *aux,
    int    *bmap
    ) = {
  knn_r_int_s4x4_row,
  //knn_r_ref_s4x4_row,
  knn_r_abs_ref_s4x4_row,
};

void (*micro_d[ 2 ]) (
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
  knn_r_ref_d4x4_row,
  knn_r_abs_ref_d4x4_row
};

void (*rankk_s[ 2 ]) (
    int    k,
    float  *a,
    float  *b,
    float  *c,
    int    ldc,
    aux_t  *aux
    ) = {
  knn_rank_k_int_s4x4,
  //knn_rank_k_ref_s4x4,
  knn_rank_k_abs_ref_s4x4
};

void (*rankk_d[ 2 ]) (
    int    k,
    double *a,
    double *b,
    double *c,
    int    ldc,
    aux_t  *aux
    ) = {
  knn_rank_k_ref_d4x4,
  knn_rank_k_abs_ref_d4x4
};



// Not yet implemented
void (*sq2nrm[ 1 ]) (
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
  sq2nrm_ref_d4x4
};

#endif // define __RNN_KERNEL_H__
