#ifndef __RNN_KERNEL_H__
#define __RNN_KERNEL_H__

#define KERNEL1(name,type) \
  name(                    \
	int    k,              \
	type   *a,             \
	type   *b,             \
	type   *c,             \
	int    ldc,            \
	aux_t  *aux            \
	)

// Float rank-k update prototype
void KERNEL1(knn_rank_k_ref_s8x8,float);
void KERNEL1(knn_rank_k_abs_ref_s8x8,float);

// Double rank-k update prototype
void KERNEL1(rnn_rank_k_asm_d8x4,double);
void KERNEL1(rnn_rank_k_abs_int_d8x4,double);

#define KERNEL2(name,type) \
  name(                    \
    int    k,              \
    int    r,              \
    type   *aa,            \
    type   *a,             \
    type   *bb,            \
    type   *b,             \
    type   *c,             \
    aux_t  *aux,           \
    int    *bmap           \
	)

// Float knn prototype
void KERNEL2(knn_r_ref_s8x8_row,float);

// Double knn prototype
void KERNEL2(rnn_r_int_d8x4_row,double);
void KERNEL2(rnn_r_1norm_int_d8x4_row,double);

#define KERNEL3(name,type) \
  name(                    \
    int    m,              \
    int    k,              \
    type   *key,           \
    int    *val,           \
    type   *D,             \
    int    *I              \
    )

// Double k-select prototype
void KERNEL3(gsknn_heapselect_int_d4,double);

#define KERNEL4(name,type) \
  name(                    \
    int    k,              \
    type   *a,             \
    type   *aa,            \
    type   *b,             \
    type   *bb,            \
    type   *c,             \
    unsigned long long ldc,\
    unsigned long long last, \
    aux_t  *aux            \
    )

// Double square 2-norm prototype
void KERNEL4(sq2nrm_asm_d8x4,double);



// Float rank-k update function pointer table
void KERNEL1((*rankk_s[ 2 ]),float)  = {
  knn_rank_k_ref_s8x8,
  knn_rank_k_abs_ref_s8x8
};

// Double rank-k update function pointer table
void KERNEL1((*rankk_d[ 2 ]),double)  = {
  rnn_rank_k_asm_d8x4,
  rnn_rank_k_abs_int_d8x4
};

// Float knn function pointer table
void KERNEL2((*micro_s[ 1 ]),float) = {
  knn_r_ref_s8x8_row
};

// Double knn function pointer table
void KERNEL2((*micro_d[ 2 ]),double) = {
  rnn_r_int_d8x4_row,
  rnn_r_1norm_int_d8x4_row
};

// Double k-select function pointer table
void KERNEL3((*kselect[ 1 ]),double) = {
  gsknn_heapselect_int_d4
};

// Double square 2-norm function pointer table
void KERNEL4((*sq2nrm[ 1 ]),double) = {
  sq2nrm_asm_d8x4
};

#endif // define __RNN_KERNEL_H__
