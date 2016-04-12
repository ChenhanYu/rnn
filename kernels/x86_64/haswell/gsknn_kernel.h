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
void KERNEL1(rank_k_ref_s,float);
void KERNEL1(rank_k_abs_ref_s,float);

// Double rank-k update prototype
void KERNEL1(rank_k_ref_d,double);
void KERNEL1(rank_k_abs_ref_d,double);
void KERNEL1(rank_k_int_d8x6,double);

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
void KERNEL2(knn_ref_row_s,float);
void KERNEL2(knn_int_row_s16x6,float);

// Double knn prototype
void KERNEL2(knn_ref_row_d,double);
void KERNEL2(knn_int_row_d8x6,double);

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


#define KERNEL5(name,type) \
  name(                    \
    int    k,              \
    type   alpha0,         \
    type   alpha1,         \
    type   *a,             \
    type   *b,             \
    type   beta0,          \
    type   *c0,            \
    type   beta1,          \
	type   *c1,            \
    int    ldc,            \
    aux_t  *aux            \
	)
    

// Double strassen prototype
void KERNEL5(strassen_ref_d,double);
void KERNEL5(strassen_int_d8x6,double);

// Float rank-k update function pointer table
void KERNEL1((*rankk_s[ 2 ]),float)  = {
  rank_k_ref_s,
  rank_k_abs_ref_s
};

// Double rank-k update function pointer table
void KERNEL1((*rankk_d[ 2 ]),double)  = {
  //rank_k_ref_d,
  rank_k_int_d8x6,
  rank_k_abs_ref_d
};

// Float knn function pointer table
void KERNEL2((*micro_s[ 1 ]),float) = {
  //knn_ref_row_s
  knn_int_row_s16x6
};

// Double knn function pointer table
void KERNEL2((*micro_d[ 1 ]),double) = {
  //knn_ref_row_d
  knn_int_row_d8x6
};

// Double k-select function pointer table
void KERNEL3((*kselect[ 1 ]),double) = {
  gsknn_heapselect_int_d4
};

// Double square 2-norm function pointer table
void KERNEL4((*sq2nrm[ 1 ]),double) = {
  sq2nrm_asm_d8x4
};

// Strassen function pointer table
void KERNEL5((*strassen_d[ 1 ]),double) = {
  //strassen_ref_d
  strassen_int_d8x6
};

#endif // define __RNN_KERNEL_H__
