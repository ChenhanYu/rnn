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
