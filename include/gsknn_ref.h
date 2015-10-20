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

void sgsrnn_ref(
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
    float  *D,
    int    *I
    );
