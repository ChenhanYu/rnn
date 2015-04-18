void dgsrnn_directKQuery(
    double *ref,
    double *query,
    long   n,
    long   m,
    long   k,
    long   dim,
    std::pair<double, long> *result,
    double *dist,                    // The following 3 parameters are
    double *sqnormr,                 // ignored in this routine.
    double *sqnormq                  // 
    );

void dgsrnn_directKQuery_var2(
    int    m,
    int    n,
    int    dim,
    int    k,
    double *query,
    double *XA2,
    int    *amap,
    double *ref,
    double *XB2,
    int    *bmap,
    std::pair<double, long> *result
    );
