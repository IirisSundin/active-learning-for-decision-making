data {
    int<lower=1> N1;
    int<lower=1> N2;
    int<lower=1> d;
    int<lower=0> Nc;
    int<lower=1> Np1;
    int<lower=0> Npp1;
    int<lower=1> Np2;
    int<lower=0> Npp2;
    vector[N1] y1;
    vector[N2] y2;
    int<lower=0,upper=1> yc[Nc];
    vector[d] x1[N1];
    vector[d] x2[N2];
    vector[d] xc1[Nc];
    vector[d] xc2[Nc];
    vector[d] xp1[Np1];
    vector[d] xp2[Np2];
    vector[d] xpp1[Npp1];
    vector[d] xpp2[Npp2];
}
transformed data {
    int<lower=1> NT1=N1+Nc+Np1+N2+Npp1;
    int<lower=1> NT2=N2+Nc+Np2+N1+Npp2;
    vector[d] xa1[NT1];
    vector[d] xa2[NT2];
    xa1[1:N1,1:d] = x1;
    xa2[1:N2,1:d] = x2;
    xa1[N1+1:N1+Nc, 1:d] = xc1;
    xa2[N2+1:N2+Nc, 1:d] = xc2;
    xa1[N1+Nc+1:N1+Nc+Np1, 1:d] = xp1;
    xa2[N2+Nc+1:N2+Nc+Np2, 1:d] = xp2;
    xa1[N1+Nc+Np1+1:N1+Nc+Np1+N2, 1:d] = x2;
    xa2[N2+Nc+Np2+1:N2+Nc+Np2+N1, 1:d] = x1;
    xa1[N1+Nc+Np1+N2+1:N1+Nc+Np1+N2+Npp1, 1:d] = xpp1;
    xa2[N2+Nc+Np2+N1+1:N2+Nc+Np2+N1+Npp2, 1:d] = xpp2;
}
parameters {
    vector<lower=1e-3>[d] length_scale1;
    real<lower=1e-3> alpha1;
    real<lower=1e-3> sigma1;
    vector<lower=1e-3>[d] length_scale2;
    real<lower=1e-3> alpha2;
    real<lower=1e-3> sigma2;
    vector[N1+Nc+Np1+N2+Npp1] eta1;
    vector[N2+Nc+Np2+N1+Npp2] eta2;
}
transformed parameters {
    vector[NT1] f1;
    vector[NT2] f2;
    matrix[NT1, NT1] L1;
    matrix[NT2, NT2] L2;
    {
        matrix[NT1, NT1] K1;
        for (n1 in 1:NT1) {
            for (n2 in 1:NT1) {
                K1[n1,n2] = alpha1 * exp( -sum((xa1[n1,1:d]- xa1[n2,1:d]) .* (xa1[n1,1:d]- xa1[n2,1:d]) ./ (length_scale1 .* length_scale1)));
            }
            K1[n1, n1] = K1[n1, n1] + 1e-6;
        }
        L1 = cholesky_decompose(K1);
    }
    {
        matrix[NT2, NT2] K2;
        for (n1 in 1:NT2) {
            for (n2 in 1:NT2) {
                K2[n1,n2] = alpha2 * exp( -sum((xa2[n1,1:d]- xa2[n2,1:d]) .* (xa2[n1,1:d]- xa2[n2,1:d]) ./ (length_scale2 .* length_scale2)));
            }
            K2[n1, n1] = K2[n1, n1] + 1e-6;
        }    
        L2 = cholesky_decompose(K2);
    }
    f1 = L1 * eta1;
    f2 = L2 * eta2;
}
model {
    length_scale1 ~ inv_gamma(1, 1);
    length_scale2 ~ inv_gamma(1, 1);
    alpha1 ~ normal(0, 1);
    alpha2 ~ normal(0, 1);
    sigma1 ~ normal(0, 1);
    sigma2 ~ normal(0, 1);
    eta1 ~ normal(0, 1);
    eta2 ~ normal(0, 1);
    y1 ~ normal(f1[1:N1], sigma1);
    y2 ~ normal(f2[1:N2], sigma2);
    yc ~ bernoulli(Phi((f1[N1+1:N1+Nc] - f2[N2+1:N2+Nc])/(sigma1+sigma2))); #yc=1: f1>f2 and yc=0: f2>f1 
}
generated quantities {
    vector[Np1] fp1;
    vector[Np2] fp2;
    vector[Npp1] fpp1;
    vector[Npp2] fpp2;
    vector[N1] fa1;
    vector[N2] fa2;
    vector[Np1] yp1;
    vector[Np2] yp2;
    fp1 = f1[N1+Nc+1:N1+Nc+Np1];
    fp2 = f2[N2+Nc+1:N2+Nc+Np2];
    fa1 = f2[N2+Nc+Np2+1:N2+Nc+Np2+N1];
    fa2 = f1[N1+Nc+Np1+1:N1+Nc+Np1+N2];
    fpp1 = f1[N1+Nc+Np1+N2+1:N1+Nc+Np1+N2+Npp1];
    fpp2 = f2[N2+Nc+Np2+N1+1:N2+Nc+Np2+N1+Npp2];
    for (i in 1:Np1)
        yp1[i] = normal_rng(fp1[i], sigma1);
    for (i in 1:Np2)
        yp2[i] = normal_rng(fp2[i], sigma2);
}
