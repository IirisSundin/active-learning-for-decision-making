data {
    int<lower=0> n;                     // number of data points
    vector[n] x;                       // explanatory variable
    vector[n] a;                      // action variable
    int<lower=0,upper=1> y[n];          // response variable
    int<lower=0> npred;
    vector[npred] xpred;
    vector[npred] apred;
}
parameters {
    vector[6] beta; //slope
}
transformed parameters {
    vector[n] eta;              // linear predictor
    vector[n] etainv;              // linear predictor
    vector[npred] etapred;
    vector[3] xr = [-3.0, 0.0, 3.0]';
    for (i in 1:n) {
        eta[i] =  beta[1]*exp(-(x[i]-xr[1])^2) + beta[2]*exp(-(x[i]-xr[2])^2)+ beta[3]*exp(-(x[i]-xr[3])^2) + beta[4]*exp(-(x[i]-xr[1])^2)*a[i] + beta[5]*exp(-(x[i]-xr[2])^2)*a[i]+ beta[6]*exp(-(x[i]-xr[3])^2)*a[i];
        etainv[i] = beta[1]*exp(-(x[i]-xr[1])^2) + beta[2]*exp(-(x[i]-xr[2])^2)+ beta[3]*exp(-(x[i]-xr[3])^2) + beta[4]*exp(-(x[i]-xr[1])^2)*(1.0-a[i]) + beta[5]*exp(-(x[i]-xr[2])^2)*(1.0-a[i])+ beta[6]*exp(-(x[i]-xr[3])^2)*(1.0-a[i]);
    }
    for (i in 1:npred) { 
        etapred[i] = beta[1]*exp(-(xpred[i]-xr[1])^2) + beta[2]*exp(-(xpred[i]-xr[2])^2)+ beta[3]*exp(-(xpred[i]-xr[3])^2) + beta[4]*exp(-(xpred[i]-xr[1])^2)*apred[i] + beta[5]*exp(-(xpred[i]-xr[2])^2)*apred[i]+ beta[6]*exp(-(xpred[i]-xr[3])^2)*apred[i];
    }
}
model {
    // observation model
    beta ~ normal(0, 3);
    y ~ bernoulli_logit(eta);
}
generated quantities {
    vector[n] theta;
    vector[n] thetainv;
    vector[n] yinv;
    vector[npred] thetapred;
    vector[npred] ypred;
    theta = inv_logit(eta);
    thetainv = inv_logit(etainv);
    for (i in 1:n)
        yinv[i] = bernoulli_rng(thetainv[i]);
    thetapred = inv_logit(etapred);
    for (i in 1:npred)
        ypred[i] = bernoulli_rng(thetapred[i]);
}