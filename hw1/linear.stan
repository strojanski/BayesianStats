data {
    int<lower=0> n;              // n_rows
    int<lower=0> m;              // n_feats
    matrix[n, m] X;     // data
    vector[n] y;        // labels
}

parameters {
    vector[m] betas;  // params
    real<lower=0> sigma; // stdev
}

model {
    betas ~ normal(0,10);
    sigma ~ cauchy(0,2);
    y ~ normal(X * betas, sigma);
}
