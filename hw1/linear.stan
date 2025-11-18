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
    y ~ normal(X * betas, sigma);
}
