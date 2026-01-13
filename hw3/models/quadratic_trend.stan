data {
    int<lower=1> n;          // number of observations
    vector[n] t;             // time (days)
    vector[n] y;             // NO2
    int<lower=1> k;          // number of seasonal frequencies
    vector[k] omega;         // frequencies
}

parameters {
    real alpha;              // intercept
    real beta1;              // linear term
    real beta2;              // quadratic term

    vector[k] beta_cos;      // cosine coefficients
    vector[k] beta_sin;      // sine coefficients

    real<lower=0> sigma;     // observation noise
}

model {
    vector[n] mu;

    // Quadratic + harmonics
    for (i in 1:n) {
        mu[i] = alpha + beta1 * t[i] + beta2 * square(t[i]);
        for (j in 1:k) {
            mu[i] += beta_cos[j] * cos(omega[j] * t[i])
                   + beta_sin[j] * sin(omega[j] * t[i]);
        }
    }

    // Priors
    alpha ~ normal(100, 10);
    beta1 ~ normal(0, 4);
    beta2 ~ normal(0, 1);

    beta_cos ~ normal(0, 5);
    beta_sin ~ normal(0, 5);

    sigma ~ exponential(.5);

    // Likelihood
    y ~ normal(mu, sigma);
}

generated quantities {
    vector[n] y_rep;
    vector[n] mu;

    for (i in 1:n) {
        mu[i] = alpha + beta1 * t[i] + beta2 * square(t[i]);
        for (j in 1:k) {
            mu[i] += beta_cos[j] * cos(omega[j] * t[i])
                   + beta_sin[j] * sin(omega[j] * t[i]);
        }
        y_rep[i] = normal_rng(mu[i], sigma);
    }
}
