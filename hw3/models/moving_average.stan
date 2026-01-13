data {
    int<lower=1> n;
    vector[n] t;
    vector[n] y;
    int<lower=1> k;
    vector[k] omega;
}

parameters {
    real alpha;
    real beta;

    vector[k] beta_cos;
    vector[k] beta_sin;

    real<lower=-1, upper=1> theta;  // MA(1) coefficient
    real<lower=0> sigma;

    vector[n] eps;                  // latent innovations
}

transformed parameters {
    vector[n] mu;
    vector[n] y_hat;

    for (i in 1:n) {
        mu[i] = alpha + beta * t[i];
        for (j in 1:k) {
            mu[i] += beta_cos[j] * cos(omega[j] * t[i])
                  +  beta_sin[j] * sin(omega[j] * t[i]);
        }
    }

    // MA(1) structure
    y_hat[1] = mu[1] + eps[1];
    for (i in 2:n) {
        y_hat[i] = mu[i] + eps[i] + theta * eps[i-1];
    }
}

model {
    // Priors
    alpha ~ normal(0, 10);
    beta  ~ normal(0, 5);

    beta_cos ~ normal(0, 5);
    beta_sin ~ normal(0, 5);

    theta ~ normal(0, 0.5);
    sigma ~ exponential(1);

    eps ~ normal(0, sigma);

    // Likelihood
    y ~ normal(y_hat, 1e-6);  // deterministic link (see note below)
}

generated quantities {
    vector[n] y_rep;
    vector[n] eps_rep;

    eps_rep[1] = normal_rng(0, sigma);
    y_rep[1] = mu[1] + eps_rep[1];

    for (i in 2:n) {
        eps_rep[i] = normal_rng(0, sigma);
        y_rep[i] = mu[i] + eps_rep[i] + theta * eps_rep[i-1];
    }
}
