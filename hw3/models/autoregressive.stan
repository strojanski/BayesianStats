data {
    int<lower=1> n;              // number of observations
    vector[n] t;                 // time index (e.g. day)
    vector[n] y;                 // NO2 observations
    int<lower=1> k;              // number of seasonal frequencies
    vector[k] omega;             // frequencies
}

parameters {
    real alpha;                  // intercept
    real beta;                   // linear trend

    vector[k] beta_cos;          // cosine coefficients
    vector[k] beta_sin;          // sine coefficients

    real<lower=-1, upper=1> phi; // AR(1) coefficient
    real<lower=0> sigma;         // noise std
}

transformed parameters {
    vector[n] mu;

    for (i in 1:n) {
        mu[i] = alpha + beta * t[i];

        for (j in 1:k) {
            mu[i] += beta_cos[j] * cos(omega[j] * t[i])
                  +  beta_sin[j] * sin(omega[j] * t[i]);
        }
    }
}

model {
    // Priors (weakly informative, adjust if needed)
    alpha ~ normal(90, 10);
    beta  ~ normal(0, 5);

    beta_cos ~ normal(0, 5);
    beta_sin ~ normal(0, 5);

    phi ~ normal(0, 0.5);      // concentrates mass near 0, inside [-1,1]
    sigma ~ exponential(1);

    // Likelihood with AR(1)
    y[1] ~ normal(mu[1], sigma / sqrt(1 - phi^2));

    for (i in 2:n) {
        y[i] ~ normal(mu[i] + phi * (y[i-1] - mu[i-1]), sigma);
    }
}

generated quantities {
    vector[n] y_rep;

    y_rep[1] = normal_rng(mu[1], sigma / sqrt(1 - phi^2));
    for (i in 2:n) {
        y_rep[i] = normal_rng(mu[i] + phi * (y_rep[i-1] - mu[i-1]), sigma);
    }
}
