data {
  int<lower=0> n;
  vector[n] t;
  vector[n] log_y;  // We pass logged data directly
  int<lower=0> k;
  vector[k] omega;
}

parameters {
  vector[k] beta_cos;
  vector[k] beta_sin;
  real alpha;
  real beta;
  real<lower=0> sigma;
}

model {
  // Narrower, more realistic priors for NO2 log-space
  alpha ~ normal(3, 1);   
  beta ~ normal(0, 0.1);
  beta_cos ~ normal(0, 0.5);
  beta_sin ~ normal(0, 0.5);
  sigma ~ exponential(2); 

  vector[n] mu = alpha + beta * t;
  for (j in 1:k) {
    mu += beta_cos[j] * cos(omega[j] * t) + beta_sin[j] * sin(omega[j] * t);
  }

  log_y ~ normal(mu, sigma); // Much more stable than lognormal()
}
