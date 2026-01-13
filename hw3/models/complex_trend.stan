data {
  int<lower=1> n;
  vector[n] t;        // standardized time (days)
  vector[n] y;

  int<lower=1> S;     // number of long seasonalities
  vector[S] period;  // periods in days (e.g. [365.25, 1826, 3650])
}

transformed data {
  vector[S] omega;
  for (s in 1:S)
    omega[s] = 2 * pi() / period[s];
}

parameters {
  real alpha;
  real beta;                 // linear trend
  real<lower=0> sigma;

  vector[S] beta_cos;
  vector[S] beta_sin;
}

model {
  // priors
  alpha ~ normal(0, 1);
  beta ~ normal(0, 0.3);
  sigma ~ normal(0, 0.5);

  // strong shrinkage for long cycles
  beta_cos ~ normal(0, 0.2);
  beta_sin ~ normal(0, 0.2);

  for (i in 1:n) {
    real ssn = 0;
    for (s in 1:S) {
      ssn += beta_cos[s] * cos(omega[s] * t[i])
           + beta_sin[s] * sin(omega[s] * t[i]);
    }

    y[i] ~ normal(alpha + beta * t[i] + ssn, sigma);
  }
}
