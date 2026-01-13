data {
  int<lower=1> n;
  vector[n] t;
  vector[n] y;

  int<lower=1> num_seasonal;                 // number of seasonalities
  array[num_seasonal] int<lower=1> J;        // Fourier terms per seasonality
  vector[num_seasonal] periods;              // periods (e.g. 7, 365.25)
}

transformed data {
  vector[num_seasonal] omega;
  int<lower=1> K = sum(J);                   // total Fourier terms

  for (s in 1:num_seasonal)
    omega[s] = 2 * pi() / periods[s];
}

parameters {
  real alpha;
  real beta;
  real<lower=0> sigma;

  vector[K] beta_cos;
  vector[K] beta_sin;
}

model {
  alpha ~ normal(0, 10);
  beta ~ normal(0, 1);
  sigma ~ normal(0, 1);
  beta_cos ~ normal(0, 1);
  beta_sin ~ normal(0, 1);

  vector[n] mu;

  for (i in 1:n) {
    real ssn = 0;
    int pos = 1;

    for (s in 1:num_seasonal) {
      for (j in 1:J[s]) {
        ssn += beta_cos[pos] * cos(j * omega[s] * t[i]) +
               beta_sin[pos] * sin(j * omega[s] * t[i]);
        pos += 1;
      }
    }

    mu[i] = alpha + beta * t[i] + ssn;
  }

  y ~ normal(mu, sigma);
}
// 
