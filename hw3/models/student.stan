// student_t_harmonic.stan
data {
  int<lower=1> n;              // number of observations
  vector[n] t;                 // time (days)
  vector[n] y;                 // NO2 measurements

  int<lower=1> num_seasonal;   // number of seasonalities
  array[num_seasonal] int<lower=1> J;  // Fourier terms per seasonality
  vector[num_seasonal] periods;        // seasonal periods
}

transformed data {
  vector[num_seasonal] omega;
  int<lower=1> K = sum(J);    // total Fourier terms

  for (s in 1:num_seasonal)
    omega[s] = 2 * pi() / periods[s];
}

parameters {
  real alpha;                  // intercept
  real beta;                   // linear trend slope
  real<lower=0> sigma;         // scale of Student-t
  real<lower=2> nu;            // degrees of freedom (heavy tail)
  vector[K] beta_cos;           // Fourier coefficients
  vector[K] beta_sin;
}

model {
  alpha ~ normal(0, 10);
  beta ~ normal(0, 1);
  sigma ~ normal(0, 1);
  nu ~ gamma(2, 0.1);            // weakly informative
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

  y ~ student_t(nu, mu, sigma);
}
