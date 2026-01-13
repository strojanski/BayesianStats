data {
  int<lower=1> n;
  vector[n] t;          // standardized time
  vector[n] y;

  int<lower=1> J_year;  // e.g. 2
  int<lower=0> J_week;  // 0 or 1
}

transformed data {
  real omega_year = 2 * pi() / 365.25;
  real omega_week = 2 * pi() / 7;
}

parameters {
  ordered[2] tau;

  real alpha;
  vector[3] beta;
  vector<lower=0>[3] sigma;

  vector[J_year] beta_cos_year;
  vector[J_year] beta_sin_year;

  vector[J_week] beta_cos_week;
  vector[J_week] beta_sin_week;
}

model {
  // priors
  beta ~ normal(0, 0.5);
  sigma ~ normal(0, 0.5);

  beta_cos_year ~ normal(0, 0.3);
  beta_sin_year ~ normal(0, 0.3);

  beta_cos_week ~ normal(0, 0.1);
  beta_sin_week ~ normal(0, 0.1);

  for (i in 1:n) {
    real ssn = 0;

    for (j in 1:J_year)
      ssn += beta_cos_year[j] * cos(j * omega_year * t[i])
           + beta_sin_year[j] * sin(j * omega_year * t[i]);

    for (j in 1:J_week)
      ssn += beta_cos_week[j] * cos(j * omega_week * t[i])
           + beta_sin_week[j] * sin(j * omega_week * t[i]);

    real mu;
    if (t[i] <= tau[1])
      mu = alpha + beta[1] * t[i] + ssn;
    else if (t[i] <= tau[2])
      mu = alpha + beta[1] * tau[1]
                 + beta[2] * (t[i] - tau[1]) + ssn;
    else
      mu = alpha + beta[1] * tau[1]
                 + beta[2] * (tau[2] - tau[1])
                 + beta[3] * (t[i] - tau[2]) + ssn;

    y[i] ~ normal(mu, sigma[
      t[i] <= tau[1] ? 1 :
      t[i] <= tau[2] ? 2 : 3
    ]);
  }
}
