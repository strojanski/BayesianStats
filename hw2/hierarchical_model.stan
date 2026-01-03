data {
  int<lower=1> N;
  vector[N] y;
  vector[N] year;
  vector[N] log_gdp;
  vector[N] life_exp;

  int<lower=1> C;                 // countries
  int<lower=1> K;                 // continents
  array[N] int<lower=1,upper=C> country;
  array[C] int<lower=1,upper=K> continent;
}

parameters {
  real alpha_global;
  vector[3] beta_global;

  vector[K] alpha_cont;
  matrix[K,3] beta_cont;

  vector[C] alpha_country;
  matrix[C,3] beta_country;

  real<lower=0> sigma_y;

  real<lower=0> sigma_alpha_cont;
  vector<lower=0>[3] sigma_beta_cont;

  real<lower=0> sigma_alpha_country;
  vector<lower=0>[3] sigma_beta_country;
}

model {
  alpha_global ~ normal(5, 2);
  beta_global ~ normal(0, 1);

  sigma_y ~ exponential(1);
  sigma_alpha_cont ~ exponential(1);
  sigma_alpha_country ~ exponential(1);
  sigma_beta_cont ~ exponential(1);
  sigma_beta_country ~ exponential(1);

  for (k in 1:K) {
    alpha_cont[k] ~ normal(alpha_global, sigma_alpha_cont);
    beta_cont[k] ~ normal(beta_global, sigma_beta_cont);
  }

  for (c in 1:C) {
    alpha_country[c] ~ normal(alpha_cont[continent[c]], sigma_alpha_country);
    beta_country[c] ~ normal(beta_cont[continent[c]], sigma_beta_country);
  }

  for (i in 1:N) {
    y[i] ~ normal(
      alpha_country[country[i]]
      + beta_country[country[i],1] * year[i]
      + beta_country[country[i],2] * log_gdp[i]
      + beta_country[country[i],3] * life_exp[i],
      sigma_y
    );
  }
}

