// // lognormal_harmonic.stan
// // Multiplicative effect!
// data {
//   int<lower=1> n;              
//   vector[n] t;                 
//   vector[n] y;                 // NO2 > 0

//   int<lower=1> num_seasonal;   
//   array[num_seasonal] int<lower=1> J;
//   vector[num_seasonal] periods;
// }

// transformed data {
//   vector[num_seasonal] omega;
//   int<lower=1> K = sum(J);

//   for (s in 1:num_seasonal)
//     omega[s] = 2 * pi() / periods[s];
// }

// parameters {
//   real alpha;
//   real beta;
//   real<lower=0> sigma;
//   vector[K] beta_cos;
//   vector[K] beta_sin;
// }

// model {
//   alpha ~ normal(0, 10);
//   beta ~ normal(0, 1);
//   sigma ~ normal(0, 1);
//   beta_cos ~ normal(0, 1);
//   beta_sin ~ normal(0, 1);

//   vector[n] mu;
//   for (i in 1:n) {
//     real ssn = 0;
//     int pos = 1;
//     for (s in 1:num_seasonal) {
//       for (j in 1:J[s]) {
//         ssn += beta_cos[pos] * cos(j * omega[s] * t[i]) +
//                beta_sin[pos] * sin(j * omega[s] * t[i]);
//         pos += 1;
//       }
//     }
//     mu[i] = alpha + beta * t[i] + ssn;
//   }

//   y ~ lognormal(mu, sigma);
// }


data {
  int<lower=0> n;
  vector[n] t;
  vector<lower=0>[n] y; // NO2 must be positive
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
  // Priors: Adjusted for log-scale
  alpha ~ normal(3, 2);  // exp(3) is ~20ppb, a reasonable NO2 baseline
  beta ~ normal(0, 0.1); // Trends on log scale are usually small
  beta_cos ~ normal(0, 0.5);
  beta_sin ~ normal(0, 0.5);
  sigma ~ exponential(1);

  // Vectorized mu calculation
  vector[n] mu = alpha + beta * t;
  
  for (j in 1:k) {
    mu += beta_cos[j] * cos(omega[j] * t) + beta_sin[j] * sin(omega[j] * t);
  }

  // Likelihood
  y ~ lognormal(mu, sigma);
}
