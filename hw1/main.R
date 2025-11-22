# install.packages("rstan")
# install.packages("cmdstanr", repos="https://mc-stan.org/r-packages/")
# install.packages("languageserver")
# install.packages("rlang")

setwd("C:/Users/sebas/one/OneDrive/grive/faks/masters/y2/s1/BayesianStats/BayesianStats/hw1")
library(cmdstanr) # for interfacing Stan
library(ggplot2) # for visualizations
library(tidyverse) # for data manipulations
library(posterior) # for extracting samples
library(bayesplot) # for some quick MCMC visualizations
library(mcmcse) # for comparing samples and calculating MCSE
library(ggdist) # for visualizing distributions
library(rstan)
library(HDInterval)
# install.packages("HDInterval")

######### Import data and model ########c#########
model <- cmdstan_model("linear.stan", quiet=TRUE)
data <- read.csv("data.csv", stringsAsFactors=TRUE)

######### Record min/max so that we can rescale to original #################
min_gdp <- min(data$gdp_per_capita)
max_gdp <- max(data$gdp_per_capita)
min_gdp

min_population <- min(data$population)
max_population <- max(data$population)

min_year <- min(data$year)
max_year <- max(data$year)

rescale <- function(x, x_min, x_max) {
  return(x / (x_max - x_min))
}

######### Min-max normalize #################
data$gdp_per_capita <- (data$gdp_per_capita - min(data$gdp_per_capita)) /
  (max(data$gdp_per_capita) - min(data$gdp_per_capita))

data$population <- (data$population - min(data$population)) /
  (max(data$population) - min(data$population))

data$year <- (data$year - min(data$year)) / 
  (max(data$year) - min(data$year))

######### Set reference category, and onehot encode #################
contrasts(data$continent) <- contr.treatment(n_distinct(data$continent))
# Onehot encode, ref=Africa
contrasts(data$continent)

onehot = list(continent = contr.treatment(n_distinct(data$continent)))
X <- model.matrix(~ gdp_per_capita + population + continent + year, data = data, contrasts.arg = onehot) # nolint
y <- data$life_expectancy

stan_data <- list(n = nrow(data), m = ncol(X), X = X, y = y)

data[1:4, ]

######### Fit model + diagnostics #################
fit <- model$sample(
    data = stan_data,
    parallel_chains = 4,
    seed = 1
)

# MCMC analysis
mcmc_trace(fit$draws("betas"))
fit$summary("betas")
mcse_mean(df_betas$population)

# Transform Betas
sigmas <- as_draws_df(fit$draws("sigma"))
df_betas <- as_draws_df(fit$draws("betas"))
df_betas <- df_betas %>% select(-.chain, -.iteration, -.draw)
colnames(df_betas) <- colnames(X)

# Store beats + sigmas
df_all <- df_betas
df_all$sigma <- sigmas$sigma
colnames(df_all) <- c(colnames(X), "sigma")

beta_mat <- as.matrix(df_betas)  # rows = iterations, cols = betas
colnames(beta_mat) <- colnames(X)

# Mean betas
betas <- matrix(colMeans(df_betas))
rownames(betas) <- colnames(X)
betas

# Convert to two columns for plotting
# df_betas["(Intercept)"] <- NULL
#%>% select(-year_rescaled, -population_rescaled, -gdp_per_capita_rescaled) %>%
df_betas_ <- df_betas %>% select(-population_rescaled, -year_rescaled, -gdp_per_capita_rescaled)
colnames(df_betas_) <- c("(Intercept)", "gdp_per_capita", "population", "Americas", "Asia", "Europe", "Oceania", "year")
colnames(df_betas_)
colnames(df_betas)

df_betas_long <- df_betas_ %>% pivot_longer(everything(), names_to = "feature", values_to = "value")
df_betas_long
windows()
ggplot(df_betas_long, aes(x = value, y = feature)) +
  stat_halfeye(point_interval = median_qi) +
  labs(title = "Posterior distributions of coefficients") +
  theme(axis.text = element_text(size=14))
contrasts(data$continent)
############################ 1. Year vs. life exp. ######################################

# Get rescaled sizes 
mcse(df_betas$year)
hdi(df_betas$year)
# 95% CI is strictly positive, we are ceratain it is rising

df_betas$year_rescaled <- rescale(df_betas$year, min_year, max_year)
inf <- mcse(df_betas$year_rescaled)
hdi(df_betas$year_rescaled)
inf
year_mean = inf$est  #mean(df_betas$year_rescaled)
year_sd = inf$se #sd(df_betas$year_rescaled)
sprintf("Each year, life expectancy goes up by %.2f pm %.2f years", year_mean, year_sd)


####################### 2. GDP and population correlation to life exp #############.

df_betas$gdp_per_capita_rescaled <- rescale(df_betas$gdp_per_capita, min_gdp, max_gdp)
df_betas$population_rescaled <- rescale(df_betas$population, min_population, max_population)

beta_gdp <- df_betas$gdp_per_capita
beta_population <- df_betas$population

mcse(df_betas$gdp_per_capita)
hdi(df_betas$gdp_per_capita)

mcse(df_betas$population)
hdi(df_betas$population)


mcse(df_betas$gdp_per_capita_rescaled)
mcse(df_betas$population_rescaled)
# y <- X %*% t(beta_mat)
# y <- colMeans(y)

cor(data$gdp_per_capita, data$population)
cor(data$gdp_per_capita, y)
cor(data$population, y)

mean_gdp <- mean(beta_gdp)
mean_population <- mean(beta_population)

ci_gdp <- quantile(beta_gdp, c(0.025, 0.975))
ci_population <- quantile(beta_population, c(0.025, 0.975))

sprintf("[normalized] GDP per capita effect: %.2f (95%% CI: %.2f, %.2f)", mean_gdp, ci_gdp[1], ci_gdp[2])
sprintf("[normalized] Population effect: %.2f (95%% CI: %.2f, %.2f)", mean_population, ci_population[1], ci_population[2])

scaled_mean_gdp <- rescale(mean_gdp, min_gdp, max_gdp)
scaled_mean_gdp

scaled_mean_pop <- rescale(mean_population, min_population, max_population)
scaled_mean_pop

############################ 3. When am I going to die #############

colnames(X)
slo_gdp <- 18765  # PPP
slo_gdp_scaled <- (slo_gdp - min_gdp) / (max_gdp - min_gdp)
# slo_gdp <- means$mean_gdp[means$continent=="Europe"]
slo_pop <- 1992060
# slo_pop <- means$mean_pop[means$continent == "Europe"]
slo_pop_scaled <- (slo_pop - min_population) / (max_population - min_population)

year <- 2001
year_scaled <- (year - min_year) / (max_year - min_year)
year_scaled

my_info <- c(
  1,
  slo_gdp_scaled, 
  slo_pop_scaled, 
  0, 0, 1, 0,
  year_scaled
  )

my_pred <- my_info %*% t(beta_mat)

my_dist <- rnorm(nrow(beta_mat), mean=my_pred, sd=sigmas$sigma)
# windows()
hist(my_dist)

my_data <- mcse(my_dist)

y_mean <- my_data$est #mean(my_pred)
y_ci <- my_data$se #quantile(my_pred, c(0.025, 0.975))
hdi(my_dist)
sprintf("I will live till %.2f pm %.2f", y_mean, y_ci)

my_dist <- as.data.frame(my_dist)
colnames(my_dist)

windows()
ggplot(my_dist, aes(x = my_dist)) +
  geom_density(alpha = 0.4) +
  xlim(50, 100) +
  xlab("Predicted life expectancy") +
  ylab("Density") +
  ggtitle("My life distribution") +
  theme_minimal()

############################ 4. Average european born in 2001 vs other 2001 ppl #############

# eu_gdp <- mean(data$gdp_per_capita[data$continent == "Europe"])
# eu_pop <- mean(data$population[data$continent == "Europe"])

# africa_gdp <- mean(data$gdp_per_capita[data$continent == "Africa"])
# africa_pop <- mean(data$population[data$continent == "Africa"])

colnames(X)
colnames(beta_mat)
contrasts(data$continent)

continent_means <- data |> group_by(continent) |> summarize(
  mean_gdp = mean(gdp_per_capita),
  mean_pop = mean(population),
  .groups = "drop"
)
continent_means

continents <- c("Africa", "Americas", "Asia", "Europe", "Oceania")

make_input_vec_mean <- function(continent, year, means, contrast_mtx) {
  gdp <- means$mean_gdp[means$continent == continent]
  pop <- means$mean_pop[means$continent == continent]

  onehot <- contrast_mtx[continent, ]

  c(1, gdp, pop, onehot, year)
}

contr <- contrasts(data$continent)
year_scaled <- (2001 - min_year) / (max_year - min_year)
year_scaled

inputs <- lapply(continents, function(continent) {
  make_input_vec_mean(continent, year_scaled, continent_means, contr)
})

mat <- matrix(NA, nrow = nrow(beta_mat), ncol = length(inputs))
for (j in seq(inputs)) {
  mat[, j] <- inputs[[j]] %*% t(beta_mat)
}

colnames(mat) <- continents

# mean_eu <- mean(eu_le)
# mean_eu_africa <- mean_eu > mean(mat[,1])
# mean_eu_americas <- mean_eu > mean(mat[,2])
# mean_eu_asia <- mean_eu > mean(mat[,3])
# mean_eu_oceania <- mean_eu > mean(mat[,5])

eu_le <- mat[ , 4]
eu_vs_africa <- eu_le - mat[, 1]
eu_vs_americas <- eu_le - mat[, 2]
eu_vs_asia <- eu_le - mat[, 3]
eu_vs_oceania <- eu_le - mat[, 5]

colMeans(mat)

# Check how much longer eu lives in years
mcse(eu_vs_africa)
mcse(eu_vs_americas)
mcse(eu_vs_asia)
mcse(eu_vs_oceania)

# Probabilities
p_eu_vs_africa <- eu_le > mat[, 1]
p_eu_vs_americas <- eu_le > mat[, 2]
p_eu_vs_asia <- eu_le > mat[, 3]
p_eu_vs_oceania <- eu_le > mat[, 5]

mcse(p_eu_vs_africa)
mcse(p_eu_vs_americas)
mcse(p_eu_vs_asia)
mcse(p_eu_vs_oceania)

# mean(eu_vs_africa)
# quantile(eu_vs_africa, c(0.025, 0.975))

# mean(eu_vs_americas)
# quantile(eu_vs_americas, c(0.025, 0.975))

# mean(eu_vs_asia)
# quantile(eu_vs_asia, c(0.025, 0.975))

# mean(eu_vs_oceania)
# quantile(eu_vs_oceania, c(0.025, 0.975))

df_long <- as.data.frame(mat) %>%
  pivot_longer(cols = everything(), names_to = "continent", values_to = "life_expectancy")


res <- c(as.integer(mean_eu_africa), as.integer(mean_eu_americas), as.integer(mean_eu_asia), as.integer(mean_eu_oceania))
res

# Plot distributions
windows()
ggplot(df_long, aes(x = life_expectancy, fill = continent)) +
  geom_density(alpha = 0.4) +
  xlim(50, 90) +
  xlab("Predicted life expectancy") +
  ylab("Density") +
  ggtitle("Posterior distribution of life expectancy by continent") +
  theme_minimal()


# individual vs avg
# avg: take means
# ind: B, sigma -> Sample normals around linear predictors. output = (B[1:8] + sigma) ~ N(XB, sigma) -> sample and compute

# Average european will outlive average african, american, asian but not oceanian

############################ 5. Any european born in 2001 vs other 2001 ppl #############
betasigma_mat = as.matrix(df_all)
colnames(betasigma_mat) <- colnames(df_all)

mat <- matrix(NA, nrow = nrow(betasigma_mat), ncol = length(inputs) + 1)
mat[, 1] <- betasigma_mat[, ncol(betasigma_mat)]
for (j in seq(inputs)) {
  mat[, j+1] <- inputs[[j]] %*% t(beta_mat) # MEAN
}
dim(mat)
colnames(mat) <- c("sigma", continents)
colnames(mat)

colnames(betasigma_mat)
# sigma + Mean for each continent
n_samples = nrow(mat)
pred_samples_africa <- rnorm(n=n_samples, mean = mat[, 2], sd = mat[, 1])
pred_samples_americas <- rnorm(n=n_samples, mean = mat[, 3], sd = mat[, 1])
pred_samples_asia <- rnorm(n=n_samples, mean = mat[, 4], sd = mat[, 1])
pred_samples_eu <- rnorm(n=n_samples, mean = mat[, 5], sd = mat[, 1])
pred_samples_oceania <- rnorm(n=n_samples, mean = mat[, 6], sd = mat[, 1])

predictive_df <- data.frame(
  africa = pred_samples_africa,
  americas = pred_samples_americas,
  asia = pred_samples_asia,
  eu = pred_samples_eu,
  oceania = pred_samples_oceania
)

# long
pred_long <- predictive_df %>%
  pivot_longer(cols = everything(), names_to = "continent", values_to = "life_expectancy")

windows()
ggplot(pred_long, aes(x = life_expectancy, fill = continent)) +
  geom_density(alpha = 0.4) +
  # xlim(50, 90) +
  xlab("Predicted life expectancy") +
  ylab("Density") +
  ggtitle("Posterior distribution of life expectancy by continent") +
  theme_minimal()

# Individual life exp
mcse(pred_samples_eu - pred_samples_africa)
mcse(pred_samples_eu - pred_samples_americas)
mcse(pred_samples_eu - pred_samples_asia)
mcse(pred_samples_eu - pred_samples_oceania)

# Probabilities
mcse(pred_samples_eu > pred_samples_africa)
mcse(pred_samples_eu > pred_samples_americas)
mcse(pred_samples_eu > pred_samples_asia)
mcse(pred_samples_eu > pred_samples_oceania)

############### Me vs. Professor #####################
my_info <- c(
  1,
  slo_gdp_scaled, 
  slo_pop_scaled, 
  0, 0, 1, 0,
  year_scaled
  )

prof_data <- my_info
colnames(X)
prof_data[8] <- (1985 - min_year) / (max_year - min_year)
prof_data[2] <- (13481 - min_gdp) / (max_gdp - min_gdp)
prof_data[3] <- (1941641 - min_population) / (max_population - min_population)

prof_data

# Predict prof
prof_pred = prof_data %*% t(beta_mat)
prof_dist = rnorm(nrow(betasigma_mat), mean=prof_pred, sd=sigmas$sigma)
mcse(prof_dist)
hdi(prof_dist)

my_pred = my_info %*% t(beta_mat)
my_dist = rnorm(nrow(betasigma_mat), mean=my_pred, sd=sigmas$sigma)
mcse(my_dist)
hdi(my_dist)

mcse(my_dist - prof_dist)
mcse(my_dist > prof_dist)
hdi(my_dist - prof_dist)

comparison_df = data.frame(
  me = my_dist,
  prof = prof_dist
)

comp_long <- comparison_df %>%
  pivot_longer(cols = everything(), names_to = "person", values_to = "life_expectancy")

# Just posterior dist
ggplot(comp_long, aes(x = life_expectancy, fill = person)) +
  geom_density(alpha = 0.4) +
  # xlim(50, 90) +
  xlab("Predicted life expectancy") +
  ylab("Density") +
  ggtitle("Posterior for me vs prof") +
  theme_minimal()


# Expected diff in years
comp <- my_dist - prof_dist
info <- mcse(comp)
info

# Probability I live longer
mcse(my_dist > prof_dist)

df_comparison <- data.frame(my_pred = as.vector(my_dist), prof_pred = as.vector(prof_dist), diff = as.vector(comp))
df_long <- pivot_longer(
  df_comparison,
  cols = c(my_pred, prof_pred), # Columns to gather
  names_to = "Predictor",        # New column for the group name (my_pred/prof_pred)
  values_to = "life_exp" # New column for the predicted values
)

# Dist with marked means
windows()
ggplot(df_long, aes(x=life_exp, fill=Predictor)) + geom_density(alpha=.5) + 
  geom_vline(aes(xintercept = mean(my_pred)), color="red") + 
  geom_vline(aes(xintercept = mean(prof_pred)), color="blue")

######################################################

ggplot(data, aes(x = gdp_per_capita, y = y)) +
  geom_point() +
  geom_smooth(method = "lm", se = TRUE, color = "blue") +
  labs(title = "Life expectancy vs GDP per capita")

# Vary one predictor while keeping others fixed at mean
pop_seq <- seq(0, 1, length.out = 100)
gdp_seq <- seq(0, 1, length.out = 100)

beta_mat[1,]
y_pop <- beta_mat[, "(Intercept)"] + beta_mat[, "population"] %*% t(pop_seq)
y_gdp <- beta_mat[, "(Intercept)"] + beta_mat[, "gdp_per_capita"] %*% t(gdp_seq)

# Mean effect
y_pop_mean <- apply(y_pop, 2, mean)
y_gdp_mean <- apply(y_gdp, 2, mean)

ggplot() +
  geom_line(aes(x = pop_seq, y = y_pop_mean), color = "blue") +
  geom_line(aes(x = gdp_seq, y = y_gdp_mean), color = "red") +
  labs(x = "Scaled predictor", y = "Predicted life expectancy",
       title = "Effect of population (blue) and GDP per capita (red)")


# Posterior vs observed
y_rep <- X %*% t(beta_mat)  # or manually: X %*% posterior samples of betas
y_mean <- apply(y_rep, 1, mean)
windows()   # Windows
plot(y, y_mean,
      ylim = c(30, 80),
      xlim = c(30, 80),
      xlab = "Observed life expectancy",
      ylab = "Predicted life expectancy",
      main = "Posterior predictive mean")
abline(0, 1, col = "red")

# Is life expectancy improving over time? Fit LR model and check slope

# How are GDP per capita and population correlated with life expectancy? Check coefficients

# Estimate how long you are expected to live given your birth year and the fact that you live in Europe. Infer

# What is the probability that an average European with the same birth year as you will live longer than their average counterparts on other continents?

# What is the probability that an individual European with the same birth year as you will live longer than an individual from another continent?

# Quantify how much longer (or shorter) your life expectancy is, compared to your professor (year of birth 1985).

