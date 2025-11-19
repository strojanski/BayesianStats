# install.packages("rstan")
# install.packages("cmdstanr", repos="https://mc-stan.org/r-packages/")
# install.packages("languageserver")

setwd("C:/Users/sebas/one/OneDrive/grive/faks/masters/y2/s1/BayesianStats/BayesianStats/hw1")

library(cmdstanr) # for interfacing Stan
library(ggplot2) # for visualizations
library(tidyverse) # for data manipulations
library(posterior) # for extracting samples
library(bayesplot) # for some quick MCMC visualizations
library(mcmcse) # for comparing samples and calculating MCSE
library(ggdist) # for visualizing distributions
library(rstan)

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

# Transform Betas
df_betas <- as_draws_df(fit$draws("betas"))
df_betas <- df_betas %>% select(-.chain, -.iteration, -.draw)
colnames(df_betas) <- colnames(X)

beta_mat <- as.matrix(df_betas)  # rows = iterations, cols = betas
colnames(beta_mat) <- colnames(X)

# Mean betas
betas <- matrix(colMeans(df_betas))
rownames(betas) <- colnames(X)
betas

# Convert to two columns for plotting
df_betas["(Intercept)"] <- NULL
df_betas_long <- df_betas %>% select(-year_rescaled, -population_rescaled, -gdp_per_capita_rescaled) %>%
  pivot_longer(everything(), names_to = "feature", values_to = "value")

windows()
ggplot(df_betas_long, aes(x = value, y = feature)) +
  stat_halfeye(point_interval = median_qi) +
  labs(title = "Posterior distributions of coefficients")
contrasts(data$continent)
############################ 1. Year vs. life exp. ######################################

# Get rescaled sizes 
df_betas$year_rescaled <- rescale(df_betas$year, min_year, max_year)
year_mean = mean(df_betas$year_rescaled)
year_sd = sd(df_betas$year_rescaled)
sprintf("Each year, life expectancy goes up by %.2f pm %.2f years", year_mean, year_sd)

############################ 2. GDP and population correlation to life exp #############.

df_betas$gdp_per_capita_rescaled <- rescale(df_betas$gdp_per_capita, min_gdp, max_gdp)
df_betas$population_rescaled <- rescale(df_betas$population, min_population, max_population)

beta_gdp <- df_betas$gdp_per_capita
beta_population <- df_betas$population

mean_gdp <- mean(beta_gdp)
mean_population <- mean(beta_population)

ci_gdp <- quantile(beta_gdp, c(0.025, 0.975))
ci_population <- quantile(beta_population, c(0.025, 0.975))

sprintf("GDP per capita effect: %.2f (95%% CI: %.2f, %.2f)", mean_gdp, ci_gdp[1], ci_gdp[2])
sprintf("Population effect: %.2f (95%% CI: %.2f, %.2f)", mean_population, ci_population[1], ci_population[2])

############################ 3. When am I going to die #############

colnames(X)
slo_gdp <- 58150  # PPP
slo_gdp_scaled <- (slo_gdp - min_gdp) / (max_gdp - min_gdp)

slo_pop <- 2120547
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

y_pred <- my_info %*% t(beta_mat)

y_mean <- mean(y_pred)
y_ci <- quantile(y_pred, c(0.025, 0.975))
sprintf("I will live till %.2f; CI: [%.2f, %.2f]", y_mean, y_ci[1], y_ci[2])

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
eu_le <- mat[ , 4]

mean_eu <- mean(eu_le)
mean_eu_africa <- mean_eu > mean(mat[,1])
mean_eu_americas <- mean_eu > mean(mat[,2])
mean_eu_asia <- mean_eu > mean(mat[,3])
mean_eu_oceania <- mean_eu > mean(mat[,5])

res <- c(as.integer(mean_eu_africa), as.integer(mean_eu_americas), as.integer(mean_eu_asia), as.integer(mean_eu_oceania))
res
# Average european will outlive average african, american, asian but not oceanian

############################ 5. Any european born in 2001 vs other 2001 ppl #############
eu_vs_africa <- eu_le > mat[, 1]
eu_vs_americas <- eu_le > mat[, 2]
eu_vs_asia <- eu_le > mat[, 3]
eu_vs_oceania <- eu_le > mat[, 5]

colMeans(mat)

# Check probabilities
mean(eu_vs_africa)
quantile(eu_vs_africa, c(0.025, 0.975))

mean(eu_vs_americas)
quantile(eu_vs_americas, c(0.025, 0.975))

mean(eu_vs_asia)
quantile(eu_vs_asia, c(0.025, 0.975))

mean(eu_vs_oceania)
quantile(eu_vs_oceania, c(0.025, 0.975))

df_long <- as.data.frame(mat) %>%
  pivot_longer(cols = everything(), names_to = "continent", values_to = "life_expectancy")

# Plot distributions
windows()
ggplot(df_long, aes(x = life_expectancy, fill = continent)) +
  geom_density(alpha = 0.4) +
  xlim(50, 90) +
  xlab("Predicted life expectancy") +
  ylab("Density") +
  ggtitle("Posterior distribution of life expectancy by continent") +
  theme_minimal()

# Europeans will still outlive everyone except oceaninans
diff_eu_africa    <- eu_le - mat[, 1]
diff_eu_americas  <- eu_le - mat[, 2]
diff_eu_asia      <- eu_le - mat[, 3]
diff_eu_oceania   <- eu_le - mat[, 5]

# Combine all differences into one vector for plotting
diff_eu_all <- c(diff_eu_africa, diff_eu_americas, diff_eu_asia, diff_eu_oceania)

prob <- mean(diff_eu_all > 0)   
ci <- quantile(diff_eu_all, c(0.025, 0.975))
sprintf("P(Europe > other continents) = %.2f%%", prob*100)

df <- data.frame(diff = diff_eu_all)

windows()
ggplot(df, aes(x = diff)) +
  geom_density(fill = "skyblue", alpha = 0.5) +
  geom_vline(xintercept = ci, color = "darkblue", linewidth=1) +  # CI lines
  geom_vline(xintercept = 0, linetype = "dashed", color = "red") +
  xlab("Life expectancy difference (Europe - other continents)") +
  ylab("Density") +
  ggtitle("Posterior distribution of Europe vs other continents")


######################################################

ggplot(data, aes(x = gdp_per_capita, y = y)) +
  geom_point() +
  geom_smooth(method = "lm", se = TRUE, color = "blue") +
  labs(title = "Life expectancy vs GDP per capita")

windows()
ggplot(data, aes(x = continent, y = y)) +
  geom_jitter(width = 0.2, alpha = 0.5) +
  geom_point(aes(y = y_mean), color = "red", size = 2) +
  labs(title = "Life expectancy by continent")


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

