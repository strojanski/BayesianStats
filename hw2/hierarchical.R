setwd("C:/Users/sebas/one/OneDrive/grive/faks/masters/y2/s1/BayesianStats/BayesianStats/hw2")
library(cmdstanr) # for interfacing Stan
library(ggplot2) # for visualizations
library(tidyverse) # for data manipulations
library(posterior) # for extracting samples
library(bayesplot) # for some quick MCMC visualizations
library(mcmcse) # for comparing samples and calculating MCSE
library(ggdist) # for visualizing distributions
library(rstan)
library(HDInterval)


cantril <- read_csv("cantril_ladder.csv")
cantril <- cantril %>%
  mutate(country = as.factor(country),
         continent = as.factor(continent))


countries_df <- read.csv("countries.csv") %>%
    mutate(country = as.factor(country))

df <- cantril %>% left_join(countries_df, by="country")

df

unique(df$country)
df
mean(df$year)
mean(df$life_expectancy)
mean(exp(df$log_gdp))

# Center year, normalize life expectancy and log_gdp
df <- df %>%
  mutate(year_c = year - mean(year),
         lifeExp_n = (life_expectancy - mean(life_expectancy)) / sd(life_expectancy),
         log_gdp_n = (log_gdp - mean(log_gdp)) / sd(log_gdp))


df <- df %>%
  mutate(
    country_id   = as.integer(factor(country)),
    continent_id = as.integer(factor(continent))
  )


stan_data <- list(
  N = nrow(df),
  y = df$score,
  year = df$year_c,
  log_gdp = df$log_gdp_n,
  life_exp = df$lifeExp_n,

  C = n_distinct(df$country_id),
  K = n_distinct(df$continent_id),

  country = df$country_id,
  continent = tapply(df$continent_id, df$country_id, unique)
)


model <- cmdstan_model("hierarchical_model.stan")


fit <- model$sample(
  data = stan_data,
  seed = 42,
  chains = 4,
  parallel_chains = 4,
)

mcmc_trace(fit$draws(variables = c("alpha_global", "beta_global[1]", "beta_global[2]", "beta_global[3]"))) +
  ggtitle("Trace plots for global parameters")


mcmc_trace(fit$draws(variables = c("sigma_y", "sigma_alpha_cont", "sigma_beta_cont"))) +
  ggtitle("Trace plots for standard deviations")

mcmc_trace(fit$draws(variables = c("alpha_country[1]", "beta_country[1,1]", "beta_country[1,2]", "beta_country[1,3]"))) +
  ggtitle("Trace plots for country-level parameters (first country)")

windows()
fit$summary("beta_country")
fit$summary("beta_cont")

mcse(fit$draws("alpha_country"))

draws_df <- as_draws_df(fit$draws())

# year trend (global)
mcse(draws_df["beta_global[1]"] > 0)

# gdp trend (global)
mcse(draws_df["beta_global[2]"] > 0)

# life expectancy trend (global)
mcse(draws_df["beta_global[3]"] > 0)


###################################
# Are humans globally happy today?
###################################

year_2025_c <- 2025 - mean(df$year)
alpha_g <- draws_df["alpha_global"]
beta_year <- draws_df["beta_global[1]"]
alpha_g
mean(as.numeric(beta_year["beta_global[1]"]))
mu_2025 <- alpha_g + beta_year * year_2025_c
mcse(mu_2025 > 5)



###################################
# Slovenia top 20 in 2025?
###################################

slovenia_id <- which(levels(factor(df$country)) == "Slovenia")
country_covars <- df %>%
  group_by(country_id) %>%
  summarise(
    log_gdp_n = mean(log_gdp_n),
    lifeExp_n = mean(lifeExp_n)
  ) %>%
  arrange(country_id)

draws <- as_draws_df(fit$draws())
S <- nrow(draws)
C <- nrow(country_covars)

mu_2025 <- matrix(NA, nrow = S, ncol = C)

for (c in 1:C) {
  mu_2025[, c] <-
    draws[[paste0("alpha_country[", c, "]")]] +
    draws[[paste0("beta_country[", c, ",1]")]] * year_2025_c +
    draws[[paste0("beta_country[", c, ",2]")]] * country_covars$log_gdp_n[c] +
    draws[[paste0("beta_country[", c, ",3]")]] * country_covars$lifeExp_n[c]
}

slovenia_rank <- apply(mu_2025, 1, function(x)
  rank(-x, ties.method = "average")[slovenia_id]
)

slovenia_rank
mcse(slovenia_rank <= 20)
mcse(slovenia_rank)
mode(slovenia_rank)
get_mode <- function(v) {
  uniqv <- unique(v)
  uniqv[which.max(tabulate(match(v, uniqv)))]
}
get_mode(slovenia_rank)
###################################
# Health vs wealth
###################################
delta_global <- draws[["beta_global[3]"]] - draws[["beta_global[2]"]]
continents <- levels(factor(df$continent))

delta_vs_global <- sapply(seq_along(continents), function(k) {
  (draws[[paste0("beta_cont[", k, ",3]")]] -
   draws[[paste0("beta_cont[", k, ",2]")]]) -
   delta_global
})

colnames(delta_vs_global) <- continents
apply(delta_vs_global, 2, function(d) mcse(d > 0))

# Prepare long-format data (same as before)
df_plot <- as.data.frame(delta_vs_global) %>%
  pivot_longer(cols = everything(),
               names_to = "continent",
               values_to = "delta_vs_global")

# Stacked vertically using facets
ggplot(df_plot, aes(x = delta_vs_global)) +
  geom_density(fill = "steelblue", alpha = 0.6) +
  geom_vline(xintercept = 0, linetype = "dashed") +
  facet_wrap(~continent, ncol = 1, scales = "free_y") +
  labs(
    x = "Continent vs Global Health–Wealth Contrast",
    y = "Density",
    title = "Posterior Distributions of Continent vs Global Contrast (Health – Wealth)"
  ) +
  theme_minimal()


posterior_df <- bind_rows(posterior_list)

# Convert to long format
posterior_long <- posterior_df %>%
  pivot_longer(cols = c("health", "wealth"),
               names_to = "variable",
               values_to = "slope")

# Ridgeline plot: health and wealth colored differently per continent
ggplot(posterior_long, aes(x = slope, y = continent, fill = variable)) +
  geom_density_ridges(alpha = 0.7, scale = 1, position = "identity") +
  geom_vline(xintercept = 0, linetype = "dashed") +
  labs(
    x = "Slope (Posterior Draws)",
    y = "",
  ) +
  scale_fill_manual(values = c("health" = "steelblue", "wealth" = "darkorange")) +
  theme_minimal() +
  theme(legend.title = element_blank())




library(ggridges)

# Extract posterior draws
draws_df <- as_draws_df(fit$draws())
continents <- levels(factor(df$continent))
K <- length(continents)

# Create a list of continent draws
posterior_list <- lapply(1:K, function(k) {
  data.frame(
    continent = continents[k],
    health = draws_df[[paste0("beta_cont[", k, ",3]")]],
    wealth = draws_df[[paste0("beta_cont[", k, ",2]")]]
  )
})

posterior_df <- bind_rows(posterior_list)

posterior_list
posterior_df <- posterior_df %>%
  mutate(continent = factor(continent))  # ensures unique grouping


table_df <- posterior_df %>%
  group_by(continent) %>%
  summarise(
    mean_LE      = mean(health),
    mean_GDP     = mean(wealth),
    P_LE_lt_GDP  = mean(health < wealth),
    mcse_LE      = as.numeric(mcse(health)[1]),
    mcse_LE_se   = as.numeric(mcse(health)[2]),
    mcse_GDP     = as.numeric(mcse(wealth)[1]),
    mcse_GDP_se  = as.numeric(mcse(wealth)[2]),
    diff = as.numeric(mcse(health < wealth)[1]),
    diff_se = as.numeric(mcse(health < wealth)[2])
  )


table_df$diff
table_df$diff_se
# Convert to long format
posterior_long <- posterior_df %>%
  pivot_longer(cols = c("health", "wealth"),
               names_to = "variable",
               values_to = "slope")

# Ridgeline plot: health and wealth colored differently per continent
ggplot(posterior_long, aes(x = slope, y = continent, fill = variable)) +
  geom_density_ridges(alpha = 0.7, scale = 1, position = "identity") +
  geom_vline(xintercept = 0, linetype = "dashed") +
  labs(
    x = "Slope (Posterior Draws)",
    y = "",
    title = "Posterior Distributions of Continent-Level Slopes: Health vs Wealth"
  ) +
  scale_fill_manual(values = c("health" = "steelblue", "wealth" = "darkorange")) +
  theme_minimal() +
  theme(legend.title = element_blank())



year_grid <- seq(min(df$year_c), max(df$year_c), length.out = 50)
C <- stan_data$C
draws <- as_draws_df(fit$draws())

alpha_hat <- sapply(1:C, function(c)
  mean(draws[[paste0("alpha_country[", c, "]")]])
)

beta_year_hat <- sapply(1:C, function(c)
  mean(draws[[paste0("beta_country[", c, ",1]")]])
)
fit_df <- expand.grid(
  country_id = 1:C,
  year_c = year_grid
)

fit_df$mu <- with(fit_df,
  alpha_hat[country_id] + beta_year_hat[country_id] * year_c
)

fit_df$country <- levels(factor(df$country))[fit_df$country_id]

windows()
ggplot(fit_df, aes(x = year_c + mean(df$year), y = mu, group = country)) +
  geom_line(alpha = 0.75, color = "steelblue") +
  labs(
    x = "Year",
    y = "Mean fitted happiness score",
    title = "Posterior Mean Country-Level Happiness Trends"
  ) +
  theme_minimal()

slovenia <- which(levels(factor(df$country)) == "Slovenia")

ggplot(fit_df, aes(x = year_c + mean(df$year), y = mu, group = country)) +
  geom_line(alpha = 0.7, color = "grey70") +
  geom_line(
    data = subset(fit_df, country_id == slovenia),
    color = "red",
    linewidth = 1.2
  ) +
  theme_minimal()


alpha_g <- mean(draws[["alpha_global"]])
beta_g  <- mean(draws[["beta_global[1]"]])

global_df <- data.frame(
  year = year_grid + mean(df$year),
  mu = alpha_g + beta_g * year_grid
)



# Posterior probability Slovenia is top 20
top20_prob <- mean(slovenia_rank <= 20)
top20_mcse <- mcse(as.numeric(slovenia_rank <= 20))

cat("Posterior probability Slovenia in top 20:", round(top20_prob, 3), 
    "±", round(top20_mcse[1], 3), "\n")

windows()
# Histogram of Slovenia ranks
ggplot(data.frame(rank = slovenia_rank), aes(x = rank)) +
  geom_histogram(binwidth = 1, fill = "skyblue", color = "black") +
  geom_vline(xintercept = 20, linetype = "dashed", color = "red") +
  labs(
    x = "Rank (1 = happiest)",
    y = "Number of posterior draws"
  ) +
  theme_minimal()




years <- 2000:2025
year_center <- years - mean(df$year)

# Extract global draws
alpha_g <- draws_df$alpha_global
beta_year <- draws_df[["beta_global[1]"]]

pred_global <- sapply(year_center, function(yc) {
  alpha_g + beta_year * yc
})

pred_summary <- data.frame(
  year = years,
  mean_happy = apply(pred_global, 2, mean),
  lower = apply(pred_global, 2, quantile, 0.025),
  upper = apply(pred_global, 2, quantile, 0.975)
)

# Plot
ggplot(pred_summary, aes(x = year, y = mean_happy)) +
  geom_line(color = "blue") +
  geom_ribbon(aes(ymin = lower, ymax = upper), alpha = 0.2, fill = "blue") +
  labs(
    title = "Global predicted happiness over years",
    x = "Year",
    y = "Predicted mean happiness (Cantril ladder)"
  ) +
  theme_minimal()

