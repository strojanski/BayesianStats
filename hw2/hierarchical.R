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


fit$summary("beta_country")

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



###################################
# Health vs wealth
###################################delta_global <- draws[["beta_global[3]"]] - draws[["beta_global[2]"]]
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
    title = "Posterior Distributions of Continent-Level Slopes: Health vs Wealth"
  ) +
  scale_fill_manual(values = c("health" = "steelblue", "wealth" = "darkorange")) +
  theme_minimal() +
  theme(legend.title = element_blank())


