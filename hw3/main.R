setwd("C:/Users/sebas/one/OneDrive/grive/faks/masters/y2/s1/BayesianStats/BayesianStats/hw3")
library(cmdstanr) # for interfacing Stan
library(ggplot2) # for visualizations
library(tidyverse) # for data manipulations
library(posterior) # for extracting samples
library(bayesplot) # for some quick MCMC visualizations
library(mcmcse) # for comparing samples and calculating MCSE
library(ggdist) # for visualizing distributions
library(rstan)
library(HDInterval)

df <- read.csv("no2.csv",stringsAsFactors=TRUE)
df$no2 <- df$no2 - mean(df$no2, na.rm = TRUE) / sd(df$no2, na.rm = TRUE)

colnames <- c("date", "no2")

acf(df$no2, na.action = na.pass, main = "ACF of NO2 levels")
# Signal
x <- df$no2
x <- x - mean(x, na.rm = TRUE)   # detrend

###############################
# Frequency Analysis
###############################

# FFT
x <- residuals(lm(no2 ~ as.numeric(as.Date(date)), data = df))
X <- fft(x)
n <- length(X)

# Frequency axis (example: daily data)
fs <- 365  # samples per year (adjust if needed)
freq <- (0:(n-1)) * fs / n

# One-sided spectrum
half <- 1:floor(n/2)

df_spec <- data.frame(
  frequency = freq[half],
  amplitude = Mod(X[half]) / n
)

ggplot(df_spec, aes(x = frequency, y = amplitude)) +
  geom_line(color = "red") +
  labs(
    title = "Frequency Spectrum of NO2 Levels",
    x = "Frequency (cycles/year)",
    y = "Amplitude"
  ) +
  theme_minimal()

# Find top 10 frequencies with highest amplitudes
top3_amps <- df_spec %>%
  arrange(desc(amplitude)) %>%
  slice(1:10)

# We noticed a trend around frequency ~52.12 -> weekly cycle
cycle <- 365 / 52.12
cycle

###############################
# Data analysis 
###############################

df_ <- df
df_$date <- as.Date(df_$date)
df_$week <- as.numeric(format(df_$date, "%U"))
df_$year <- as.numeric(format(df_$date, "%Y"))

windows()
ggplot(df_, aes(x=date, y=no2)) +
  geom_line(color = "blue") +
  labs(
    title = "NO2 Levels Over Time",
    x = "Date",
    y = "NO2 Levels") +
  theme_minimal()

install.packages("devtools")
library(devtools)
devtools::find_rtools()
##########################################
# We can see that there is a weekly cycle. Now we want to model the trend and seasonality.
##########################################

model <- cmdstan_model("harmonic.stan")

days_since_start <- as.numeric(df_$date - min(df_$date)) + 1
data_list <- list(
  n = nrow(df),
  t = days_since_start,
  y = df$no2,
  k = 1,
  omega = 2 * pi / cycle
)

fit <- model$sample(
  data = data_list,
  seed = 123,
  chains = 4,
  parallel_chains = 4,
  iter_warmup = 1000,
  iter_sampling = 1000
)

draws_df <- as_draws_df(fit$draws())

mcmc_trace(draws_df, pars = c("sigma", "alpha", "beta", "beta_cos[1]", "beta_sin[1]")) +
  ggtitle("MCMC Trace Plots for Harmonic Model Parameters")


fit$summary()

### Plot 
draws_df <- draws_df %>% select(-lp__, -.draw, -.chain, -.iteration)

df_decomp <- data.frame(
    idx=charachter(),
    type=charachter(),
    day=integer(),
    no2=numeric()
)
draws_df[1,]

colnames(draws_df) <- c("beta_cos", "beta_sin", "alpha", "beta", "sigma")

for (day in 1:nrow(draws_df)) {
    params <- draws_df[day, ]

    trend <- params$alpha + params$beta * day

    seasonal <- params$beta_cos * cos(df[day]$omega * day) + params$beta_sin * sin(df[day]$omega * day)
    
    df_decomp <- rbind(df_decomp, data.frame(
        idx = day,
        type = "trend",
        day = day,
        no2 = trend
    ))
    
    df_decomp <- rbind(df_decomp, data.frame(
        idx = day,
        type = "seasonal",
        day = day,
        no2 = seasonal
    ))
    
    df_decomp <- rbind(df_decomp, data.frame(
        idx = day,
        type = "observed",
        day = day,
        no2 = trend + seasonal
    ))
}

# Posterior means
params <- draws_df %>%
  summarise(
    alpha = mean(alpha),
    beta = mean(beta),
    beta_cos = mean(beta_cos),
    beta_sin = mean(beta_sin)
  )

n_days <- nrow(df)
omega <- 2 * pi / cycle
days <- 1:n_days

# Decompose trend and seasonal
df_decomp <- data.frame(
  day = rep(days, 3),
  type = rep(c("trend", "seasonal", "fitted"), each = n_days),
  no2 = c(
    params$alpha + params$beta * days,  # trend
    params$beta_cos * cos(omega * days) + params$beta_sin * sin(omega * days),  # seasonal
    (params$alpha + params$beta * days) + (params$beta_cos * cos(omega * days) + params$beta_sin * sin(omega * days))  # fitted
  )
)
df_decomp

windows()
ggplot(df_decomp, aes(x = day, y = no2, color = type)) +
  geom_line() +
  labs(title = "Trend + Seasonal Decomposition from Harmonic Model",
       x = "Day",
       y = "NO2 Level") +
  theme_minimal()

library(dplyr)
library(ggplot2)

# Posterior mean of parameters
params <- draws_df %>%
  summarise(
    alpha = mean(alpha),
    beta = mean(beta),
    beta_cos = mean(beta_cos),
    beta_sin = mean(beta_sin)
  )

# Days since start
days <- as.numeric(df$date - min(df$date)) + 1
omega <- 2 * pi / cycle

# Fitted values
fitted <- params$alpha + params$beta * days +
  params$beta_cos * cos(omega * days) +
  params$beta_sin * sin(omega * days)

df_plot <- data.frame(
  date = df$date,
  observed = df$no2,
  fitted = fitted
)
