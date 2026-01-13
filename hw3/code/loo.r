library(loo)

setwd("C:/Users/sebas/one/OneDrive/Namizje/repos/BayesianStats/hw3")
log_lik_quad <- as.matrix(read.csv("code/log_lik_quad.csv"))
log_lik_quad
loo_res_quad <- loo(log_lik_quad[,1:10373]) 
ncol(log_lik_quad)
nrow(log_lik_quad)
loo_res_quad


log_lik_spl <- as.matrix(read.csv("code/log_lik_simple.csv"))
log_lik_spl
loo_res_spl <- loo(log_lik_spl) 

loo_res_spl


log_lik_long <- as.matrix(read.csv("code/log_lik_longseason.csv"))
log_lik_long
loo_res_long <- loo(log_lik_long) 

loo_res_long


loo_compare(loo_res_spl, loo_res_long, loo_res_quad)
loo_res_quad$se_looic

df = read.csv("no2.csv",stringsAsFactors=TRUE)
min(df$no2)
max(df$no2)

library(ggplot2)

# Collect results
model_names <- c("Simple", "Multi-seasonal", "Quadratic")
looic_vals  <- c(loo_res_spl$estimates["looic","Estimate"],
                 loo_res_long$estimates["looic","Estimate"],
                 loo_res_quad$estimates["looic","Estimate"])

looic_se    <- c(loo_res_spl$estimates["looic","SE"],
                 loo_res_long$estimates["looic","SE"],
                 loo_res_quad$estimates["looic","SE"])

df_plot <- data.frame(
  Model = model_names,
  LOOIC = looic_vals,
  SE    = looic_se
)

# Plot
ggplot(df_plot, aes(x = Model, y = LOOIC)) +
  geom_point(size = 4) +
  geom_errorbar(aes(ymin = LOOIC - SE, ymax = LOOIC + SE), width = 0.15, size = 1) +
  theme_minimal(base_size = 14) +
    scale_x_discrete(limits = c("Simple", "Multi-seasonal", "Quadratic")) +
  labs(
    y = "LOOIC (± SE)",
    x = ""
  ) + 
  theme_minimal(base_size = 14) +
  theme(
    axis.text.x  = element_text(size = 16),
    axis.text.y  = element_text(size = 16),
    axis.title.y = element_text(size = 18, face = "bold"),
  )
ggsave(
  filename = "looic_comparison.jpg",
  plot = last_plot(),
#   width = 6,
#   height = 7,
  dpi = 300
)

looic_vals <- c(
  simple = loo_res_spl$looic,
  multiseasonal = loo_res_long$looic,
  quadratic = loo_res_quad$looic
)

delta <- looic_vals - min(looic_vals)
weights <- exp(-0.5 * delta) / sum(exp(-0.5 * delta))

weights
