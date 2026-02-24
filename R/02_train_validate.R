# R/02_train_validate.R
# Train + validate with SCR hold-out test set

suppressPackageStartupMessages({
  library(readxl)
  library(dplyr)
  library(caret)
  library(xgboost)
  library(writexl)
  library(e1071)
  library(ggplot2)
  library(SHAPforxgboost)
})

# ----------------------------
# 0) Paths + helpers
# ----------------------------
set.seed(123)

data_path   <- file.path("data", "updated_older_DB.xlsx")
outputs_dir <- "outputs"
models_dir  <- "models"

dir.create(outputs_dir, showWarnings = FALSE, recursive = TRUE)
dir.create(models_dir,  showWarnings = FALSE, recursive = TRUE)

stopifnot(file.exists(data_path))

# ----------------------------
# 1) Load data
# ----------------------------
df <- read_excel(data_path)
cat("Raw data dimensions:", paste(dim(df), collapse = " x "), "\n")

# ----------------------------
# 2) Prepare data
# ----------------------------
model_data <- df %>%
  filter(
    !is.na(rotD100_pga_g),
    !is.na(Mw),
    !is.na(event_depth_km),
    !is.na(epicentral_distance_km),
    !is.na(hypocentral_distance_km),
    !is.na(vs30_m_s),
    !is.na(azimuth_cos),
    !is.na(azimuth_sin),
    !is.na(event_latitude),
    !is.na(event_longitude),
    !is.na(station_latitude),
    !is.na(station_longitude),
    !is.na(tectonic_setting_station)
  ) %>%
  select(
    rotD100_pga_g, Mw, event_depth_km,
    epicentral_distance_km, hypocentral_distance_km,
    vs30_m_s, azimuth_cos, azimuth_sin,
    event_latitude, event_longitude,
    station_latitude, station_longitude,
    tectonic_setting_station
  ) %>%
  mutate(
    tectonic_setting_station = as.factor(tectonic_setting_station),
    vs30_div_epi = vs30_m_s / pmax(epicentral_distance_km, .Machine$double.eps)
  )

cat("Modeling data rows:", nrow(model_data), "\n")

# ----------------------------
# 3) Split SCR vs non-SCR
# ----------------------------
scr_data <- model_data %>%
  filter(toupper(as.character(tectonic_setting_station)) == "SCR")

non_scr_data <- model_data %>%
  filter(toupper(as.character(tectonic_setting_station)) != "SCR")

cat("SCR recordings:", nrow(scr_data), "\n")
cat("Non-SCR recordings:", nrow(non_scr_data), "\n")

stopifnot(nrow(scr_data) > 10)  # basic sanity

# ----------------------------
# 4) Stratified SCR split (70/30)
# ----------------------------
scr_train_index <- createDataPartition(scr_data$rotD100_pga_g, p = 0.7, list = FALSE)
scr_train <- scr_data[scr_train_index, ]
scr_test  <- scr_data[-scr_train_index, ]

train_data <- bind_rows(non_scr_data, scr_train)
test_data  <- scr_test

cat("Training set:", paste(dim(train_data), collapse=" x "), "\n")
cat("SCR test set:", paste(dim(test_data),  collapse=" x "), "\n")

# ----------------------------
# 5) Define model formula + CV
# ----------------------------
model_formula <- rotD100_pga_g ~ Mw + event_depth_km + hypocentral_distance_km +
  azimuth_cos + event_latitude + event_longitude +
  station_latitude + station_longitude + vs30_div_epi

control <- trainControl(method = "cv", number = 5)

# Fixed hyperparameters selected from prior tuning
xgb_grid <- expand.grid(
  nrounds = 1200,
  max_depth = 14,
  eta = 0.01,
  gamma = 0,
  colsample_bytree = 0.8,
  min_child_weight = 1,
  subsample = 0.9
)

# ----------------------------
# 6) Train model
# ----------------------------
xgb_model <- train(
  model_formula,
  data = train_data,
  method = "xgbTree",
  preProcess = c("center", "scale"),
  trControl = control,
  tuneGrid = xgb_grid
)

# ----------------------------
# 7) Evaluate on SCR test set
# ----------------------------
xgb_pred_scr <- predict(xgb_model, newdata = test_data)

scr_rmse <- RMSE(xgb_pred_scr, test_data$rotD100_pga_g)
scr_mae  <- MAE(xgb_pred_scr,  test_data$rotD100_pga_g)
scr_r2   <- R2(xgb_pred_scr,   test_data$rotD100_pga_g)

cat("\nPerformance on SCR hold-out test set:\n")
print(data.frame(Dataset = "SCR Test Set", RMSE = scr_rmse, MAE = scr_mae, R2 = scr_r2))

# Compare caret CV vs test
best <- xgb_model$bestTune
cv_row <- dplyr::inner_join(xgb_model$results, best, by = names(best)) %>% dplyr::slice(1)

comparison_results <- rbind(
  data.frame(Dataset = "CV (mean over folds)", RMSE = cv_row$RMSE, MAE = NA, R2 = cv_row$Rsquared),
  data.frame(Dataset = "SCR Test Set", RMSE = scr_rmse, MAE = scr_mae, R2 = scr_r2)
)

cat("\nCV vs Test comparison:\n")
print(comparison_results)

write_xlsx(comparison_results, file.path(outputs_dir, "cv_vs_scr_test.xlsx"))

# ----------------------------
# 8) Bootstrap CIs on SCR test set
# ----------------------------
bootstrap_eval <- function(data, model, n = 1000){
  rmse_vals <- numeric(n)
  mae_vals  <- numeric(n)
  r2_vals   <- numeric(n)

  for(i in seq_len(n)){
    idx <- sample.int(nrow(data), size = nrow(data), replace = TRUE)
    samp <- data[idx, , drop = FALSE]
    preds <- predict(model, newdata = samp)
    rmse_vals[i] <- RMSE(preds, samp$rotD100_pga_g)
    mae_vals[i]  <- MAE(preds,  samp$rotD100_pga_g)
    r2_vals[i]   <- R2(preds,   samp$rotD100_pga_g)
  }

  list(rmse = rmse_vals, mae = mae_vals, r2 = r2_vals)
}

B <- 1000
boot <- bootstrap_eval(test_data, xgb_model, n = B)

boot_summary <- data.frame(
  Metric = c("RMSE","MAE","R2"),
  Mean   = c(mean(boot$rmse), mean(boot$mae), mean(boot$r2)),
  CI_low = c(quantile(boot$rmse, 0.025), quantile(boot$mae, 0.025), quantile(boot$r2, 0.025)),
  CI_high= c(quantile(boot$rmse, 0.975), quantile(boot$mae, 0.975), quantile(boot$r2, 0.975))
)

print(boot_summary)
write_xlsx(boot_summary, file.path(outputs_dir, "scr_test_bootstrap_summary.xlsx"))

plot_boot <- function(vals, metric, filename){
  dfp <- data.frame(val = vals)
  p <- ggplot(dfp, aes(x = val)) +
    geom_histogram(aes(y = ..density..), bins = 40, alpha = 0.7) +
    geom_density(linewidth = 0.8) +
    theme_minimal() +
    labs(title = paste0("Bootstrap Distribution: ", metric),
         x = metric, y = "Density")
  ggsave(file.path(outputs_dir, filename), p, width = 8, height = 6, dpi = 300)
}

plot_boot(boot$rmse, "RMSE", "boot_RMSE.png")
plot_boot(boot$mae,  "MAE",  "boot_MAE.png")
plot_boot(boot$r2,   "R2",   "boot_R2.png")

# ----------------------------
# 9) Predicted vs Real + export predictions
# ----------------------------
results_df <- test_data %>%
  mutate(Predicted_rotD100_pga_g = xgb_pred_scr) %>%
  select(rotD100_pga_g, Predicted_rotD100_pga_g)

p_pred_vs_real <- ggplot(results_df, aes(x = rotD100_pga_g, y = Predicted_rotD100_pga_g)) +
  geom_point(alpha = 0.7) +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed") +
  theme_minimal() +
  labs(title = "Predicted vs. Real RotD100 PGA (SCR Test Set)",
       x = "Real RotD100 PGA (g)",
       y = "Predicted RotD100 PGA (g)")

ggsave(file.path(outputs_dir, "pred_vs_real_scr_test.png"),
       plot = p_pred_vs_real, width = 8, height = 6, dpi = 300)

write_xlsx(results_df, file.path(outputs_dir, "pred_vs_real_scr_test.xlsx"))

# ----------------------------
# 10) Variable importance
# ----------------------------
xgb_varimp <- varImp(xgb_model)$importance
xgb_varimp$Variable <- rownames(xgb_varimp)
xgb_varimp <- xgb_varimp %>% arrange(desc(Overall))

p_imp <- ggplot(xgb_varimp, aes(x = reorder(Variable, Overall), y = Overall)) +
  geom_bar(stat = "identity") +
  coord_flip() +
  theme_minimal() +
  labs(title = "Variable Importance (XGB)",
       x = "Predictor", y = "Importance")

ggsave(file.path(outputs_dir, "xgb_variable_importance.png"),
       plot = p_imp, width = 8, height = 6, dpi = 300)

write_xlsx(xgb_varimp, file.path(outputs_dir, "xgb_variable_importance.xlsx"))

# ----------------------------
# 11) Save model
# ----------------------------
saveRDS(xgb_model, file = file.path(models_dir, "xgb_model_train_validate.rds"))

# ----------------------------
# 12) SHAP summary 
# ----------------------------
train_matrix <- model.matrix(
  ~ Mw + event_depth_km + hypocentral_distance_km + azimuth_cos + event_latitude + event_longitude +
    station_latitude + station_longitude + vs30_div_epi - 1,
  data = train_data
)

xgb_booster <- xgb_model$finalModel
shap_values <- shap.values(xgb_model = xgb_booster, X_train = train_matrix)
shap_long   <- shap.prep(shap_contrib = shap_values$shap_score, X_train = train_matrix)

png(filename = file.path(outputs_dir, "xgb_shap_summary.png"), width = 2000, height = 1600, res = 300)
shap.plot.summary(shap_long)
dev.off()

cat("\nDone. Outputs in /outputs and model in /models.\n")
