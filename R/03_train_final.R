# ============================================
# R/03_train_final.R
# Final model trained on full database (no CV)
# ============================================

suppressPackageStartupMessages({
  library(readxl)
  library(dplyr)
  library(caret)
  library(xgboost)
  library(writexl)
  library(e1071)
  library(ggplot2)
  library(SHAPforxgboost)
  library(ggbeeswarm)
  library(viridis)
  library(scales)
})

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
cat("Raw data dimensions:", paste(dim(df), collapse=" x "), "\n")

# ----------------------------
# 2) Prepare data
# ----------------------------
final_data <- df %>%
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

cat("Final modeling data rows:", nrow(final_data), "\n")

# ----------------------------
# 3) Fixed hyperparameters
# ----------------------------
final_grid <- data.frame(
  nrounds = 1200,
  max_depth = 14,
  eta = 0.01,
  gamma = 0,
  colsample_bytree = 0.8,
  min_child_weight = 1,
  subsample = 0.9
)

final_ctrl <- trainControl(method = "none")

model_formula <- rotD100_pga_g ~ Mw + event_depth_km + hypocentral_distance_km +
  azimuth_cos + event_latitude + event_longitude +
  station_latitude + station_longitude + vs30_div_epi

# ----------------------------
# 4) Train final model (no CV)
# ----------------------------
final_model <- train(
  model_formula,
  data = final_data,
  method = "xgbTree",
  preProcess = c("center","scale"),
  trControl = final_ctrl,
  tuneGrid = final_grid
)

cat("Final model trained on full database.\n")
print(final_model$bestTune)

# In-sample sanity (do not report as generalization)
pred_train <- predict(final_model, newdata = final_data)
ins_rmse <- RMSE(pred_train, final_data$rotD100_pga_g)
ins_mae  <- MAE(pred_train,  final_data$rotD100_pga_g)
ins_r2   <- R2(pred_train,   final_data$rotD100_pga_g)
cat(sprintf("In-sample: RMSE=%.4f  MAE=%.4f  R2=%.4f\n", ins_rmse, ins_mae, ins_r2))

# ----------------------------
# 5) Variable importance (publication style)
# ----------------------------
pretty_names <- c(
  station_latitude        = "Station Latitude",
  hypocentral_distance_km = "Hypocentral Distance (km)",
  Mw                      = "Moment Magnitude (Mw)",
  azimuth_cos             = "Azimuth Station–Epicenter (cosine)",
  event_latitude          = "Event Latitude",
  event_longitude         = "Event Longitude",
  event_depth_km          = "Hypocentral Depth (km)",
  vs30_div_epi            = "Vs30/Re",
  station_longitude       = "Station Longitude"
)

xgb_varimp <- varImp(final_model)$importance
xgb_varimp$variable <- rownames(xgb_varimp)

xgb_varimp <- xgb_varimp %>%
  mutate(variable_pretty = dplyr::recode(variable, !!!pretty_names, .default = variable)) %>%
  arrange(desc(Overall)) %>%
  mutate(variable_pretty = factor(variable_pretty, levels = rev(unique(variable_pretty))))

p_imp <- ggplot(xgb_varimp, aes(x = variable_pretty, y = Overall)) +
  geom_col(width = 0.75) +
  coord_flip() +
  theme_classic(base_size = 12) +
  labs(
    title = "Variable Importance (Final XGB)",
    x = NULL,
    y = "Relative importance"
  )

ggsave(file.path(outputs_dir, "final_varimp.png"),
       p_imp, width = 7.2, height = 5.2, dpi = 600, bg = "white")

write_xlsx(xgb_varimp, file.path(outputs_dir, "final_varimp.xlsx"))

# ----------------------------
# 6) SHAP summary (uses SAME preprocessing as caret)
# ----------------------------
pred_vars <- c(
  "Mw", "event_depth_km", "hypocentral_distance_km",
  "azimuth_cos", "event_latitude", "event_longitude",
  "station_latitude", "station_longitude", "vs30_div_epi"
)

X_scaled_df <- predict(final_model$preProcess, final_data[, pred_vars])
X_mat <- as.matrix(X_scaled_df)

booster <- final_model$finalModel
shap_vals <- shap.values(xgb_model = booster, X_train = X_mat)
shap_long <- shap.prep(shap_contrib = shap_vals$shap_score, X_train = X_mat)

# Rename columns for clarity (depends on package version)
shap_long <- shap_long %>%
  rename(shap_value = value, feature_value_scaled = rfvalue) %>%
  mutate(
    variable_chr    = as.character(variable),
    variable_pretty = dplyr::recode(variable_chr, !!!pretty_names, .default = variable_chr)
  )

# Order features by mean(|SHAP|)
feat_order <- shap_long %>%
  group_by(variable_pretty) %>%
  summarize(mean_abs = mean(abs(shap_value), na.rm = TRUE), .groups = "drop") %>%
  arrange(desc(mean_abs)) %>%
  pull(variable_pretty)

shap_long$variable_pretty <- factor(shap_long$variable_pretty, levels = rev(feat_order))

# Scale color per feature (0..1)
shap_long <- shap_long %>%
  group_by(variable_pretty) %>%
  mutate(
    value_scaled = {
      rng <- range(feature_value_scaled, na.rm = TRUE)
      if (!is.finite(rng[1]) || !is.finite(rng[2]) || (rng[2] - rng[1]) == 0) 0.5
      else (feature_value_scaled - rng[1]) / (rng[2] - rng[1])
    }
  ) %>%
  ungroup()

p_shap <- ggplot(shap_long, aes(x = shap_value, y = variable_pretty)) +
  geom_violin(fill = "grey90", color = NA, alpha = 0.9, scale = "width", trim = TRUE) +
  ggbeeswarm::geom_quasirandom(aes(color = value_scaled), alpha = 0.75, size = 1.2) +
  geom_vline(xintercept = 0, linetype = "dashed", linewidth = 0.4) +
  scale_color_viridis_c(
    option = "C",
    breaks = c(0, 1),
    labels = c("Low", "High"),
    name = "Feature value (scaled)"
  ) +
  theme_classic(base_size = 12) +
  labs(title = "SHAP Summary (Final XGB)",
       x = "SHAP value (impact on prediction)",
       y = NULL)

ggsave(file.path(outputs_dir, "final_shap_summary.png"),
       p_shap, width = 8.2, height = 5.6, dpi = 600, bg = "white")

# ----------------------------
# 7) Save final model
# ----------------------------
saveRDS(final_model, file.path(models_dir, "xgb_final_full_db.rds"))

cat("\nDone. Outputs in /outputs and model in /models.\n")
