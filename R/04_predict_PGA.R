# ============================================
# R/04_predict_PGA.R
# Predict RotD100 PGA using the trained model
# (computes vs30_div_epi internally)
# ============================================

suppressPackageStartupMessages({
  library(readxl)
  library(dplyr)
  library(caret)
  library(writexl)
})

# ----------------------------
# 0) Paths
# ----------------------------
model_path   <- file.path("models", "xgb_final_full_db.rds")

# Put your prediction input file inside /data and set the filename here:
newdata_path <- file.path("data", "prediction_input.xlsx")

# Output written to /outputs
output_path  <- file.path("outputs", "predictions_rotd100_pga.xlsx")

dir.create("outputs", showWarnings = FALSE, recursive = TRUE)

stopifnot(file.exists(model_path))
stopifnot(file.exists(newdata_path))

# ----------------------------
# 1) Load model + new data
# ----------------------------
xgb_model <- readRDS(model_path)
new_data  <- read_excel(newdata_path)

cat("Model loaded from:", model_path, "\n")
cat("New data dimensions:", paste(dim(new_data), collapse=" x "), "\n")

# ----------------------------
# 2) Required raw inputs
#    (we compute vs30_div_epi from vs30_m_s and epicentral_distance_km)
# ----------------------------
required_raw <- c(
  "Mw",
  "event_depth_km",
  "hypocentral_distance_km",
  "azimuth_cos",
  "event_latitude",
  "event_longitude",
  "station_latitude",
  "station_longitude",
  "vs30_m_s",
  "epicentral_distance_km"
)

missing_vars <- setdiff(required_raw, names(new_data))
if(length(missing_vars) > 0){
  stop(
    "Missing required columns in new_data: ",
    paste(missing_vars, collapse = ", ")
  )
}

# ----------------------------
# 3) Feature engineering (must match training)
# ----------------------------
new_data <- new_data %>%
  mutate(
    vs30_div_epi = vs30_m_s / pmax(epicentral_distance_km, .Machine$double.eps)
  )

# ----------------------------
# 4) Predict
# ----------------------------
# caret::predict() will apply the same center/scale preprocessing stored in the model.
pred <- predict(xgb_model, newdata = new_data)

out <- new_data %>%
  mutate(predicted_rotD100_pga_g = pred)

# ----------------------------
# 5) Save output
# ----------------------------
write_xlsx(out, path = output_path)
cat("Predictions saved to:", output_path, "\n")
