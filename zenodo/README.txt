XGB2.9G: XGBOOST GROUND-MOTION MODEL FOR ROTD100 PGA
Archive README prepared 2 November 2026
Author: Oscar da Silva Costa

1. MODEL

XGB2.9G predicts RotD100 peak ground acceleration (PGA) in percent g (%g).
The final deployment model was fitted to 9,579 retained ESM records.
It uses separate Vs30 and epicentral-distance predictors and does not use
station longitude or the Vs30/epicentral-distance ratio as direct predictors.
Source-to-site direction is calculated consistently with the training encoding.

The primary model file for this version is XGB2.9G_FULL_DB.rds.
This is an inference-only caret model containing the fitted XGBoost trees,
formula and preprocessing. Embedded training records, training callbacks and
fit-specific environments have been removed. Predictions were verified to be
unchanged from the accepted original model on all 9,579 retained records.
The exported file is therefore not byte-identical to the original unsanitized
caret training object, although its predictions are identical.

If XGB2.9G_EVALUATION.rds is also included, it is the separate model fitted
on 8,236 training records for SCR development-validation assessment. Use the
FULL_DB model, not the evaluation model, for the manuscript scenario predictions.
Any accompanying *_preprocessing.csv files contain fitted centres/scales for
reference. Do not apply these manually before calling predict on the caret RDS.

2. REQUIRED RAW PREDICTORS

Name                         Definition / unit
Mw                           Moment magnitude
event_depth_km               Hypocentral depth, km
hypocentral_distance_km       Hypocentral distance, km
azimuth_cos                  Source-to-site directional cosine (definition below)
event_latitude               Epicentre latitude, WGS84 decimal degrees
event_longitude              Epicentre longitude, WGS84 decimal degrees
station_latitude             Site latitude, WGS84 decimal degrees
vs30_m_s                     Vs30, m/s
epicentral_distance_km        Epicentral distance, km

Station longitude is additionally needed when deriving distance and direction
from coordinates, even though it is not itself a direct model predictor.

Let a be event latitude, b station latitude and dl the station-minus-event
longitude difference, all in radians:

  east  = sin(dl) * cos(b)
  north = cos(a)*sin(b) - sin(a)*cos(b)*cos(dl)
  azimuth_cos = cos(atan2(north, east))

This is the initial spherical source-to-site direction referenced
counterclockwise from east, not the cosine of the usual north-referenced
bearing. Keep all event-dependent inputs consistent with the same earthquake.

3. SOFTWARE AND PREDICTION

Reference environment: R 4.4.2, caret 7.0-1, xgboost 1.7.11.1.
Newer major package versions are not claimed to be compatible or to yield
identical refits without verification.

Example R commands, using a CSV containing the nine raw predictors:

  library(caret)
  library(xgboost)
  model <- readRDS("XGB2.9G_FULL_DB.rds")
  inputs <- read.csv("prediction_input.csv", check.names=FALSE)
  inputs$RotD100_PGA_pctg <- as.numeric(predict(model, newdata=inputs))
  write.csv(inputs, "predictions.csv", row.names=FALSE)

The caret object applies fitted centering/scaling exactly once. Do not
pre-scale inputs. Output is already percent g (%g), not g. Do not multiply
by 100. The response is not log-transformed. No positivity clipping is applied.
The historical training target name rotD100_pga_g also refers to values in %g.
Always supply newdata; training records are not embedded in this export.

4. REPRODUCIBILITY AND SITE GRID

Source code, actual fitting scripts, prediction helpers, the municipal Vs30
grid, a complete Scenario A example and selected results are available at:
https://github.com/oscardasilvacosta/SCR-Ground-Motion-Model

The main site grid contains 1,375 centres inside the adopted Lisbon municipal
boundary, with unchanged Vs30 values and coordinates. The separately labelled
original grid contains 2,652 sites, including sites outside Lisbon. The nominal
250 m grid retains its original angular spacing; it is not newly resampled.
The static grid alone is not a complete model input: magnitude, source location,
depth and derived distance/direction predictors must also be supplied.

The retained ESM training records and original source workbook are not included
in the public release. Exact retraining requires legitimate access to the
processed retained input data. Downloading the upstream ESM flatfile alone does
not recover every integrated metadata field and preparation step. The upstream
database is identified by https://doi.org/10.13127/esm/flatfile.1.0.

5. EVALUATION AND LIMITATIONS

R2 below denotes squared Pearson correlation between observed and predicted PGA.
MAE and RMSE are expressed in percent-g percentage points.

Assessment                                      R2       MAE (%g)  RMSE (%g)
SCR recording-level validation (1,343 records)    0.8013   1.0759    3.3046
Strict event holdout, mean of 50 repetitions     0.4087   2.8468    5.9416
SCR event holdout, non-SCR retained, mean of 50   0.5399   1.8527    5.0293

The primary SCR subset was reused during configuration selection and is not
an untouched or event-disjoint test. The retained-non-SCR protocol can include
training recordings from SCR test earthquakes. The strict protocol removes
all recordings of the held-out earthquakes. The two protocols use identical
SCR test recordings within each repetition.

Model selection considered performance, input interpretability and spatial
behaviour, not the largest R2 or smallest error alone. Conditional bootstrap
intervals do not include model-refitting or model-selection uncertainty.
Split-to-split Monte Carlo variation is not a confidence interval.

The observed training magnitude range is Mw 4.5-7.8, with sparse support at
large magnitudes. Scenario H (Mw 8.7) is outside that range. Non-monotonic
conditional magnitude/distance behaviour and occasional nonpositive predictions
remain limitations. This is research software, not a certified seismic-hazard
or engineering-design tool. GMPE comparisons do not remove these limitations.

6. LINKS AND LICENSING

Associated Zenodo deposit: https://zenodo.org/records/18756043
Source code: https://github.com/oscardasilvacosta/SCR-Ground-Motion-Model

Use the published archive's version metadata and DOI when citing its specific
XGB2.9G release. This README does not assign a new DOI or imply that an earlier
version of the deposit contains the revised model.

The repository's MIT licence applies to its source code and software
documentation. It does not grant redistribution rights for third-party ESM
observations or geospatial source products. No training-data redistribution
permission is granted by this model README.
