# XGB2.9G: SCR Ground-Motion Model

Research code, fitted inference models and selected manuscript results for
**XGB2.9G**, an XGBoost model predicting raw **RotD100 peak ground acceleration
(PGA) in percent g (%g)**. This revision supersedes the XGB2.2G workflow previously
shown on this repository's main branch. Git history preserves the older code.

## What Changed

The revised model uses **separate Vs30 and epicentral-distance predictors** and
does not directly use station longitude or the Vs30/distance ratio. Nine inputs
remain. Source-to-site angular calculations use one consistent spherical
convention in training and prediction. Station longitude is still needed to
derive distance and direction. This is not a coordinate-free model.

Configuration selection considered predictive performance, predictor
interpretability and spatial behaviour in Lisbon scenario simulations. It was
not a selection solely by the largest R-squared or smallest error.

## Quick Prediction

The repository includes the final FULL_DB model and the separate evaluation
model. They are **inference-only caret objects**: embedded training records,
training callbacks and fit-specific environments have been removed. Their
predictions were checked against the original objects on all 9,579 retained
records and are unchanged. See [model verification](models/inference_export_verification.json).

Use the reference R environment described in [reproducibility](documentation/REPRODUCIBILITY.md):
R 4.4.2, caret 7.0-1 and xgboost 1.7.11.1. Different versions are not claimed to
give bitwise-identical refits.

```sh
# Complete saved Scenario A inputs, restricted to Lisbon municipal centres:
Rscript R/04_predict_PGA.R models/XGB2.9G_FULL_DB.rds examples/Scenario_A_complete_model_inputs.csv outputs/Scenario_A.csv

# Build event-dependent inputs from the static Vs30 grid:
Rscript R/05_predict_scenario_grid.R models/XGB2.9G_FULL_DB.rds data/sites/Lisbon_Vs30_prediction_grid.csv A outputs/Scenario_A_from_grid.csv

# Check the public prediction example:
Rscript tests/smoke_prediction.R
```

The grid has **1,375 site centres inside Lisbon**, with unchanged Vs30 values
and coordinates. CSV and single-sheet Excel versions are in `data/sites/`.
The original broader 2,652-point grid is separately labelled. The static grid
alone lacks the earthquake-specific predictors; the complete example includes
them. No interpolation or smoothing is applied by these prediction scripts.

Output values are already **%g**, not g. Do not multiply by 100, log-transform
the target or pre-scale inputs. The saved caret model applies its fitted
centering and scaling once. Nonpositive predictions are not silently clipped.

## Predictors

| Input | Unit / meaning |
| --- | --- |
| `Mw` | Moment magnitude |
| `event_depth_km` | Hypocentral depth, km |
| `hypocentral_distance_km` | Hypocentral distance, km |
| `azimuth_cos` | Initial source-to-site direction cosine, counterclockwise from east |
| `event_latitude`, `event_longitude` | Epicentre, WGS84 decimal degrees |
| `station_latitude` | Site latitude, WGS84 decimal degrees |
| `vs30_m_s` | Vs30, m/s |
| `epicentral_distance_km` | Epicentral distance, km |

The historical target name `rotD100_pga_g` is retained in training code but its
values are in **%g**. See [methods and limitations](documentation/METHODS.md)
for the exact angular definition and distance handling.

## Evaluation

The retained database contains 9,579 records: 4,479 SCR and 5,100 records
treated as non-SCR in the adopted strategy. The latter retain the original
`UNK` label in provenance; their non-SCR interpretation follows the authors'
classification, not the label text alone.

The evaluation fit uses 8,236 training records, with 1,343 SCR recordings
reserved for development validation. The final deployment fit uses all records.

| Assessment | Predictive R2 (1-SSE/SST) | Pearson r-squared | MAE (%g) | RMSE (%g) |
| --- | ---: | ---: | ---: | ---: |
| SCR recording-level validation, n=1,343 | 0.8012 | 0.8013 | 1.0759 | 3.3046 |
| Separate mixed recording-level validation, n=2,872 | 0.7016 | 0.7017 | 1.0074 | 3.2929 |
| Strict whole-event holdout, mean of 50 repetitions | 0.2968 | 0.4087 | 2.8468 | 5.9416 |
| SCR event holdout, non-SCR records retained, mean of 50 repetitions | 0.5118 | 0.5399 | 1.8527 | 5.0293 |

**These protocols are not interchangeable.** The recording-level SCR subset
was inspected during configuration selection and is not event-disjoint or an
untouched final test. The retained-non-SCR protocol can include training
recordings from held-out SCR test earthquakes. The strict protocol removes
every recording of the held-out earthquakes.

Fixed-model bootstrap intervals, permutation diagnostics, paired Monte Carlo
results and station comparisons are in [results](results/README.md). Bootstrap
intervals are conditional on the fitted model and selected validation data.
They do not account for refitting or model-selection uncertainty. Permutation
results do not prove absence of leakage.

## Reproduction and Data Access

```sh
# Requires the authors' processed source workbook, obtained with permission:
Rscript R/01_prepare_data.R /path/to/private_source.xlsx /path/to/new_private_prepared_data

# Or supply the canonical retained training CSV directly:
Rscript R/02_train_validate.R /path/to/private_training_database.csv outputs/new_evaluation
Rscript R/03_train_final.R /path/to/private_training_database.csv outputs/new_full_fit
```

The ESM-derived training records, record-level ESM predictions and original
workbook are **not distributed here**. The upstream ESM flatfile is identified
by [DOI 10.13127/esm/flatfile.1.0](https://doi.org/10.13127/esm/flatfile.1.0).
Downloading that flatfile alone does not reconstruct the exact processed input
table and integrated metadata. Exact retraining therefore remains conditional
on legitimate access to the retained input data. See [data availability](data/README.md).

Initial development included five-fold cross-validation and grid search, with
limited manual refinements reported by the authors. The accepted fixed-setting
refits do not rerun that search. The complete historical candidate-search trace
is not archived here; `R/09_grid_search.R` performs a clearly labelled **new**
training-only five-fold search when supplied a candidate grid.

## Repository Guide

- `R/`: data preparation, actual model fitting, prediction, diagnostics and SHAP.
- `python/`: conditional bootstrap and result plotting.
- `config/`: adopted parameters, scenario definitions and non-observational split metadata.
- `models/`: inference-only fitted models and preprocessing parameters.
- `data/sites/`: static Lisbon Vs30 grids, not ESM training data.
- `examples/`: complete Scenario A inputs and expected prediction checks.
- `results/`: selected public scenario/station outputs, aggregate validation tables and figures.
- `tests/`: prediction, input-validation and release-content checks.

## Limits and Archive Status

This is a research model, not a certified seismic-hazard or engineering-design
tool. Sparse large-magnitude support, geographic dependence, non-monotonic
conditional scaling and occasional nonpositive predictions remain limitations.
Scenario H (Mw 8.7) is outside the observed training magnitude range, 4.5-7.8.
Good agreement in empirical binned trends does not establish universal physical
scaling or prediction uncertainty.

The older [Zenodo archive, DOI 10.5281/zenodo.18756043](https://doi.org/10.5281/zenodo.18756043),
was linked to XGB2.2G. **This GitHub update does not update that Zenodo record**
and does not assert that it contains XGB2.9G. A corresponding new archive
version must be published separately before claiming a version-specific DOI.

The [MIT licence](LICENSE) applies to source code and software documentation.
It does not grant redistribution rights for third-party ESM or geospatial data.
See [licence scope](documentation/LICENSING.md) and [citation metadata](CITATION.cff).
