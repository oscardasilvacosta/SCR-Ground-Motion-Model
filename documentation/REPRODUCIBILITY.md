# Reproducibility

## Reference Environment

The saved fits were produced with R 4.4.2, caret 7.0-1 and xgboost 1.7.11.1,
using four XGBoost threads. `config/R_sessionInfo.txt` records the environment
used to verify the public exports. The original analysis also used jsonlite
1.8.9. `readxl` is needed only for source-workbook preparation.

Package versions matter for serialization, caret's model adapter and fitting.
Installing current packages is not a promise of exact reconstruction. Use the
reference versions in an isolated R library. Historical versions are available
through the official CRAN source archives for
[xgboost](https://cran.r-project.org/src/contrib/Archive/xgboost/) and
[caret](https://cran.r-project.org/src/contrib/Archive/caret/).
Building source packages may require the matching compiler toolchain.
This repository does not install software automatically or bundle R packages.

Python bootstrap code requires NumPy and pandas; optional result plotting also
requires Matplotlib. The bootstrap method uses `numpy.random.default_rng`, seed
20261020, recording resampling first and event-cluster resampling second.

## What Can Be Reproduced Without Restricted Data

- FULL_DB inference from the supplied municipal grid and scenario definitions.
- All six Lisbon scenario predictions, checked against retained outputs.
- The complete Scenario A example and its expected prediction values.
- Station-summary errors and ratios from the supplied ten station cases.
- Checks of published aggregate bootstrap intervals and Monte Carlo summaries.

The fitted public objects retain the learned trees, formula and preprocessing,
not the original training records, training callbacks, resampling predictions,
or original evaluation model's fitted-data payload. Export checks compare both
public models to the original models on all 9,579 private records. Zero observed
prediction difference was required. Public file hashes consequently differ from
the original unsanitized caret RDS hashes.

## Exact Retraining

`R/01_prepare_data.R` applies the retained complete-case rule to the authors'
source workbook, preserves input distances and site values, and harmonizes the
angular encoding. It checks for 9,579 retained rows, assigning canonical record
IDs in source order. It does not download, invent or reverse-engineer restricted
records. That workbook already incorporates upstream preparation and metadata.

Alternatively, pass the canonical retained `training_database.csv` to the
training scripts. It must contain `record_id`, the nine raw predictors,
`station_longitude`, and `rotD100_pga_g` already expressed in percent g.
`config/record_partition.csv` contains only record indices, group/split labels,
training order and anonymous event-group codes, not observed ground motions or
the per-record geographic/site feature table. Training order is significant.

The evaluation fit uses set.seed(123), caret method `none` and final-fit seed
365209. The full-data fit uses set.seed(123), method `none`. Both use the same
fixed parameters. Centering and scaling are fitted to the selected training
data only. No validation records enter evaluation-model fitting.

`R/02_train_validate.R` and `R/03_train_final.R` perform actual fitting, not
just loading the published model or checking predictions. They write into a new
directory and do not overwrite accepted model artifacts. New output model
objects are also stripped of record-level training payloads.

The original historical grid-search trial sequence and manual refinements are
not reconstructed by these scripts. `R/09_grid_search.R` is an optional new
five-fold search; its candidate-grid CSV must have the seven hyperparameter
columns listed in `config/fixed_hyperparameters.csv` and multiple combinations.

## Additional Analyses

```sh
Rscript R/06_event_monte_carlo.R /private/training_database.csv outputs/new_MC50
Rscript R/07_diagnostics.R mixed /private/training_database.csv outputs/new_mixed
Rscript R/07_diagnostics.R permutation /private/training_database.csv outputs/new_permutation
Rscript R/08_shap.R models/XGB2.9G_EVALUATION.rds /private/validation_inputs.csv outputs/new_SHAP
python python/bootstrap_metrics.py /private/validation_predictions.csv outputs/new_bootstrap
```

Monte Carlo uses the same 50 saved row-index partitions as the adopted paired
analysis and seeds 20261009 + repetition. The saved event selection originally
used event Mw quartiles crossed with median SCR PGA quartiles. It must not be
described as outcome-blind event sampling. The retained protocol keeps all
non-SCR rows; the strict protocol excludes all held-out-event rows. Both are
scored on identical SCR test records. No full source event coordinates or
observed PGA values are embedded in the index file.

The bootstrap helper joins private predictions to anonymous event clusters,
keeps all selected-event recordings together and reports pointwise percentile
intervals conditional on the fitted model. It neither refits nor fixes
training/test earthquake overlap. The exact historical folds are not claimed
to be newly executed by the fixed-fit scripts.

## Checks

```sh
Rscript tests/smoke_prediction.R
python tests/check_release.py
```

The public smoke test checks stripped-model structure, Scenario A values,
all six municipal scenarios, geometry, invalid inputs and expected warnings.
Checks requiring restricted training data are recorded separately in the
release verification; they are not falsely presented as publicly runnable
without those data. Broad 50-repetition refits are expensive and are not
silently rerun on every repository check.
