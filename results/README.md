# Published Results

This folder contains selected research outputs, not the ESM training table or
per-record ESM validation observations. The original restricted inputs are kept
separate. `asset_provenance.json` records copied asset sources and checksums;
`release_verification.json` records checks actually performed for this update.

## Aggregate Tables

- `SCR_validation_metrics.csv`: primary recording-level development-validation results.
- `SCR_and_mixed_bootstrap_intervals.csv`: 1,000 conditional bootstrap replicates per scheme; both recording and event-cluster schemes are labelled.
- `paired_MC50_summary.csv` and `paired_MC50_repetition_metrics.csv`: strict and retained-non-SCR protocols; the percentile columns describe across-split variability, not confidence intervals.
- `paired_MC50_event_overlap.csv`: overlap audit for the retained protocol.
- `permutation_20_metrics.csv`: 20 shuffled-training-target diagnostic fits; not a proof against leakage.
- `predictor_configuration_comparison.csv`: reused-validation comparisons, including the selected configuration; it is not the best configuration on every error metric.
- `IN_SAMPLE_full_DB_metrics.csv`: explicitly in-sample, not validation.
- `scenario_spatial_summaries.csv` and `scenario_XGB_to_GMPE_ratios.csv`: full-grid and municipal-centre scopes are separate. Median sitewise ratios differ from ratios of spatial means.
- `local_contrast_summary.csv`: broad-grid and municipal-only pairs are explicitly distinguished.
- `real_recordings_station_predictions.csv`: ten previously inspected station cases, their supplied observed PGA and separate model estimates.
- `real_recordings_available_and_common_subset_metrics.csv`: all-available and common-nine station errors, never silently interchanged.

Primary SCR event-cluster 95% bootstrap intervals are R2_SSE 0.6717-0.8804,
MAE 0.8695-1.3416 %g, RMSE 2.5430-4.1453 %g. These are conditional on the fixed
model and reused validation recordings, not full model-development uncertainty.

## Scenarios

`scenarios/` contains 1,375 municipal-centre predictions per scenario A, C, E,
F, G and H, with input coordinates, site values and retained GMPE levels.
The extra finite-rupture GMPE distance columns are not interchangeable with
XGB epicentral/hypocentral distances. No OpenQuake recalculation occurs here.

P50 uses marginal/model-mixture medians; CENT is the legacy weighted average
of branch medians. P16/P84 are marginal distribution quantiles, not XGB
prediction intervals. Model component definitions are not harmonized by a new
conversion. See `documentation/METHODS.md`.

## Figures

The selected figures cover model selection, SCR validation, holdout/full-data
SHAP, Lisbon scenario maps, GMPE comparisons, local contrasts, the combined
labelled P16/P50/P84 boxplots, real-station errors and support-aware empirical
magnitude/distance trends. Figure filenames and captions identify the scopes;
legacy `surface` ratio images use their retained municipal-centre definitions.
Numeric tables provide the authoritative full-grid/municipal distinction.

The support-aware empirical markers are bin means from the 1,343 SCR validation
records, not single stations and not constant-Vs30/constant-coordinate curves.
They do not eliminate the non-monotonic conditional responses of the model.
SHAP distinguishes explanations of the fitted full-data model from explanations
of evaluation-model predictions on SCR holdout records. Neither is causal.

For ten real-station cases, XGB2.9G has MAE 3.55698 and RMSE 5.306995 %g. On
the common-nine cases it has MAE 3.92598 and RMSE 5.593511 %g. Five-GMPE and
Kotha lack IV.MGR; no estimate is invented. All-available bars are therefore not
a fully matched ranking. The public plotting helper can reproduce that plot:

```sh
python python/plot_results.py outputs/new_station_figure
```

This update verifies current data, code and selected images. It does not claim
to reexecute every original GIS workflow, OpenQuake configuration, historical
grid-search trial or every possible manuscript layout from scratch.
