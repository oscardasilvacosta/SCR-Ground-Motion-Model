# Changelog

## XGB2.9G - 2026-11-02

- Replaced the XGB2.2G main-branch workflow with the adopted nine-predictor XGB2.9G configuration.
- Separated Vs30 and epicentral distance; removed direct station longitude and the ratio predictor.
- Unified the angular convention and corrected the explicit PGA output unit to percent g.
- Added actual data preparation, fixed-setting fitting, prediction, Monte Carlo, permutation, mixed-validation, bootstrap and SHAP code.
- Added final and evaluation inference-only models; embedded training records and callbacks removed and predictions verified unchanged.
- Added the municipal Vs30 grid, original broader grid, complete Scenario A example and scenario definitions.
- Added current aggregate results, selected figures and explicit evaluation/uncertainty limitations.
- Kept restricted ESM records out of the public tree and clarified conditional retraining reproducibility.
- Retained MIT for source code; distinguished third-party data rights.
- Clarified that the old XGB2.2G Zenodo record was not updated by this change.
- Simplified reader-facing performance tables to Pearson R2 and prepared an XGB2.9G archive README for the upcoming Zenodo upload.
