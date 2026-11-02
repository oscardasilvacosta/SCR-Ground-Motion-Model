# XGB2.9G Models

- `XGB2.9G_FULL_DB.rds`: final deployment model fitted to 9,579 records.
- `XGB2.9G_EVALUATION.rds`: evaluation model fitted to 8,236 training records.
- `*_preprocessing.csv`: fitted feature centres and scales, not raw observations.
- `inference_export_verification.json`: export identity and prediction checks.

Use the FULL_DB model for scenario predictions and the EVALUATION model for the
reserved SCR validation data. Both are inference-only caret objects; training
records and callbacks were removed without changing predictions. Always supply
`newdata` to `predict`. Training summaries, resamples and `update(model)` are not
supported by these stripped objects; use the actual fitting scripts instead.

The caret object handles preprocessing. Never scale twice. Predictions are
RotD100 PGA already in percent g, not g. The reference xgboost version is
1.7.11.1. Newer xgboost major versions are not claimed to be compatible without
verification. The associated Zenodo deposit is at
https://zenodo.org/records/18756043; its XGB2.9G upload is being prepared
separately. The upload README is in `zenodo/README.txt`. Check the model version
and file name rather than assuming an earlier deposit contains the revised fit.
