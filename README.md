XGBoost Ground Motion Model for RotD100 Peak Ground Acceleration in Stable Continental Regions

This repository contains the complete machine learning framework used to develop an XGBoost-based ground motion model for the prediction of RotD100 peak ground acceleration (PGA) in Stable Continental Regions (SCRs).

The model integrates ground motion records from the European Strong Motion (ESM) database (https://esm-db.eu/#/products/flat_file) with additional publicly available site and tectonic metadata. The workflow includes feature engineering, cross-validation, model selection, full-database training, and SHAP-based interpretability analysis.

Repository Structure

R/

02_train_validate.R – Model development, cross-validation, and SCR hold-out evaluation

03_train_final.R – Final model training on the full processed database

04_predict_PGA.R – Script for generating PGA predictions using the trained model

data/

Placeholder directory for user-supplied input data (not included)

models/

Trained model is archived externally (see below)

outputs/

Directory for generated prediction outputs

Data Availability

Ground motion records were obtained from the European Strong Motion (ESM) database (https://esm-db.eu/#/products/flat_file). Additional open-source site and tectonic parameters were integrated during preprocessing.

Due to redistribution restrictions of the source databases, the compiled training dataset is not included in this repository.

Users must retrieve the original ESM data and associated open-source variables independently, then execute the preprocessing and training scripts provided here to reproduce the model.

Trained Model Archive

The final trained XGBoost model (full-database training) is archived on Zenodo:

https://doi.org/10.5281/zenodo.18756043

Archive contents:

XGB2.2G_FINAL_fullDB.rds

This model was generated using:
R/03_train_final.R

Reproducibility Workflow

Retrieve ground motion records from the ESM database.

Integrate additional publicly available metadata as described in the manuscript.

Run:

R/02_train_validate.R for validation and model selection.

R/03_train_final.R for final training.

Use R/04_predict_PGA.R to generate PGA predictions for new input data.

The prediction script reconstructs engineered features (e.g., Vs30/Re) internally to ensure consistency with the training pipeline.

License

This repository and the archived trained model are distributed under the MIT License.
