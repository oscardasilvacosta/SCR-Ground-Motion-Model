
# Data

`sites/` contains the retained Lisbon prediction grid: grid_id, WGS84 latitude,
WGS84 longitude and Vs30 in m/s. The main CSV/Excel contains 1,375 municipal
centres. The separately labelled original grid has 2,652 sites, including those
outside Lisbon. Nominal 250 m spacing is represented by original angular
increments, not resampled to a new projected lattice. Site values are unchanged.

The ESM-derived training dataset is **not included**. Obtain the authors'
processed workbook or canonical retained table with the required permissions.
The upstream database is https://doi.org/10.13127/esm/flatfile.1.0, but that
download alone does not recover all integrated metadata and selection history.

Keep source and prepared training records outside this repository or in the
ignored `data/private/` directory. Do not commit them or serialize unsanitized
caret training objects into public folders. The static site grid alone cannot
predict PGA; append coherent earthquake and derived geometry inputs or use
`R/05_predict_scenario_grid.R`.
