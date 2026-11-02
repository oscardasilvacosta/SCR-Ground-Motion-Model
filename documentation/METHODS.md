# Methods and Limitations

## Units and Geometry

The raw response is RotD100 PGA in **percent g**. The historical training column
name ends in `_g`; this is not a unit conversion instruction. No logarithmic
response transform or positivity clipping is applied.

For event latitude a, station latitude b and longitude difference dl, all in radians:

```text
east  = sin(dl) * cos(b)
north = cos(a)*sin(b) - sin(a)*cos(b)*cos(dl)
azimuth_cos = cos(atan2(north, east))
```

This is counterclockwise from east, equivalent to the sine of the usual
clockwise-from-north bearing. One cosine does not uniquely encode a full-circle
direction and is not evidence of rupture directivity.

Historical training Re/Rh values are preserved for exact reproduction. For new
sites, missing Re is calculated with the haversine expression and spherical
radius 6371.0088 km. Missing Rh is sqrt(Re^2 + depth^2). Supplied distances are
not silently overwritten; they must be independently checked for the intended
source definition. Never change event coordinates/depth and retain stale
event-dependent columns from a different scenario.

## Evaluation Design

Of the 4,479 SCR records, 3,136 enter training and 1,343 development validation.
All 5,100 auxiliary records labelled `UNK` enter training under the authors'
non-SCR classification. The original label remains available to avoid claiming
that the string `UNK` itself independently verifies a tectonic class.

The primary split is recording-level and PGA-stratified, not event-disjoint.
It contains 305 validation earthquakes; 280 also contribute training records.
It informed configuration selection and must not be presented as a fresh,
untouched or whole-earthquake-independent test. There is no additional fixed
internal test partition in this adopted strategy. Mixed recording validation is
a separate 70/30 partition of the combined dataset, not the primary SCR split.

The paired 50-repetition analysis evaluates the same SCR test recordings under
strict earthquake exclusion and non-SCR-retained training. Better retained-pool
scores cannot be attributed solely to training-set size: access to other
recordings from shared test earthquakes also changes.

## Model Support

The retained dataset spans Mw 4.5-7.8, approximately Re 0.5-561 km,
Rh 4-685 km, depth 0.1-620 km and Vs30 90-2925 m/s. Its 95th-percentile values
are Mw 6.5, Re 321.64 km, Rh 329.04 km and Vs30 900 m/s. Ties mean that more
than 95% can lie at or below a listed quantile. Marginal ranges do not establish
joint multivariate support or a validated applicability domain.

The indicated Mw 7.3 threshold in support-aware figures is exploratory, not a
validated model limit or fitted change point. Mw 7.8 is the observed training
maximum. The model has non-monotonic conditional magnitude/distance responses,
including reversals inside marginal data ranges; empirical binned agreement
does not remove those limitations. Thirty-three primary SCR validation
predictions are nonpositive and remain included in error metrics.

The training pool includes one mainland Portuguese-station recording, IP.LIS,
from the 1969 event (Mw 7.8 in the adopted data), in training rather than SCR
validation. It contains no Azores-station recordings. The SCR recording split
therefore does not directly establish Portuguese-site generalization.

## Scenario and GMPE Comparisons

All configurations use the same retained site coordinates and Vs30 values.
The original grid has 2,652 points; 1,375 centres are inside the adopted Lisbon
municipal boundary. The nominal spacing is 250 m, implemented with angular
increments of 0.00288 degrees longitude and 0.00225 degrees latitude, not a newly
constructed exactly equidistant projected grid. Municipal inclusion is a
centre-point rule, not exact area weighting of clipped edge-cell footprints.
Boundary agency/vintage were not independently verified.

Current comparison tables retain the precomputed OpenQuake GMPE fields. This
GitHub update does not rerun OpenQuake or invent new source/logic-tree settings.
P50 is the distribution or model-mixture median; CENT is the legacy arithmetic
weighted average of branch medians. They differ for weighted combinations.
P16/P84 are marginal 15.87/84.13% distribution levels, not confidence bounds on
a mean. XGB2.9G has one point prediction, not newly fitted P16/P84 outputs.
RotD100, Kotha RotD50 and legacy horizontal-component definitions are not
silently converted, so agreement is not an accuracy ranking.

Local contrasts use 100*(max(Pi,Pj)/min(Pi,Pj)-1). Neighbourhood maps can retain
outside-municipality neighbours before display clipping; municipal-only pair
summaries require both endpoints inside. Neither smooth maps nor high spatial
cohesion prove prediction accuracy.

## Real Recordings and Empirical Scaling

The ten station cases correspond to nine source solutions. Saved station
predictions average nine individual cell predictions around each station, not
one prediction at the average inputs. Their observed horizontal component was
retained as supplied and is not independently verified RotD100. These already
inspected article cases are not a new blind external test.

IV.MGR lacks Five-GMPE and Kotha estimates; no value is imputed. All-available
errors use n=10 for XGB, AB06 and ABMIX, and n=9 for Five-GMPE/Kotha. Common-nine
errors are retained separately for matched comparisons.

The support-aware magnitude/distance figure aggregates the 1,343 SCR validation
recordings into bins. Markers are arithmetic bin means, not individual records.
Other site/source inputs vary as observed. Bins with fewer than five records or
two events are hollow and unconnected. Observed-mean bars use 2,000 fixed-model
event-cluster bootstrap repetitions; single-event bins have no event-bootstrap
interval. These are empirical multivariate-composition trends, not isolated
fixed-site magnitude scaling or attenuation curves. Increasing observed and
predicted means with distance is not evidence of inverse physical attenuation.
