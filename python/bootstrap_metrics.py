"""Conditional fixed-model bootstrap, in the original recording-then-cluster RNG order."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def metrics(y, p):
    e = p - y
    denominator = np.sum((y - y.mean()) ** 2)
    return {
        'R2_SSE': float(1 - np.sum(e**2) / denominator) if denominator > 0 else np.nan,
        'R2_Pearson_squared': float(np.corrcoef(y, p)[0, 1]**2) if np.std(y)>0 and np.std(p)>0 else np.nan,
        'MAE_pctg': float(np.abs(e).mean()),
        'RMSE_pctg': float(np.sqrt(np.mean(e**2))),
    }


def bootstrap(frame, groups, replicates=1000, seed=20261020):
    y = frame.observed_pctg.to_numpy(float)
    p = frame.predicted_pctg.to_numpy(float)
    if not len(y) or not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError('Finite observed/predicted PGA values are required.')
    indices = [np.flatnonzero(groups == key) for key in pd.unique(groups)]
    rng = np.random.default_rng(seed)
    rows = []
    for scope in ['recording_bootstrap', 'event_cluster_bootstrap']:
        for rep in range(replicates):
            sample = (rng.integers(0, len(y), len(y)) if scope == 'recording_bootstrap' else
                      np.concatenate([indices[i] for i in rng.integers(0, len(indices), len(indices))]))
            rows.append({'scope':scope, 'repetition':rep+1, 'n_resampled':len(sample),
                         **metrics(y[sample], p[sample])})
    draws = pd.DataFrame(rows)
    summary = []
    for scope, subset in draws.groupby('scope'):
        for metric, point in metrics(y, p).items():
            summary.append({'scope':scope, 'metric':metric, 'point_estimate':point,
                            'percentile_95_low':subset[metric].quantile(.025),
                            'percentile_95_high':subset[metric].quantile(.975),
                            'n_replicates':replicates})
    return draws, pd.DataFrame(summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('private_predictions', type=Path)
    parser.add_argument('new_output_directory', type=Path)
    parser.add_argument('--replicates', type=int, default=1000)
    args = parser.parse_args()
    if args.replicates < 1:
        parser.error('replicates must be positive')
    if args.new_output_directory.exists():
        parser.error('Choose a NEW output directory')
    frame = pd.read_csv(args.private_predictions)
    if frame.record_id.duplicated().any():
        raise ValueError('Predictions must have unique record_id values.')
    partition = pd.read_csv(Path(__file__).resolve().parents[1] / 'config/record_partition.csv').set_index('record_id')
    groups = partition.loc[frame.record_id, 'bootstrap_event_group'].to_numpy()
    draws, summary = bootstrap(frame, groups, args.replicates)
    args.new_output_directory.mkdir(parents=True)
    draws.to_csv(args.new_output_directory / 'bootstrap_replicates.csv', index=False)
    summary.to_csv(args.new_output_directory / 'bootstrap_intervals.csv', index=False)
    print(summary.to_string(index=False))


if __name__ == '__main__':
    main()
