# Cluster results (ViZDoom `doom_benchmark`, APPO)

**Generated** by `scripts/agg_runs.py`; provenance: ctlab /nfs/home/kfedorin/DOOM/train_dir, 245 sf_log.txt, 2026-11-12; parsed from `/tmp/doomc`.
Do not edit by hand; re-run the script instead.

Metric: `last20` = mean of the last 20 per-iteration **Avg episode reward**
values logged by Sample Factory (one value every 5 s of training), per run.
`mean ± s.d.` is over runs with the *sample* standard deviation (n-1).
Every row names the `sf_log.txt` it was parsed from, so each number is
checkable against the raw log.

Code identity: `git_hash` is what Sample Factory recorded in `config.json`,
`diff md5` is a hash of that run's own `git.diff` after removing the engine-
generated `_vizdoom.ini` hunk (it carries a launch timestamp, so hashing the
raw diff made two identical runs differ). Both are weak: the `1fa02f7`
checkout is dirty and five `mamba3_50m*` runs carry the same diff, so what
separates those arms is the *arguments*, not the patch -- which is why `rnn`
is in the tables and the diff is never used to decide which runs are seeds.
`rnn_size` is the engine parameter, not a cell count: the SSM cores use a
flattened size (mamba3 16960 = d_model 256 / d_state 64, mamba3 66688 =
512 / 128).

## 50M env frames

| arm | per-run last20 | n | mean ± s.d. | wall h | rnn | code (git / diff md5) | source |
|---|---|---|---|---|---|---|---|
| `gdn` | 14.67 / 17.31 / 11.04 / 12.63 / 9.76 | 5 | 13.08 ± 2.99 | 1.39 | gdn/73728 | fad9545db8/74dab16a, fad9545db8/74dab16a, fad9545db8/74dab16a, fad9545db8/74dab16a, fad9545db8/74dab16a | `train_dir/gdn_50m_seed1/sf_log.txt`, `train_dir/gdn_50m_seed2/sf_log.txt`, `train_dir/gdn_50m_seed3/sf_log.txt`, `train_dir/gdn_50m_seed4/sf_log.txt`, `train_dir/gdn_50m_seed5/sf_log.txt` |
| `gru_hpov4` | 15.10 / 14.12 / 15.31 | 3 | 14.84 ± 0.64 | 0.83 | gru/512 | 1fa02f7527/d791fabb, 1fa02f7527/d791fabb, 1fa02f7527/d791fabb | `train_dir/gru_hpov4_50m_seed1/sf_log.txt`, `train_dir/gru_hpov4_50m_seed2/sf_log.txt`, `train_dir/gru_hpov4_50m_seed3/sf_log.txt` |
| `mamba3_large` | 13.89 / 13.48 / 16.07 | 3 | 14.48 ± 1.39 | 1.34 | mamba3/66688 | 1fa02f7527/d791fabb, 1fa02f7527/d791fabb, 1fa02f7527/d791fabb | `train_dir/m3_large_50m_seed1/sf_log.txt`, `train_dir/m3_large_50m_seed2/sf_log.txt`, `train_dir/m3_large_50m_seed3/sf_log.txt` |
| `mamba3_hpov4` | 13.30 / 13.35 / 14.31 | 3 | 13.65 ± 0.57 | 0.71 | mamba3/16960 | 1fa02f7527/d791fabb, 1fa02f7527/d791fabb, 1fa02f7527/d791fabb | `train_dir/mamba3_hpov4_50m_seed1/sf_log.txt`, `train_dir/mamba3_hpov4_50m_seed2/sf_log.txt`, `train_dir/mamba3_hpov4_50m_seed3/sf_log.txt` |

## 250M env frames

| arm | per-run last20 | n | mean ± s.d. | wall h | rnn | code (git / diff md5) | source |
|---|---|---|---|---|---|---|---|
| `gdn` | 13.70 / 16.96 / 14.29 | 3 | 14.98 ± 1.74 | 7.00 | gdn/73728 | fad9545db8/74dab16a, fad9545db8/74dab16a, fad9545db8/74dab16a | `train_dir/gdn_250m_seed1/sf_log.txt`, `train_dir/gdn_250m_seed2/sf_log.txt`, `train_dir/gdn_250m_seed3/sf_log.txt` |
| `gru_baseline` | 24.57 | 1 | 24.57 | 4.27 | gru/512 | 1fa02f7527/d791fabb | `train_dir/gru_250m_baseline/sf_log.txt` |
| `gru_v4` | 5.35* | 1 | 5.35 | 4.01 | gru/512 | 1fa02f7527/d791fabb | `train_dir/gru_250m_v4/sf_log.txt` |
| `gru_v4_lr3` | 23.87 | 1 | 23.87 | 4.27 | gru/512 | 1fa02f7527/d791fabb | `train_dir/gru_250m_v4_lr3/sf_log.txt` |
| `mamba3_best` | 14.26 | 1 | 14.26 | 6.62 | mamba3/66688 | 1fa02f7527/79d4bdd1 | `train_dir/mamba3_best_250m/sf_log.txt` |
| `mamba3_hpov4` | 20.08 / 18.58 / 17.29 | 3 | 18.65 ± 1.40 | 3.44 | mamba3/16960 | 1fa02f7527/d791fabb, 1fa02f7527/d791fabb, 1fa02f7527/d791fabb | `train_dir/mamba3_hpov4_250m/sf_log.txt`, `train_dir/mamba3_hpov4_250m_seed2/sf_log.txt`, `train_dir/mamba3_hpov4_250m_seed3/sf_log.txt` |

## Single runs on a different code state (not seed groups)

One run each, on a different `git.diff` than the seed groups above.
Averaging them into the `mamba3` rows would mix code states, so they stay
separate: the 50M numbers below are not seeds of one configuration.

| arm | per-run last20 | n | mean ± s.d. | wall h | rnn | code (git / diff md5) | source |
|---|---|---|---|---|---|---|---|
| `mamba3_final` | 17.65 | 1 | 17.65 | 1.33 | mamba3/66688 | 1fa02f7527/225a3a28 | `train_dir/mamba3_50m_final/sf_log.txt` |
| `mamba3_m` | 0.80 | 1 | 0.80 | 1.47 | mamba3/66688 | 1fa02f7527/5d460f3a | `train_dir/mamba3_50m_m/sf_log.txt` |
| `mamba3_best` | 15.54 | 1 | 15.54 | 1.35 | mamba3/66688 | 1fa02f7527/79d4bdd1 | `train_dir/mamba3_best_50m/sf_log.txt` |
| `mamba3_fixed` | 0.43 | 1 | 0.43 | 0.29 | mamba3/66688 | 1fa02f7527/5d460f3a | `train_dir/mamba3_fixed_50m/sf_log.txt` |
| `mamba3_fixed_v2` | 13.20 | 1 | 13.20 | 1.35 | mamba3/66688 | 1fa02f7527/5d460f3a | `train_dir/mamba3_fixed_v2_50m/sf_log.txt` |

## Runs killed before their budget (not averaged above)

Different step counts, so these are listed rather than combined.

| arm | per-run last20 | n | mean ± s.d. | wall h | rnn | code (git / diff md5) | source |
|---|---|---|---|---|---|---|---|
| `gru_baseline` | 6.42 / 13.60 / 9.98 | 3 | 10.00 ± 3.59 | 0.58 | gru/512 | 1fa02f7527/5d460f3a, 1fa02f7527/5d460f3a, 1fa02f7527/1ee49ccb | `train_dir/gru_baseline_50m/sf_log.txt`, `train_dir/gru_baseline_50m_m/sf_log.txt`, `train_dir/gru_baseline_50m_seed42/sf_log.txt` |
| `gru_optimized` | 9.30 / 14.31 | 2 | 11.80 ± 3.54 | 0.28 | gru/512 | 1fa02f7527/5d460f3a, 1fa02f7527/1ee49ccb | `train_dir/gru_optimized_50m/sf_log.txt`, `train_dir/gru_optimized_50m_seed42/sf_log.txt` |

## Excluded run directories

| group | dirs | why |
|---|---|---|
| `hpo` | 168 | 10.03M-frame hyper-parameter trials, not final arms |
| `smoke` | 14 | under 10M frames |
| `no_series` | 18 | crashed or restarted before logging a reward |
| `unclassified` | 8 | not in the run manifest -- investigate before citing |

`*` marks a run that collapsed (peaked mid-training, then decayed) and
must not be read as a low-scoring sample of a healthy configuration.

## Regenerate

```bash
python scripts/agg_runs.py --train-dir /path/to/train_dir \
    --csv results/runs.csv --md results/CLUSTER_RESULTS.md
```
