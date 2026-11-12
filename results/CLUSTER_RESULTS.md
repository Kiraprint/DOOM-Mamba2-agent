# Cluster results (ViZDoom `doom_benchmark`, APPO)

**Generated** by `scripts/agg_runs.py` on 2026-11-12 from the 242 `sf_log.txt` files in
the 245 run directories under `/tmp/doomc`; 3 of those directories hold a `config.json`
but no log and are listed as `no_log` below.
Do not edit by hand; re-run the script instead.

Metric: `last20` = mean of the last 20 per-iteration **Avg episode reward**
values logged by Sample Factory (one value every 5 s of training), per run.
`mean ± s.d.` is over runs with the *sample* standard deviation (n-1),
computed from the unrounded 20-value means; a row with n=1 is a single run
and prints no s.d. Every row names the `sf_log.txt` it was parsed from, so
each number is checkable against the raw log, and `results/runs.csv` carries
the frame counts behind each bucket.

Bucketing comes from the logs, never from directory names: `frames reached`
is the largest `Collected {0: N}` line in the log (or its largest `Total num
frames: N` when that line is missing) -- never the last one, because a log
that holds a relaunch can end with a smaller count -- and its budget is
`train_for_env_steps` from its own `config.json`. A run counts as finished
at 99% of that budget. `seed` in `results/runs.csv` is the `seed` field of
`config.json` and is blank where the run did not record one -- the runs are
replicates of an arm, not a numbered seed series.

Code identity: `git_hash` is what Sample Factory recorded in `config.json`,
`diff md5` is a hash of that run's own `git.diff` after removing the engine-
generated `_vizdoom.ini` hunk (it carries a launch timestamp, so hashing the
raw diff made two identical runs differ). Both are weak: most runs come from
the same dirty `1fa02f7` checkout, and runs can share a diff signature while
differing in arguments. That is why which runs are replicates of an arm comes
from `MANIFEST` in the script, and why the script fails if a run that reached
its budget is missing from it.
`rnn_size` is the engine parameter, not a cell count: the SSM cores use a
flattened size (mamba3 16960 = d_model 256 / d_state 64, mamba3 66688 =
512 / 128).

Where the code lives: the campaign ran in the `DOOM` repository from a dirty
`1fa02f7` checkout (and `fad9545` for the GDN runs, pinned here as branch
`cluster/gdn`). The checkout was edited between batches, so there is no single
working-tree patch: every distinct state is published as
`results/code_states/<signature>.diff`, and the table below says which runs
came from it. The `DOOM` repository alone cannot reproduce these numbers --
the signatures differ from its `master`.

## 50M env frames

| arm | per-run last20 | n | mean ± s.d. | wall h | rnn | code (git / diff md5) | source |
|---|---|---|---|---|---|---|---|
| `gdn` | 14.67 / 17.31 / 11.04 / 12.63 / 9.76 | 5 | 13.08 ± 2.99 | 1.41 | gdn/73728 | fad9545db8/74dab16a, fad9545db8/74dab16a, fad9545db8/74dab16a, fad9545db8/74dab16a, fad9545db8/74dab16a | `train_dir/gdn_50m_seed1/sf_log.txt`, `train_dir/gdn_50m_seed2/sf_log.txt`, `train_dir/gdn_50m_seed3/sf_log.txt`, `train_dir/gdn_50m_seed4/sf_log.txt`, `train_dir/gdn_50m_seed5/sf_log.txt` |
| `gru_baseline` | 13.60 | 1 | 13.60 | 1.13 | gru/512 | 1fa02f7527/5d460f3a | `train_dir/gru_baseline_50m_m/sf_log.txt` |
| `gru_hpov4` | 15.10 / 14.12 / 15.31 | 3 | 14.84 ± 0.63 | 0.84 | gru/512 | 1fa02f7527/d791fabb, 1fa02f7527/d791fabb, 1fa02f7527/d791fabb | `train_dir/gru_hpov4_50m_seed1/sf_log.txt`, `train_dir/gru_hpov4_50m_seed2/sf_log.txt`, `train_dir/gru_hpov4_50m_seed3/sf_log.txt` |
| `mamba3_large` | 13.89 / 13.48 / 16.07 | 3 | 14.48 ± 1.39 | 1.35 | mamba3/66688 | 1fa02f7527/d791fabb, 1fa02f7527/d791fabb, 1fa02f7527/d791fabb | `train_dir/m3_large_50m_seed1/sf_log.txt`, `train_dir/m3_large_50m_seed2/sf_log.txt`, `train_dir/m3_large_50m_seed3/sf_log.txt` |
| `mamba3_hpov4` | 13.30 / 13.35 / 14.31 | 3 | 13.66 ± 0.57 | 0.72 | mamba3/16960 | 1fa02f7527/d791fabb, 1fa02f7527/d791fabb, 1fa02f7527/d791fabb | `train_dir/mamba3_hpov4_50m_seed1/sf_log.txt`, `train_dir/mamba3_hpov4_50m_seed2/sf_log.txt`, `train_dir/mamba3_hpov4_50m_seed3/sf_log.txt` |

## 250M env frames

| arm | per-run last20 | n | mean ± s.d. | wall h | rnn | code (git / diff md5) | source |
|---|---|---|---|---|---|---|---|
| `gdn` | 13.70 / 16.96 / 14.29 | 3 | 14.98 ± 1.73 | 7.01 | gdn/73728 | fad9545db8/74dab16a, fad9545db8/74dab16a, fad9545db8/74dab16a | `train_dir/gdn_250m_seed1/sf_log.txt`, `train_dir/gdn_250m_seed2/sf_log.txt`, `train_dir/gdn_250m_seed3/sf_log.txt` |
| `gru_baseline` | 24.57 | 1 | 24.57 | 4.28 | gru/512 | 1fa02f7527/d791fabb | `train_dir/gru_250m_baseline/sf_log.txt` |
| `gru_v4` | 5.35* | 1 | 5.35 | 4.02 | gru/512 | 1fa02f7527/d791fabb | `train_dir/gru_250m_v4/sf_log.txt` |
| `gru_v4_lr3` | 23.87 | 1 | 23.87 | 4.28 | gru/512 | 1fa02f7527/d791fabb | `train_dir/gru_250m_v4_lr3/sf_log.txt` |
| `mamba3_best` | 14.26 | 1 | 14.26 | 6.63 | mamba3/66688 | 1fa02f7527/79d4bdd1 | `train_dir/mamba3_best_250m/sf_log.txt` |
| `mamba3_hpov4` | 20.08 / 18.58 / 17.29 | 3 | 18.65 ± 1.40 | 3.45 | mamba3/16960 | 1fa02f7527/d791fabb, 1fa02f7527/d791fabb, 1fa02f7527/d791fabb | `train_dir/mamba3_hpov4_250m/sf_log.txt`, `train_dir/mamba3_hpov4_250m_seed2/sf_log.txt`, `train_dir/mamba3_hpov4_250m_seed3/sf_log.txt` |

## Single runs (not replicates of any arm above)

Each row is one run. None of them is a seed of a 50M arm, and averaging
them into the `mamba3` rows would mix code states: their `git.diff`
signature is never the `d791fabb` of the 50M seed groups. The diff alone
does not make each of them a distinct code state either -- some of these
runs share their signature with each other and differ only in arguments
(all of them share the same `rnn`, `batch_size` and `rollout`), so each is
listed on its own rather than treated as a configuration in its own right.

| arm | per-run last20 | n | mean ± s.d. | wall h | rnn | code (git / diff md5) | source |
|---|---|---|---|---|---|---|---|
| `mamba3_final` | 17.65 | 1 | 17.65 | 1.34 | mamba3/66688 | 1fa02f7527/225a3a28 | `train_dir/mamba3_50m_final/sf_log.txt` |
| `mamba3_m` | 0.80* | 1 | 0.80 | 1.48 | mamba3/66688 | 1fa02f7527/5d460f3a | `train_dir/mamba3_50m_m/sf_log.txt` |
| `mamba3_best` | 15.54 | 1 | 15.54 | 1.36 | mamba3/66688 | 1fa02f7527/79d4bdd1 | `train_dir/mamba3_best_50m/sf_log.txt` |
| `mamba3_fixed_v2` | 13.20 | 1 | 13.20 | 1.36 | mamba3/66688 | 1fa02f7527/5d460f3a | `train_dir/mamba3_fixed_v2_50m/sf_log.txt` |

## Runs killed before their budget

None of these reached its own `train_for_env_steps` budget, and they have
different step counts, so no mean is printed: each row is a list.
A run that did reach its budget here would fail the guard and have to be
moved into a headline arm.

| arm | per-run last20 | frames reached | n | wall h | rnn | code (git / diff md5) | source |
|---|---|---|---|---|---|---|---|
| `gru_baseline` | 6.42 / 9.98 | 13.43M / 25.18M | 2 | 0.32 | gru/512 | 1fa02f7527/5d460f3a, 1fa02f7527/1ee49ccb | `train_dir/gru_baseline_50m/sf_log.txt`, `train_dir/gru_baseline_50m_seed42/sf_log.txt` |
| `gru_optimized` | 9.30 / 14.31 | 8.21M / 35.90M | 2 | 0.28 | gru/512 | 1fa02f7527/5d460f3a, 1fa02f7527/1ee49ccb | `train_dir/gru_optimized_50m/sf_log.txt`, `train_dir/gru_optimized_50m_seed42/sf_log.txt` |

## Code states (one patch per distinct working tree)

Derived from each run's own `git.diff`, with the engine-generated
`_vizdoom.ini` hunk removed so a launch timestamp cannot split one state
into two. Each patch is committed as `results/code_states/<signature>.diff`.
A signature shared by many runs means the same code with different
arguments -- never that they are replicates of one configuration.

| signature | runs | git_hash | example run | patch |
|---|---|---|---|---|
| `d791fabb` | 65 | 1fa02f7527 | `gru_250m_baseline` | `results/code_states/d791fabb.diff` |
| `4a4b295e` | 53 | 1fa02f7527 | `hpo_mamba2_0_20260805_125544` | `results/code_states/4a4b295e.diff` |
| `ebcf066c` | 27 | 1fa02f7527 | `hpo_0_20260727_215734` | `results/code_states/ebcf066c.diff` |
| `80343fa0` | 18 | 1fa02f7527 | `hpo_0_20260728_234741` | `results/code_states/80343fa0.diff` |
| `79d4bdd1` | 17 | 1fa02f7527 | `hpo_0_20260726_025755` | `results/code_states/79d4bdd1.diff` |
| `5d460f3a` | 15 | 1fa02f7527 | `gru_baseline_50m` | `results/code_states/5d460f3a.diff` |
| `225a3a28` | 12 | 1fa02f7527 | `m3_fixed_test` | `results/code_states/225a3a28.diff` |
| `74dab16a` | 9 | fad9545db8 | `gdn_250m_seed1` | `results/code_states/74dab16a.diff` |
| `1ee49ccb` | 9 | 1fa02f7527 | `gru_100k` | `results/code_states/1ee49ccb.diff` |
| `d41d8cd9` | 6 | 1fa02f7527 | `gru_q100k` | `results/code_states/d41d8cd9.diff` |
| `86d28965` | 2 | fad9545db8 | `gdn2_smoke_100k` | `results/code_states/86d28965.diff` |
| `4583f292` | 2 | 1fa02f7527 | `hpo_0_20260726_024255` | `results/code_states/4583f292.diff` |
| `395fd704` | 1 | 1fa02f7527 | `hpo_0_20260726_023554` | `results/code_states/395fd704.diff` |
| `8e88def6` | 1 | 1fa02f7527 | `hpo_0_20260726_025225` | `results/code_states/8e88def6.diff` |
| `5478083d` | 1 | 1fa02f7527 | `hpo_0_20260728_234454` | `results/code_states/5478083d.diff` |
| `715f5f31` | 1 | 1fa02f7527 | `trial_0_20260725_180456` | `results/code_states/715f5f31.diff` |
| `3cebef49` | 1 | 1fa02f7527 | `trial_0_20260725_213525` | `results/code_states/3cebef49.diff` |
| `f748ac04` | 1 | 1fa02f7527 | `trial_0_20260726_022927` | `results/code_states/f748ac04.diff` |
| `af5c6de7` | 1 | 1fa02f7527 | `trial_0_20260726_023257` | `results/code_states/af5c6de7.diff` |

## Restarted runs (several training sessions in one log)

The HPO driver appended each relaunch to the first trial's log, and every
relaunch wrote its config into a *different* `train_dir` -- so one file here
holds several separate runs and no `last20` over it means anything. Sessions
are counted from the `Saving configuration to ...` line each launch prints,
not from a backwards step in `Collected`: a relaunch that repeats the
previous count leaves no trace in that series. Frames reached is the largest
`Collected` value, and the wall clock is the **sum of the per-session spans**
-- a first-to-last subtraction crosses the gaps between relaunches and comes
out negative here.

| run | sessions | frames reached | budget | wall h | code | source |
|---|---|---|---|---|---|---|
| `gdn2_smoke_100k` | 2 |  | 50000000 | 0.46 | fad9545db8/86d28965 | `train_dir/gdn2_smoke_100k/sf_log.txt` |
| `kda_smoke_100k` | 2 |  | 50000000 | 0.45 | fad9545db8/86d28965 | `train_dir/kda_smoke_100k/sf_log.txt` |
| `m3_trial_0_20260731_021223` | 3 |  | 50000000 | 1.00 | 1fa02f7527/225a3a28 | `train_dir/m3_trial_0_20260731_021223/sf_log.txt` |
| `m3_trial_1_20260731_223057` | 3 | 50020352 | 50000000 | 4.40 | 1fa02f7527/225a3a28 | `train_dir/m3_trial_1_20260731_223057/sf_log.txt` |
| `m3_trial_4_20260804_155155` | 4 | 50020352 | 50000000 | 3.99 | 1fa02f7527/225a3a28 | `train_dir/m3_trial_4_20260804_155155/sf_log.txt` |

## Excluded run directories

| group | dirs | why |
|---|---|---|
| `quarantined` | 1 | never trained; each one is named below |
| `hpo` | 168 | hyper-parameter trials at 10.03M frames (50 mamba3, 50 gru, 3 mamba2, 65 legacy `hpo_<N>_` trials), not final arms |
| `smoke` | 20 | under 10M frames |
| `no_series` | 15 | crashed before logging a reward |
| `restarted` | 5 | several training sessions in one log; published separately above, no reward statistic |
| `no_log` | 3 | directory has a `config.json` but no `sf_log.txt`: `m3_trial_2_20260731_031230`, `m3_trial_3_20260801_025455`, `m3_trial_7_20260804_195138` |
| `unclassified` | 0 | empty by construction: the script exits nonzero if a run lands here |

Quarantined, not tabulated:
- `mamba3_fixed_50m` -- collected 0 env frames (`Total num frames: 0`, `Fps is ... 0.0`) and logged the same 0.43 reward 209 times: the worker never stepped the env, so this is not a 50M result (`train_dir/mamba3_fixed_50m/sf_log.txt`).

`*` marks a run that collapsed (peaked mid-training, then decayed) or is
otherwise degenerate, and must not be read as a low-scoring sample of a
healthy configuration.

## Regenerate

```bash
python scripts/agg_runs.py --train-dir /tmp/doomc \
    --csv results/runs.csv --md results/CLUSTER_RESULTS.md \
    --date 2026-11-12
```
