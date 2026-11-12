---
license: mit
tags:
  - reinforcement-learning
  - vizdoom
  - sample-factory
  - appo
  - gru
library_name: sample-factory
---

# doom_benchmark, APPO + GRU, 250M frames

Recurrent baseline for ViZDoom's deathmatch arena, trained with Sample Factory 2.1.1
(APPO). This is the best-scoring run of a 245-run architecture comparison; the
evidence for every number lives in the GitHub repo, not here.

## Results

| metric | value | where |
|---|---|---|
| `last20` (mean of last 20 per-iteration avg episode reward) | **24.57** | `train_dir/gru_250m_baseline/sf_log.txt` |
| best checkpoint reward (Sample Factory `save_best_metric`) | **24.48** | `checkpoint_p0/best_000015260_250019840_reward_24.480.pth` |
| final mean episode reward | 23.96 | same log |
| wall clock | 4.3 h | log timestamps |
| throughput | 16.2k FPS | `Collected {0: 250019840}, FPS: 16235.8` |

Single seed. A second run of the same family at a different LR reached 23.87; a third
collapsed to 5.35 (peak 16.54) — reported as an LR instability, not averaged in.

## Trained configuration

```
env=doom_benchmark (ViZDoom battle.cfg, Discrete(9))
algo=APPO  use_rnn=True  rnn_type=gru  rnn_size=512  rnn_num_layers=1
env_frameskip=4  res=128x72 (from 160x120)  normalize_input=True
num_workers=8  num_envs_per_worker=8  batch_size=4096  rollout=64
recurrence=32  gamma=0.99  ppo_clip_ratio=0.1  num_batches_to_accumulate=2
train_for_env_steps=250,000,000  device=gpu
```

Exact values: `gru_250m_baseline/config.json`.

## Evaluate

```bash
huggingface-cli download Kiraprint/doom-sf-gru-250m --local-dir ./train_dir
EXPERIMENT=gru_250m_baseline SAVE_VIDEO=1 NO_RENDER=1 \
  ./scripts/eval_sf_checkpoint.sh
```

Requires Sample Factory 2.1.1 + ViZDoom 1.3.0 (`pip install 'sample-factory[vizdoom]'`)
on Linux, and `ffmpeg` on PATH if you record video. The script pins
`TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`: SF 2.1.1 calls `torch.load()` without
`weights_only`, and torch >= 2.6 then refuses this checkpoint and reports it as
`'NoneType' object is not subscriptable`.

Recorded episodes from this checkpoint scored 30.0, 27.0 and 20.0 — those are
single episodes on the training distribution, not a held-out evaluation.

## Limitations

One arena (`doom_bankmark` = deathmatch score, no level completion), one action
space, one seed, no held-out eval split. Do not read this as navigation or
level-solving competence.
