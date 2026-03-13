# DOOM / VizDoom RL

Reinforcement learning agents for ViZDoom: custom PPO (Mamba) and **Sample-Factory SOTA** reproduction.

## Sample-Factory SOTA (VizDoom)

[Sample-Factory](https://www.samplefactory.dev/) is a high-throughput RL library that reaches state-of-the-art on VizDoom with APPO (asynchronous PPO). Use it to reproduce paper results and train strong agents quickly.

### Setup

Sample-Factory uses its own dependency stack (older torch/gymnasium), so run it in a **separate environment**:

```bash
python -m venv .venv_sf && source .venv_sf/bin/activate
pip install -r scripts/requirements_sf.txt
```

Requires Linux or macOS (no Windows). Best on GPU + many CPU cores (up to ~100k interactions/s). From the project root, run the scripts below (they call `python -m sf_examples.vizdoom.*`).

### Train

```bash
# Quick run: doom_basic, 2M steps
./scripts/sample_factory_train.sh doom_basic 2000000

# Paper-style: 4B env steps (stop anytime with Ctrl+C, resume with same cmd)
./scripts/sample_factory_train.sh doom_defend_the_center 4000000000
```

Scale workers to your machine (default: 20 workers × 16 envs):

```bash
NUM_WORKERS=8 NUM_ENVS_PER_WORKER=10 ./scripts/sample_factory_train.sh doom_basic 10000000
```

### Enjoy / Evaluate

```bash
./scripts/sample_factory_enjoy.sh doom_basic DoomSOTA_doom_basic
```

### Env names (Sample-Factory ↔ Gymnasium)

| Sample-Factory              | Gymnasium (this repo)   |
|----------------------------|--------------------------|
| `doom_basic`               | VizdoomBasic-v1         |
| `doom_deadly_corridor`     | VizdoomCorridor-v1      |
| `doom_defend_the_center`   | VizdoomDefendCenter-v1  |
| `doom_defend_the_line`     | VizdoomDefendLine-v1    |
| `doom_health_gathering`    | VizdoomHealthGathering-v1 |
| `doom_health_gathering_supreme` | —                   |
| `doom_battle` / `doom_battle2` | — (paper SOTA)     |
| `doom_duel_bots` / `doom_deathmatch_bots` | —              |

### Monitoring

```bash
tensorboard --logdir=./train_dir
```

### References

- [Sample-Factory VizDoom docs](https://www.samplefactory.dev/09-environment-integrations/vizdoom/)
- [SF2 VizDoom Battle report (W&B)](https://wandb.ai/andrewzhang505/sample_factory/reports/VizDoom-Battle-Environments--VmlldzoyMzcyODQx)
- HuggingFace models: [doom_battle](https://huggingface.co/andrewzhang505/sample-factory-2-doom-battle), [doom_battle2](https://huggingface.co/andrewzhang505/sample-factory-2-doom-battle2)

---

## Custom PPO (Mamba)

- **Train:** `ppo_train.py` (or `ppo_train.ipynb`) — PPO with SSDMamba2Combatant on Vizdoom (e.g. Corridor).
- **Eval:** `eval.py --model <path> --env <env_id>` — run checkpoint and plot reward histogram.
