#!/usr/bin/env python3
"""Rebuild results/runs.csv and results/CLUSTER_RESULTS.md from Sample-Factory logs.

One per-run artifact survived this study: train_dir/<run>/sf_log.txt
(results_summary.json was gitignored upstream and never committed). Every number
in results/CLUSTER_RESULTS.md is therefore derived here from those logs.

    python scripts/agg_runs.py --train-dir <train_dir> \
        --csv results/runs.csv --md results/CLUSTER_RESULTS.md

Exits nonzero if any run reaching 50M env frames is left unclassified, so a
renamed run directory can never silently drop a seed from the tables -- which is
exactly what happened when a regex assumed a `_seed1` suffix: mamba3_hpov4_250m
is seed 1 of a 3-seed arm.
"""

import argparse
import csv
import glob
import os
import re
import statistics
import sys
from datetime import datetime

MIN_REPORT_STEPS = 50_000_000

RE_AVG = re.compile(
    r"\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d),\d+\].*Avg episode reward: \[\(0, '(-?\d+\.?\d*)'\)\]"
)
RE_FPS = re.compile(r"Collected \{0: (\d+)\}, FPS: ([\d.]+)")

# run directory -> (arm, scale, seed). Explicit on purpose.
MANIFEST = {}


def _add(arm, scale, names, partial=False, collapsed=False):
    for i, name in enumerate(names, start=1):
        MANIFEST[name] = dict(arm=arm, scale=scale, seed=i,
                              partial=partial, collapsed=collapsed)


_add("gru_hpov4", "50M", ["gru_hpov4_50m_seed%d" % i for i in (1, 2, 3)])
_add("mamba3_large", "50M", ["m3_large_50m_seed%d" % i for i in (1, 2, 3)])
_add("mamba3_hpov4", "50M", ["mamba3_hpov4_50m_seed%d" % i for i in (1, 2, 3)])
_add("gdn", "50M", ["gdn_50m_seed%d" % i for i in range(1, 6)])
_add("gru_baseline", "250M", ["gru_250m_baseline"])
_add("gru_v4_lr3", "250M", ["gru_250m_v4_lr3"])
_add("gru_v4", "250M", ["gru_250m_v4"], collapsed=True)
_add("mamba3_hpov4", "250M", ["mamba3_hpov4_250m", "mamba3_hpov4_250m_seed2",
                              "mamba3_hpov4_250m_seed3"])
_add("gdn", "250M", ["gdn_250m_seed%d" % i for i in (1, 2, 3)])
_add("mamba3_best", "250M", ["mamba3_best_250m"])

# Killed before their step budget: never averaged into a headline row.
PARTIAL = {
    "gru_baseline_50m": "gru_baseline",
    "gru_baseline_50m_m": "gru_baseline",
    "gru_baseline_50m_seed42": "gru_baseline",
    "gru_optimized_50m": "gru_optimized",
    "gru_optimized_50m_seed42": "gru_optimized",
}

# Single runs on a different `git.diff` than the seed groups that share their
# arm name (mamba3_50m_m and mamba3_fixed_50m are 66-line diffs, the seed groups
# are ~1000-line diffs). They stay one-row arms rather than being merged into a
# "mamba3" average that would hide two different code states.
SINGLE = {
    "mamba3_50m_m": ("mamba3_m", "50M"),
    "mamba3_best_50m": ("mamba3_best", "50M"),
    "mamba3_fixed_50m": ("mamba3_fixed", "50M"),
    "mamba3_fixed_v2_50m": ("mamba3_fixed_v2", "50M"),
    "mamba3_50m_final": ("mamba3_final", "50M"),
}


def parse_log(path):
    """(series, steps, fps, wall_h) for one sf_log.txt."""
    text = open(path, errors="replace").read()
    series = [(ts, float(v)) for ts, v in RE_AVG.findall(text)]
    stamps = [ts for ts, _ in series]
    fps = RE_FPS.findall(text)
    steps = int(fps[-1][0]) if fps else None
    rate = float(fps[-1][1]) if fps else None
    wall = None
    if len(stamps) > 1:
        t0 = datetime.fromisoformat(stamps[0])
        t1 = datetime.fromisoformat(stamps[-1])
        wall = round((t1 - t0).total_seconds() / 3600.0, 2)
    return series, steps, rate, wall


def classify(name, points, steps):
    if name in MANIFEST:
        return "reported"
    if name in SINGLE:
        return "single"
    if name in PARTIAL:
        return "partial"
    if name.startswith("hpo"):
        return "hpo"
    if points == 0:
        return "no_series"
    if steps is not None and steps < 10_000_000:
        return "smoke"
    return "unclassified"


def code_signature(diff_text):
    """Hash of the diff with ViZDoom's generated config removed.

    `_vizdoom.ini` is rewritten by the engine at every launch, so its hunk carries
    a timestamp: two runs of identical code hash differently (gru_hpov4 seeds 1 and
    3 differed only in that line). Dropping that file plus `index` lines leaves a
    signature of the code itself.
    """
    import hashlib
    keep, skipping = [], False
    for line in diff_text.splitlines():
        if line.startswith("diff --git "):
            skipping = "_vizdoom.ini" in line
        if skipping or line.startswith("index "):
            continue
        keep.append(line)
    return hashlib.md5("\n".join(keep).encode()).hexdigest()[:8]


def provenance(run_dir):
    """git_hash from config.json plus the code signature of the run's git.diff.

    config.json's git_hash alone is not enough: five mamba3_50m* runs share
    1fa02f7 and differ in code, because the core was edited in the working tree.
    The per-run git.diff is the only record of what actually ran.
    """
    git_hash, diff_md5 = "", ""
    rnn_type, rnn_size = "", ""
    cfg = os.path.join(run_dir, "config.json")
    if os.path.exists(cfg):
        import json
        try:
            conf = json.load(open(cfg))
        except ValueError:
            conf = {}
        git_hash = (conf.get("git_hash") or "")[:10]
        rnn_type = conf.get("rnn_type") or ""
        rnn_size = conf.get("rnn_size") or ""
    patch = os.path.join(run_dir, "git.diff")
    if os.path.exists(patch):
        diff_md5 = code_signature(open(patch, errors="replace").read())
    return git_hash, diff_md5, rnn_type, rnn_size


def collect(train_dir):
    rows = []
    for path in sorted(glob.glob(os.path.join(train_dir, "*", "sf_log.txt"))):
        name = os.path.basename(os.path.dirname(path))
        series, steps, fps, wall = parse_log(path)
        values = [v for _, v in series]
        meta = MANIFEST.get(name, {})
        single = SINGLE.get(name, ("", ""))
        git_hash, diff_md5, rnn_type, rnn_size = provenance(os.path.dirname(path))
        row = dict(
            run=name,
            group=classify(name, len(values), steps),
            arm=meta.get("arm") or PARTIAL.get(name) or single[0],
            scale=meta.get("scale") or ("50M" if name in PARTIAL else single[1]),
            seed=meta.get("seed", ""),
            points=len(values),
            steps=steps if steps is not None else "",
            last20=round(statistics.mean(values[-20:]), 2) if values else "",
            final=round(values[-1], 2) if values else "",
            best=round(max(values), 2) if values else "",
            fps=fps if fps is not None else "",
            wall_h=wall if wall is not None else "",
            collapsed="yes" if meta.get("collapsed") else "",
            rnn_type=rnn_type,
            rnn_size=rnn_size,
            git_hash=git_hash,
            diff_md5=diff_md5,
            log="train_dir/%s/sf_log.txt" % name,
        )
        rows.append(row)
    return rows


def _fmt(value, spec="%.2f"):
    return "" if value == "" else spec % value


def arm_rows(rows, scale, groups=("reported",)):
    """{arm: [rows]} for one scale, restricted to the given groups."""
    buckets = {}
    for r in rows:
        if r["scale"] != scale or not r["arm"] or r["group"] not in groups:
            continue
        buckets.setdefault(r["arm"], []).append(r)
    for arm in buckets:
        buckets[arm].sort(key=lambda r: str(r["seed"]))
    return buckets


def render_table(buckets):
    lines = [
        "| arm | per-run last20 | n | mean ± s.d. | wall h | rnn | code (git / diff md5) | source |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for arm, rs in buckets.items():
        vals = [r["last20"] for r in rs if r["last20"] != ""]
        walls = [r["wall_h"] for r in rs if r["wall_h"] != ""]
        per_run = " / ".join(_fmt(r["last20"]) + ("*" if r["collapsed"] else "") for r in rs)
        mean = statistics.mean(vals) if vals else ""
        sd = statistics.stdev(vals) if len(vals) > 1 else None
        mean_s = _fmt(mean)
        if sd is not None:
            mean_s += " ± %.2f" % sd
        src = ", ".join("`%s`" % r["log"] for r in rs)
        code = ", ".join("%s/%s" % (r["git_hash"] or "?", r["diff_md5"] or "?") for r in rs)
        rnns = sorted({"%s/%s" % (r["rnn_type"], r["rnn_size"]) for r in rs if r["rnn_type"]})
        lines.append("| `%s` | %s | %d | %s | %s | %s | %s | %s |" % (
            arm, per_run, len(rs), mean_s, _fmt(statistics.mean(walls) if walls else ""),
            ", ".join(rnns), code, src))
    return lines


def render_markdown(rows, label, generated):
    reported = arm_rows(rows, "50M")
    reported250 = arm_rows(rows, "250M")
    partial = arm_rows(rows, "50M", groups=("partial",))
    singles = arm_rows(rows, "50M", groups=("single",))
    counts = {}
    for r in rows:
        counts[r["group"]] = counts.get(r["group"], 0) + 1
    out = [
        "# Cluster results (ViZDoom `doom_benchmark`, APPO)",
        "",
        "**Generated** by `scripts/agg_runs.py`; provenance: %s; parsed from `%s`." % (label, generated),
        "Do not edit by hand; re-run the script instead.",
        "",
        "Metric: `last20` = mean of the last 20 per-iteration **Avg episode reward**",
        "values logged by Sample Factory (one value every 5 s of training), per run.",
        "`mean ± s.d.` is over runs with the *sample* standard deviation (n-1).",
        "Every row names the `sf_log.txt` it was parsed from, so each number is",
        "checkable against the raw log.",
        "",
        "Code identity: `git_hash` is what Sample Factory recorded in `config.json`,",
        "`diff md5` is a hash of that run's own `git.diff` after removing the engine-",
        "generated `_vizdoom.ini` hunk (it carries a launch timestamp, so hashing the",
        "raw diff made two identical runs differ). Both are weak: the `1fa02f7`",
        "checkout is dirty and five `mamba3_50m*` runs carry the same diff, so what",
        "separates those arms is the *arguments*, not the patch -- which is why `rnn`",
        "is in the tables and the diff is never used to decide which runs are seeds.",
        "`rnn_size` is the engine parameter, not a cell count: the SSM cores use a",
        "flattened size (mamba3 16960 = d_model 256 / d_state 64, mamba3 66688 =",
        "512 / 128).",
        "",
        "## 50M env frames",
        "",
    ]
    out += render_table(reported)
    out += ["", "## 250M env frames", ""]
    out += render_table(reported250)
    if singles:
        out += [
            "",
            "## Single runs on a different code state (not seed groups)",
            "",
            "One run each, on a different `git.diff` than the seed groups above.",
            "Averaging them into the `mamba3` rows would mix code states, so they stay",
            "separate: the 50M numbers below are not seeds of one configuration.",
            "",
        ]
        out += render_table(singles)
    if partial:
        out += [
            "",
            "## Runs killed before their budget (not averaged above)",
            "",
            "Different step counts, so these are listed rather than combined.",
            "",
        ]
        out += render_table(partial)
    out += [
        "",
        "## Excluded run directories",
        "",
        "| group | dirs | why |",
        "|---|---|---|",
        "| `hpo` | %d | 10.03M-frame hyper-parameter trials, not final arms |" % counts.get("hpo", 0),
        "| `smoke` | %d | under 10M frames |" % counts.get("smoke", 0),
        "| `no_series` | %d | crashed or restarted before logging a reward |" % counts.get("no_series", 0),
        "| `unclassified` | %d | not in the run manifest -- investigate before citing |" % counts.get("unclassified", 0),
        "",
        "`*` marks a run that collapsed (peaked mid-training, then decayed) and",
        "must not be read as a low-scoring sample of a healthy configuration.",
        "",
        "## Regenerate",
        "",
        "```bash",
        "python scripts/agg_runs.py --train-dir /path/to/train_dir \\",
        "    --csv results/runs.csv --md results/CLUSTER_RESULTS.md",
        "```",
        "",
    ]
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-dir", required=True)
    ap.add_argument("--csv", default="results/runs.csv")
    ap.add_argument("--md", default="results/CLUSTER_RESULTS.md")
    ap.add_argument("--generated", default="2026-11-12",
                    help="provenance label written into the markdown")
    ap.add_argument("--allow-unclassified", action="store_true")
    args = ap.parse_args()

    rows = collect(args.train_dir)
    if not rows:
        sys.exit("no */sf_log.txt under %s" % args.train_dir)

    for path in (args.csv, args.md):
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
    with open(args.csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with open(args.md, "w") as f:
        f.write(render_markdown(rows, args.generated, args.train_dir))

    bad = [r for r in rows if r["group"] == "unclassified"
           and isinstance(r["steps"], int) and r["steps"] >= MIN_REPORT_STEPS]
    print("%d runs -> %s, %s" % (len(rows), args.csv, args.md))
    for r in bad:
        print("  UNCLASSIFIED >=50M: %s (%s steps)" % (r["run"], r["steps"]))
    if bad and not args.allow_unclassified:
        sys.exit("%d run(s) reaching %d frames are not in MANIFEST; add them "
                 "instead of letting a seed fall out of the tables"
                 % (len(bad), MIN_REPORT_STEPS))


if __name__ == "__main__":
    main()
