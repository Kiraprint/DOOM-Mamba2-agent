#!/usr/bin/env python3
"""Rebuild results/runs.csv and results/CLUSTER_RESULTS.md from Sample-Factory logs.

One per-run artifact survived this study: train_dir/<run>/sf_log.txt
(results_summary.json was gitignored upstream and never committed). Every number
in results/CLUSTER_RESULTS.md is derived here from those logs.

    python scripts/agg_runs.py --train-dir <train_dir> \
        --csv results/runs.csv --md results/CLUSTER_RESULTS.md

Bucketing is driven by the logs, not by directory names:

* a run's budget is `train_for_env_steps` from its own `config.json`;
* the frames it got through are its last `Collected {0: N}` line, or, when that
  line is absent, the largest `Total num frames: N` line;
* a run is *finished* when those frames reach 99% of the budget (`Total num
  frames` is logged every 5 s, so the last one sits a little under the budget
  even for a run that completed);
* `hpo`/`smoke` name and size buckets are only consulted *after* the budget test,
  so a renamed 50M run cannot be filed away as an HPO trial or a smoke.

Nothing is written unless every check passes; the script then writes both files
through `os.replace`, so a failing regeneration cannot clobber the committed
tables. It exits nonzero if:

* a run named in MANIFEST/SINGLE/PARTIAL/BROKEN has no directory, or has one with
  no `sf_log.txt` in it (a renamed or deleted run cannot silently shrink `n`);
* a finished run with a >= 50M budget is not in MANIFEST or SINGLE -- whatever
  its directory is called, and even if its `Collected` lines were stripped;
* a run filed as PARTIAL actually reached its budget;
* a MANIFEST/SINGLE run collected 0 frames (it never trained: quarantine it);
* any run is left `unclassified`.
"""

import argparse
import csv
import glob
import io
import json
import os
import re
import statistics
import sys
import tempfile
from datetime import datetime

MIN_REPORT_FRAMES = 50_000_000   # a budget at or above this must be in the manifest
SMOKE_FRAMES = 10_000_000        # below this a run is a smoke test
BUDGET_SATISFIED = 0.99          # fraction of train_for_env_steps that counts as finished

RE_AVG = re.compile(
    r"\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d),\d+\].*Avg episode reward: \[\(0, '(-?\d+\.?\d*)'\)\]"
)
RE_COLLECTED = re.compile(r"Collected \{0: (\d+)\}, FPS: ([\d.]+)")
RE_TOTAL_FRAMES = re.compile(r"Total num frames: (\d+)")
RE_DATE = re.compile(r"^\d{4}-\d\d-\d\d$")

# run directory -> arm. Explicit on purpose: this mapping, not the directory name,
# decides what is a replicate of what.
MANIFEST = {}


def _add(arm, scale, names, collapsed=()):
    for name in names:
        MANIFEST[name] = dict(arm=arm, scale=scale, collapsed=name in collapsed)


# Full-budget runs that form the headline arms. `seed=` in each run's config.json
# is the Sample Factory seed; it is `null` for most of these runs, and the script
# never substitutes a position in this list for it.
_add("gru_hpov4", "50M", ["gru_hpov4_50m_seed%d" % i for i in (1, 2, 3)])
_add("mamba3_large", "50M", ["m3_large_50m_seed%d" % i for i in (1, 2, 3)])
_add("mamba3_hpov4", "50M", ["mamba3_hpov4_50m_seed%d" % i for i in (1, 2, 3)])
_add("gdn", "50M", ["gdn_50m_seed%d" % i for i in range(1, 6)])
# The only plain-GRU 50M run that reached its budget (one session, 50,020,352
# frames). Its two siblings, gru_baseline_50m and gru_baseline_50m_seed42, were
# killed early and stay in PARTIAL.
_add("gru_baseline", "50M", ["gru_baseline_50m_m"])
_add("gru_baseline", "250M", ["gru_250m_baseline"])
_add("gru_v4_lr3", "250M", ["gru_250m_v4_lr3"])
_add("gru_v4", "250M", ["gru_250m_v4"], collapsed=["gru_250m_v4"])
_add("mamba3_hpov4", "250M", ["mamba3_hpov4_250m", "mamba3_hpov4_250m_seed2",
                              "mamba3_hpov4_250m_seed3"])
_add("gdn", "250M", ["gdn_250m_seed%d" % i for i in (1, 2, 3)])
_add("mamba3_best", "250M", ["mamba3_best_250m"])

# Killed before their step budget: never averaged into a headline row, and the
# table they land in prints no mean. A run listed here that turns out to have
# reached its budget fails the guard instead of being reported as partial.
PARTIAL = {
    "gru_baseline_50m": "gru_baseline",
    "gru_baseline_50m_seed42": "gru_baseline",
    "gru_optimized_50m": "gru_optimized",
    "gru_optimized_50m_seed42": "gru_optimized",
}

# Single runs that are not replicates of any arm above.
SINGLE = {
    "mamba3_50m_m": dict(arm="mamba3_m", scale="50M", collapsed=True),
    "mamba3_best_50m": dict(arm="mamba3_best", scale="50M", collapsed=False),
    "mamba3_fixed_v2_50m": dict(arm="mamba3_fixed_v2", scale="50M", collapsed=False),
    "mamba3_50m_final": dict(arm="mamba3_final", scale="50M", collapsed=False),
}

# Runs whose log shows the agent never trained. Reported as quarantined, never as
# a data point; a MANIFEST/SINGLE run that collected 0 frames fails the guard and
# has to be moved here.
BROKEN = {
    "mamba3_fixed_50m":
        "collected 0 env frames (`Total num frames: 0`, `Fps is ... 0.0`) and logged "
        "the same 0.43 reward 209 times: the worker never stepped the env, so this "
        "is not a 50M result",
}

COLUMNS = [
    "run", "group", "arm", "scale", "seed", "points",
    "collected_frames", "frames_reached", "budget",
    "last20", "final", "best", "fps", "wall_h", "collapsed",
    "rnn_type", "rnn_size", "git_hash", "diff_md5", "log",
]


def read_config(run_dir):
    path = os.path.join(run_dir, "config.json")
    if not os.path.exists(path):
        return {}
    try:
        return json.load(open(path))
    except ValueError:
        return {}


def parse_log(path):
    """(series, collected, max_total_frames, fps, wall_h) for one sf_log.txt.

    `collected` is the last `Collected {0: N}` frame count (one session), kept
    apart from the largest `Total num frames: N` (which accumulates across a
    restart). `Samples:` is deliberately not used as a frame count: it is a
    different unit.
    """
    text = open(path, errors="replace").read()
    series = [(ts, float(v)) for ts, v in RE_AVG.findall(text)]
    stamps = [ts for ts, _ in series]
    collected = RE_COLLECTED.findall(text)
    totals = [int(v) for v in RE_TOTAL_FRAMES.findall(text)]
    steps = int(collected[-1][0]) if collected else None
    rate = float(collected[-1][1]) if collected else None
    wall = None
    if len(stamps) > 1:
        t0 = datetime.fromisoformat(stamps[0])
        t1 = datetime.fromisoformat(stamps[-1])
        wall = round((t1 - t0).total_seconds() / 3600.0, 2)
    return series, steps, (max(totals) if totals else None), rate, wall


def classify(name, points, frames, budget):
    if name in BROKEN:
        return "broken"
    if name in MANIFEST:
        return "reported"
    if name in SINGLE:
        return "single"
    if name in PARTIAL:
        return "partial"
    # Name and size buckets are only reached after this: anything that got
    # through a reportable budget is a headline run and has to be in the manifest.
    if is_finished(frames, budget) and budget >= MIN_REPORT_FRAMES:
        return "unclassified"
    if name.startswith("hpo"):
        return "hpo"
    if points == 0:
        return "no_series"
    if frames is None or frames < SMOKE_FRAMES:
        return "smoke"
    return "unclassified"


def is_finished(frames, budget):
    return (frames is not None and budget is not None
            and frames >= BUDGET_SATISFIED * budget)


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


def provenance(run_dir, conf):
    """git_hash from config.json plus the code signature of the run's git.diff.

    config.json's git_hash alone is not enough: the checkout is dirty and runs
    that share `1fa02f7527` differ in code. The per-run git.diff is the only
    record of what actually ran, and even it is weak -- see the generated header.
    """
    git_hash = (conf.get("git_hash") or "")[:10]
    rnn_type = conf.get("rnn_type") or ""
    rnn_size = conf.get("rnn_size") or ""
    diff_md5 = ""
    patch = os.path.join(run_dir, "git.diff")
    if os.path.exists(patch):
        diff_md5 = code_signature(open(patch, errors="replace").read())
    return git_hash, diff_md5, rnn_type, rnn_size


def _int_or_blank(value):
    return value if isinstance(value, int) and not isinstance(value, bool) else ""


def collect(train_dir):
    rows = []
    for path in sorted(glob.glob(os.path.join(train_dir, "*", "sf_log.txt"))):
        name = os.path.basename(os.path.dirname(path))
        run_dir = os.path.dirname(path)
        series, steps, max_frames, fps, wall = parse_log(path)
        conf = read_config(run_dir)
        values = [v for _, v in series]
        frames = steps if steps is not None else max_frames
        budget = _int_or_blank(conf.get("train_for_env_steps"))
        meta = MANIFEST.get(name, {})
        single = SINGLE.get(name, {})
        exact = statistics.mean(values[-20:]) if values else ""
        row = dict(
            run=name,
            group=classify(name, len(values), frames, budget),
            arm=meta.get("arm") or PARTIAL.get(name) or single.get("arm", ""),
            scale=meta.get("scale") or single.get("scale")
            or ("50M" if name in PARTIAL else ""),
            # The seed Sample Factory actually ran with (config.json), never a
            # position in MANIFEST: most of these runs recorded `seed: null`.
            seed=_int_or_blank(conf.get("seed")),
            points=len(values),
            collected_frames=_int_or_blank(steps),
            frames_reached=_int_or_blank(frames),
            budget=budget,
            last20=round(exact, 2) if values else "",
            final=round(values[-1], 2) if values else "",
            best=round(max(values), 2) if values else "",
            fps=fps if fps is not None else "",
            wall_h=wall if wall is not None else "",
            collapsed="yes" if (meta.get("collapsed") or single.get("collapsed")) else "",
            rnn_type=conf.get("rnn_type") or "",
            rnn_size=conf.get("rnn_size") or "",
            git_hash="",
            diff_md5="",
            log="train_dir/%s/sf_log.txt" % name,
            _exact=exact,
        )
        row["git_hash"], row["diff_md5"], rnn_type, rnn_size = provenance(run_dir, conf)
        row["rnn_type"], row["rnn_size"] = rnn_type, rnn_size
        rows.append(row)
    return rows


def validate(rows, train_dir):
    """Every reason the committed tables must not be written. Empty list = ok."""
    problems = []
    found = {r["run"] for r in rows}
    dirs = {os.path.basename(p) for p in glob.glob(os.path.join(train_dir, "*"))
            if os.path.isdir(p)}
    for name in sorted(set(MANIFEST) | set(SINGLE) | set(PARTIAL) | set(BROKEN)):
        if name not in dirs:
            problems.append("manifest run %s: no such directory under %s "
                            "(renamed or deleted run would silently change n)"
                            % (name, train_dir))
        elif name not in found:
            problems.append("manifest run %s: directory exists but has no "
                            "sf_log.txt under %s" % (name, train_dir))
    for r in rows:
        if r["group"] == "unclassified":
            problems.append(
                "%s is unclassified: %s frames reached, budget %s -- add it to "
                "MANIFEST or SINGLE, or quarantine it in BROKEN"
                % (r["run"], r["frames_reached"] or "no", r["budget"] or "unknown"))
        if r["group"] in ("reported", "single") and r["frames_reached"] == 0:
            problems.append("%s is filed as a headline run but collected 0 frames: "
                            "quarantine it in BROKEN" % r["run"])
        if r["group"] == "partial" and is_finished(r["frames_reached"], r["budget"]):
            problems.append(
                "%s reached its budget (%s of %s frames) but is filed as PARTIAL: "
                "move it to MANIFEST" % (r["run"], r["frames_reached"], r["budget"]))
    return problems


def _fmt(value, spec="%.2f"):
    return "" if value == "" else spec % value


def _mframes(value):
    return "" if value == "" else "%.2fM" % (value / 1e6)


def arm_rows(rows, scale, groups=("reported",)):
    """{arm: [rows]} for one scale, restricted to the given groups."""
    buckets = {}
    for r in rows:
        if r["scale"] != scale or not r["arm"] or r["group"] not in groups:
            continue
        buckets.setdefault(r["arm"], []).append(r)
    for arm in buckets:
        buckets[arm].sort(key=lambda r: r["run"])
    return buckets


def render_table(buckets, aggregate=True):
    head = "| arm | per-run last20 | n | mean ± s.d. | wall h | rnn | code (git / diff md5) | source |"
    rule = "|---|---|---|---|---|---|---|---|"
    if not aggregate:
        head = ("| arm | per-run last20 | frames reached | n | wall h | rnn | "
                "code (git / diff md5) | source |")
    lines = [head, rule]
    for arm, rs in buckets.items():
        vals = [r["_exact"] for r in rs if r["_exact"] != ""]
        walls = [r["wall_h"] for r in rs if r["wall_h"] != ""]
        per_run = " / ".join(_fmt(r["last20"]) + ("*" if r["collapsed"] else "") for r in rs)
        src = ", ".join("`%s`" % r["log"] for r in rs)
        code = ", ".join("%s/%s" % (r["git_hash"] or "?", r["diff_md5"] or "?") for r in rs)
        rnns = sorted({"%s/%s" % (r["rnn_type"], r["rnn_size"]) for r in rs if r["rnn_type"]})
        wall = _fmt(statistics.mean(walls) if walls else "")
        if aggregate:
            mean = statistics.mean(vals) if vals else ""
            sd = statistics.stdev(vals) if len(vals) > 1 else None
            mean_s = _fmt(mean)
            if sd is not None:
                mean_s += " ± %.2f" % sd
            lines.append("| `%s` | %s | %d | %s | %s | %s | %s | %s |" % (
                arm, per_run, len(rs), mean_s, wall, ", ".join(rnns), code, src))
        else:
            frames = " / ".join(_mframes(r["frames_reached"]) or "no log line" for r in rs)
            lines.append("| `%s` | %s | %s | %d | %s | %s | %s | %s |" % (
                arm, per_run, frames, len(rs), wall, ", ".join(rnns), code, src))
    return lines


def hpo_breakdown(rows):
    by = {"mamba3": 0, "gru": 0, "mamba2": 0, "numbered": 0}
    top = 0
    for r in rows:
        if r["group"] != "hpo":
            continue
        name = r["run"]
        if name.startswith("hpo_mamba3"):
            by["mamba3"] += 1
        elif name.startswith("hpo_gru"):
            by["gru"] += 1
        elif name.startswith("hpo_mamba2"):
            by["mamba2"] += 1
        else:
            by["numbered"] += 1
        if r["frames_reached"] != "":
            top = max(top, r["frames_reached"])
    return by, (top or None)


def render_markdown(rows, train_dir, dir_count, date):
    reported = arm_rows(rows, "50M")
    reported250 = arm_rows(rows, "250M")
    partial = arm_rows(rows, "50M", groups=("partial",))
    singles = arm_rows(rows, "50M", groups=("single",))
    counts = {}
    for r in rows:
        counts[r["group"]] = counts.get(r["group"], 0) + 1
    no_log = sorted(os.path.basename(p) for p in glob.glob(os.path.join(train_dir, "*"))
                    if os.path.isdir(p) and not os.path.exists(os.path.join(p, "sf_log.txt")))
    hpo, hpo_top = hpo_breakdown(rows)
    out = [
        "# Cluster results (ViZDoom `doom_benchmark`, APPO)",
        "",
        "**Generated** by `scripts/agg_runs.py` on %s from the %d `sf_log.txt` files in"
        % (date, len(rows)),
        "the %d run directories under `%s`; %d of those directories hold a `config.json`"
        % (dir_count, train_dir, len(no_log)),
        "but no log and are listed as `no_log` below.",
        "Do not edit by hand; re-run the script instead.",
        "",
        "Metric: `last20` = mean of the last 20 per-iteration **Avg episode reward**",
        "values logged by Sample Factory (one value every 5 s of training), per run.",
        "`mean ± s.d.` is over runs with the *sample* standard deviation (n-1),",
        "computed from the unrounded 20-value means; a row with n=1 is a single run",
        "and prints no s.d. Every row names the `sf_log.txt` it was parsed from, so",
        "each number is checkable against the raw log, and `results/runs.csv` carries",
        "the frame counts behind each bucket.",
        "",
        "Bucketing comes from the logs, never from directory names: `frames reached`",
        "is the run's last `Collected {0: N}` line (or its largest `Total num",
        "frames: N` when that line is missing) and its budget is",
        "`train_for_env_steps` from its own `config.json`. A run counts as finished",
        "at 99% of that budget. `seed` in `results/runs.csv` is the `seed` field of",
        "`config.json` and is blank where the run did not record one -- the runs are",
        "replicates of an arm, not a numbered seed series.",
        "",
        "Code identity: `git_hash` is what Sample Factory recorded in `config.json`,",
        "`diff md5` is a hash of that run's own `git.diff` after removing the engine-",
        "generated `_vizdoom.ini` hunk (it carries a launch timestamp, so hashing the",
        "raw diff made two identical runs differ). Both are weak: most runs come from",
        "the same dirty `1fa02f7` checkout, and runs can share a diff signature while",
        "differing in arguments. That is why which runs are replicates of an arm comes",
        "from `MANIFEST` in the script, and why the script fails if a run that reached",
        "its budget is missing from it.",
        "`rnn_size` is the engine parameter, not a cell count: the SSM cores use a",
        "flattened size (mamba3 16960 = d_model 256 / d_state 64, mamba3 66688 =",
        "512 / 128).",
        "",
        "Where the code lives: the campaign ran in the `DOOM` repository. Most runs",
        "are on its `master` (1fa02f7), the GDN runs on `cluster/gdn` (fad9545, pushed",
        "pinned so the core is not only on the cluster). The uncommitted working-tree",
        "patch those runs shared is committed here as",
        "`results/working_tree_1fa02f7.diff` -- without it the per-run `git.diff` files,",
        "which are the only exact code record, cannot be reconstructed from the repo.",
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
            "## Single runs (not replicates of any arm above)",
            "",
            "Each row is one run. None of them is a seed of a 50M arm, and averaging",
            "them into the `mamba3` rows would mix code states: their `git.diff`",
            "signature is never the `d791fabb` of the 50M seed groups. The diff alone",
            "does not make each of them a distinct code state either -- some of these",
            "runs share their signature with each other and differ only in arguments",
            "(all of them share the same `rnn`, `batch_size` and `rollout`), so each is",
            "listed on its own rather than treated as a configuration in its own right.",
            "",
        ]
        out += render_table(singles)
    if partial:
        out += [
            "",
            "## Runs killed before their budget",
            "",
            "None of these reached its own `train_for_env_steps` budget, and they have",
            "different step counts, so no mean is printed: each row is a list.",
            "A run that did reach its budget here would fail the guard and have to be",
            "moved into a headline arm.",
            "",
        ]
        out += render_table(partial, aggregate=False)
    out += [
        "",
        "## Excluded run directories",
        "",
        "| group | dirs | why |",
        "|---|---|---|",
        "| `quarantined` | %d | never trained; each one is named below |" % counts.get("broken", 0),
        "| `hpo` | %d | hyper-parameter trials at %s frames (%d mamba3, %d gru, %d mamba2, "
        "%d legacy `hpo_<N>_` trials), not final arms |" % (
            counts.get("hpo", 0), _mframes(hpo_top) or "10M",
            hpo["mamba3"], hpo["gru"], hpo["mamba2"], hpo["numbered"]),
        "| `smoke` | %d | under %dM frames |" % (
            counts.get("smoke", 0), SMOKE_FRAMES // 1_000_000),
        "| `no_series` | %d | crashed or restarted before logging a reward |" % counts.get("no_series", 0),
        "| `no_log` | %d | directory has a `config.json` but no `sf_log.txt`: %s |" % (
            len(no_log), ", ".join("`%s`" % n for n in no_log) or "none"),
        "| `unclassified` | %d | empty by construction: the script exits nonzero if a run "
        "lands here |" % counts.get("unclassified", 0),
        "",
    ]
    if counts.get("broken"):
        out.append("Quarantined, not tabulated:")
        for name, reason in sorted(BROKEN.items()):
            out.append("- `%s` -- %s (`train_dir/%s/sf_log.txt`)." % (name, reason, name))
        out.append("")
    out += [
        "`*` marks a run that collapsed (peaked mid-training, then decayed) or is",
        "otherwise degenerate, and must not be read as a low-scoring sample of a",
        "healthy configuration.",
        "",
        "## Regenerate",
        "",
        "```bash",
        "python scripts/agg_runs.py --train-dir %s \\" % train_dir,
        "    --csv results/runs.csv --md results/CLUSTER_RESULTS.md \\",
        "    --date %s" % date,
        "```",
        "",
    ]
    return "\n".join(out)


def _atomic_write(path, text):
    parent = os.path.dirname(path) or "."
    os.makedirs(parent, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=parent, prefix=".tmp-", text=True)
    try:
        with os.fdopen(fd, "w", newline="") as f:
            f.write(text)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-dir", required=True)
    ap.add_argument("--csv", default="results/runs.csv")
    ap.add_argument("--md", default="results/CLUSTER_RESULTS.md")
    ap.add_argument("--date", default="2026-11-12",
                    help="date stamp of the run being reported, YYYY-MM-DD; it is the "
                         "only free text in the generated markdown")
    args = ap.parse_args()
    if not RE_DATE.match(args.date):
        sys.exit("--date must be YYYY-MM-DD, got %r" % args.date)

    rows = collect(args.train_dir)
    if not rows:
        sys.exit("no */sf_log.txt under %s" % args.train_dir)

    problems = validate(rows, args.train_dir)
    if problems:
        print("%d problem(s); nothing written:" % len(problems), file=sys.stderr)
        for p in problems:
            print("  %s" % p, file=sys.stderr)
        sys.exit(1)

    dir_count = sum(1 for p in glob.glob(os.path.join(args.train_dir, "*"))
                    if os.path.isdir(p))
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=COLUMNS, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    _atomic_write(args.csv, buf.getvalue())
    _atomic_write(args.md, render_markdown(rows, args.train_dir, dir_count, args.date))
    print("%d sf_log.txt under %d dirs -> %s, %s"
          % (len(rows), dir_count, args.csv, args.md))


if __name__ == "__main__":
    main()
