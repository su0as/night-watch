"""Build TabPFN's table: one row per training-log window, labelled 1 once the run is broken."""

from __future__ import annotations

import csv
import math
from pathlib import Path

from ..features import FEATURE_NAMES, window_features
from ..runlog import RunReader, list_runs
from .groundtruth import failures_for_run

META = ["run_id", "step", "t", "label"]


def run_windows(run_dir: str | Path, k: int = 8, min_rows: int = 6) -> list[dict]:
    reader = RunReader(run_dir)
    rows = reader.metrics()
    fails = failures_for_run(run_dir)
    onset = min((f.onset_t for f in fails), default=math.inf)
    out = []
    for i in range(min_rows, len(rows) + 1):
        feats = window_features(rows[:i], k)
        t = float(rows[i - 1]["t"])
        out.append({"run_id": reader.run_id, "step": int(rows[i - 1].get("step", 0)), "t": t,
                    "label": 1 if t >= onset else 0, **feats})
    return out


def build_table(runs_dir: str | Path, out_csv: str | Path, k: int = 8) -> dict:
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    n_rows = n_pos = 0
    runs = list_runs(runs_dir)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=META + FEATURE_NAMES)
        w.writeheader()
        for run in runs:
            for row in run_windows(run, k):
                w.writerow({key: ("" if isinstance(v, float) and math.isnan(v) else v) for key, v in row.items()})
                n_rows += 1
                n_pos += row["label"]
    return {"runs": len(runs), "rows": n_rows, "positive": n_pos, "path": str(out_csv)}
