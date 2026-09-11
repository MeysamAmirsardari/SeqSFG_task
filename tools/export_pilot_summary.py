#!/usr/bin/env python3
"""Write per-cell counts from recorded sessions to verification/pilot_summary.json.

`data/` is gitignored and stays that way: it holds participant codes, consent records,
timestamps, response times and per-trial rows. None of that is in the file this writes.

What goes in: the task, the condition labels, and two integers per cell (how many trials, how
many of them were answered a particular way). Listeners are relabelled L1, L2, ... in the order
given on the command line. That is enough to redraw the figures in the overview notebook from a
clone that has no data directory, and not enough to identify anyone.

    python3 tools/export_pilot_summary.py data/P01/session_06 data/P01/session_07 ...
"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path


def _yesno_counts(rows, keys):
    """(key) -> n_signal, hits, n_noise, false_alarms, for a 'y'/'n' log."""
    out = defaultdict(lambda: [0, 0, 0, 0])
    for r in rows:
        k = keys(r)
        present = r.get("present", "1" if r.get("target_position") == "1" else "0") == "1"
        said_yes = r["response"] == "y"
        if present:
            out[k][0] += 1
            out[k][1] += int(said_yes)
        else:
            out[k][2] += 1
            out[k][3] += int(said_yes)
    return {"|".join(map(str, k)): dict(zip(("n_signal", "hits", "n_noise", "false_alarms"), v))
            for k, v in sorted(out.items())}


def summarise(sdir: Path) -> dict:
    rows = list(csv.DictReader(open(sdir / "trials.csv")))
    meta = json.loads((sdir / "session.json").read_text())
    cfg = meta.get("config", {})
    common = {"config_hash": meta.get("config_hash"), "status": meta.get("status"),
              "steps_ms": cfg.get("steps_ms"), "n_trials_logged": len(rows),
              "settings": {k: cfg.get(k) for k in
                           ("tone_dur_ms", "ramp_ms", "interval_dur_ms", "n_components",
                            "n_elements", "tones_per_channel", "figure_band_channels",
                            "iei_min_ms", "iei_max_ms", "matched_incidence",
                            "practice_criterion", "practice_n")}}
    prac = defaultdict(lambda: [0, 0])
    for r in rows:
        if r.get("block") != "practice":
            continue
        c = prac[r.get("variant", "?")]
        c[0] += 1
        c[1] += int(r["correct"])
    if prac:
        common["practice"] = {k: {"n": v[0], "n_correct": v[1]} for k, v in sorted(prac.items())}
    if not rows:
        return {**common, "task": "empty", "cells": {}}
    if "phase" in rows[0]:                       # exposure log
        d = meta.get("design", {})
        main = [r for r in rows if r["phase"] in ("pre", "post")]
        return {**common, "task": "exposure",
                "orders": d.get("heard_order"), "trained": d.get("trained"),
                "order_overlap": d.get("order_overlap"),
                "cells": _yesno_counts(main, lambda r: (r["phase"], r["role"], float(r["step_ms"])))}
    main = [r for r in rows if r.get("block") == "main"]
    if main and main[0].get("variant") == "yesno":                 # single-interval yes/no
        return {**common, "task": "yesno",
                "cells": _yesno_counts(main, lambda r: (float(r["step_ms"]),))}
    cells = defaultdict(lambda: [0, 0])                            # two-interval forced choice
    for r in main:
        c = cells[(r["variant"], float(r["step_ms"]))]
        c[0] += 1
        c[1] += int(r["correct"])
    return {**common, "task": "2ifc",
            "cells": {"|".join(map(str, k)): {"n": v[0], "n_correct": v[1]}
                      for k, v in sorted(cells.items())}}


def main(argv):
    if not argv:
        raise SystemExit(__doc__)
    listeners, out = {}, {}
    for p in argv:
        sdir = Path(p)
        code = sdir.parent.name
        label = listeners.setdefault(code, f"L{len(listeners) + 1}")
        out[f"{label}/{sdir.name}"] = summarise(sdir)
    dest = Path("verification/pilot_summary.json")
    dest.write_text(json.dumps(
        {"_note": "Per-cell counts only. No participant codes, timestamps, response times or "
                  "per-trial rows. Written by tools/export_pilot_summary.py.",
         "sessions": out}, indent=1) + "\n")
    print(f"wrote {dest}: {len(out)} sessions, {len(listeners)} listener(s)")


if __name__ == "__main__":
    main(sys.argv[1:])
