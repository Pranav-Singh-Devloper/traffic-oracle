"""Generate data-driven docs (hybrid, control, ablations) from results/metrics CSVs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from traffic.evaluation.report_tables import to_markdown

MET = Path("results/metrics")
DISC = (
    "> This is an offline simulation on a synthetic network. It demonstrates how forecasts could "
    "inform a control policy. No live or real-world traffic control is performed or claimed.\n"
)


def control() -> None:
    s = pd.read_csv(MET / "control_summary.csv")
    cols = ["policy"] + [c for c in s.columns if c.endswith("_mean")]
    t = s[cols].copy()
    t.columns = [c.replace("_mean", "") for c in t.columns]
    txt = f"""# Simulated control results (auto-generated)

{DISC}
Assumptions: [control_assumptions.md](control_assumptions.md). Mean over 5 test weekdays; standard deviations are in `results/metrics/control_summary.csv`.

{to_markdown(t, 1)}

Reading guide: lower TTS, queue and congested cell-minutes are better; higher mainline speed is better.

Figures: `results/figures/control/`.
"""
    Path("docs/control_results.md").write_text(txt)


def main() -> None:
    if (MET / "control_summary.csv").exists():
        control()
        print("wrote docs/control_results.md")


if __name__ == "__main__":
    main()
