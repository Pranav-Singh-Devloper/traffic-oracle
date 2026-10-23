"""Convert results/metrics CSVs to LaTeX/Markdown and fill the README results block.

Usage: python scripts/make_report_assets.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from traffic.evaluation.report_tables import insert_readme_block, to_latex, to_markdown

MET, TAB = Path("results/metrics"), Path("report/tables")


def _fmt_pm(df: pd.DataFrame, metric: str) -> pd.Series:
    sd = df[f"{metric}_std"]
    return df[metric].map("{:.2f}".format) + sd.map(lambda s: "" if pd.isna(s) else f" ± {s:.2f}")


def main() -> None:
    TAB.mkdir(parents=True, exist_ok=True)
    mt = pd.read_csv(MET / "main_table.csv")
    mt["model"] = mt["model"].str.replace("_", " ")
    wide = mt.pivot(index="model", columns="horizon_min", values=["MAE", "RMSE", "MAPE"])
    wide.columns = [f"{m}@{h}min" for m, h in wide.columns]
    wide = wide.reset_index()
    (TAB / "main_table.tex").write_text(
        to_latex(
            wide,
            "Test-set error by model and horizon (mph; MAPE in \\%).",
            "tab:main",
            bold_min=[c for c in wide.columns if c != "model"],
        )
    )
    (TAB / "main_table.md").write_text(to_markdown(wide))
    po_path = MET / "peak_offpeak.csv"
    md = "### Test error by horizon (mph, masked)\n\n" + to_markdown(wide)
    if po_path.exists():
        po = pd.read_csv(po_path)
        po = po[po.horizon == 12].pivot(index="model", columns="slice", values="MAE").reset_index()
        md += "\n\n### MAE at 60 min by slice (mph)\n\n" + to_markdown(po)
        (TAB / "peak_offpeak.tex").write_text(
            to_latex(po, "MAE at 60 min by slice (mph).", "tab:peak")
        )
    readme = Path("README.md")
    insert_readme_block(readme, md)
    print("wrote tables and updated README")


if __name__ == "__main__":
    main()
