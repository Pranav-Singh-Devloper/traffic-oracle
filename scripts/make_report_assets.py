"""Convert results/metrics CSVs to LaTeX/Markdown and fill the README results block.

Usage: python scripts/make_report_assets.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from traffic.evaluation.report_numbers import write_numbers
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
    # LaTeX table: MAE and RMSE only (MAPE stays in the CSV / README) so it fits one column.
    tex_wide = wide[["model"] + [c for c in wide.columns if c.startswith(("MAE", "RMSE"))]].copy()
    tex_wide.columns = [c.replace("min", "") for c in tex_wide.columns]
    (TAB / "main_table.tex").write_text(
        to_latex(
            tex_wide,
            "Test-set error (mph) by model and horizon (minutes).",
            "tab:main",
            bold_min=[c for c in tex_wide.columns if c != "model"],
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
    sig_path = MET / "significance.csv"
    if sig_path.exists():
        sg = pd.read_csv(sig_path)
        sg["pair"] = (sg.model_a + " vs " + sg.model_b).str.replace("_", " ")
        sg["h"] = sg.horizon * 5
        w = sg.pivot(index="pair", columns="h", values="dm_stat").round(1)
        pv = sg.pivot(index="pair", columns="h", values="p_value")
        out = w.copy()
        for c in w.columns:
            out[c] = [f"{v:.1f}" + ("*" if pv.loc[i, c] < 0.05 else "") for i, v in w[c].items()]
        out.columns = [f"DM@{c}" for c in out.columns]
        (TAB / "significance.tex").write_text(
            to_latex(
                out.reset_index(),
                "Diebold-Mariano statistics (negative: first model better; * p<0.05).",
                "tab:sig",
            )
        )
    n = write_numbers(MET, TAB / "numbers.tex")
    import shutil

    Path("report/figures").mkdir(parents=True, exist_ok=True)
    for src in list(Path("results/figures").rglob("*.pdf")) + list(
        Path("results/figures").rglob("*.png")
    ):
        if "training" in src.parts:
            continue
        shutil.copy(src, Path("report/figures") / src.name)
    print(f"wrote tables, {n} number macros, figures; updated README")


if __name__ == "__main__":
    main()
