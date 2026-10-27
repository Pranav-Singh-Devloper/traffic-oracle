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
    import yaml

    rows = [
        ("max_depth", "3--10", "max_depth"),
        ("learning rate", "0.02--0.3 (log)", "learning_rate"),
        ("subsample", "0.5--1.0", "subsample"),
        ("colsample_bytree", "0.5--1.0", "colsample_bytree"),
        ("min_child_weight", "1--20", "min_child_weight"),
        ("$\\lambda$ (L2)", "0.001--10 (log)", "reg_lambda"),
        ("$\\gamma$", "0--5", "gamma"),
        ("loss", "sq.\\ / abs.\\ / pseudo-Huber", "objective"),
        ("trees (cap)", "400 (early stopping)", "n_estimators"),
    ]
    best = {
        k: yaml.safe_load(Path(f"configs/ml/{k}.best.yaml").read_text())["model"]["params"]
        for k in ("xgb_per_segment", "xgb_global", "xgb_spatial", "rf_multioutput")
    }

    def fmt(v):
        if isinstance(v, float):
            return f"{v:.3g}"
        return (
            str(v)
            .replace("reg:absoluteerror", "abs.\\ error")
            .replace("reg:squarederror", "sq.\\ error")
        )

    trows = []
    for label, rng, key in rows:
        trows.append(
            [label, rng]
            + [
                fmt(best[k].get(key, "--"))
                for k in ("xgb_per_segment", "xgb_global", "xgb_spatial")
            ]
        )
    tdf = pd.DataFrame(
        trows, columns=["parameter", "search range", "per-segment", "pooled", "spatial"]
    )
    (TAB / "tuning.tex").write_text(
        to_latex(
            tdf,
            "XGBoost search space and selected values (Optuna, validation only).",
            "tab:tuning",
        )
    )
    rfp = best["rf_multioutput"]
    rdf = pd.DataFrame(
        [
            ["trees", "60--150", str(rfp["n_estimators"])],
            ["max_depth", "8--30", str(rfp["max_depth"])],
            ["max_features", "0.2 / 0.33 / 0.5", str(rfp["max_features"])],
            ["min_samples_leaf", "1--20", str(rfp["min_samples_leaf"])],
        ],
        columns=["parameter", "search range", "selected"],
    )
    (TAB / "tuning_rf.tex").write_text(
        to_latex(rdf, "Random Forest search space and selected values (6 trials).", "tab:tuningrf")
    )
    ab_path = MET / "ablations.csv"
    if ab_path.exists():
        ab = pd.read_csv(ab_path)
        ab = ab[ab.metric == "MAE"].pivot_table(
            index=["ablation", "variant"], columns="horizon_min", values="mean"
        )
        ref = ab.loc[("A0", "default")]
        t = pd.DataFrame({"setting": [f"{a} {v}".replace("_", " ") for a, v in ab.index]})
        for h in (15, 30, 60):
            t[f"MAE@{h}"] = ab[h].to_numpy()
        for h in (15, 30, 60):
            t[f"d@{h}"] = (ab[h] - ref[h]).to_numpy()
        t = t.sort_values("setting", key=lambda x: x.str.replace(r"^A(\d) ", r"A0\1 ", regex=True))
        (TAB / "ablations.tex").write_text(
            to_latex(
                t,
                "Ablations (1 seed, 12 epochs): MAE (mph) and change vs. default.",
                "tab:abl",
            )
        )
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
