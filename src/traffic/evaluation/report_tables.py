"""CSV -> Markdown / LaTeX (booktabs) tables and README block insertion."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

START, END = "<!-- RESULTS_START -->", "<!-- RESULTS_END -->"


def _fmt(v, digits: int = 2) -> str:
    if isinstance(v, float | np.floating):
        return "-" if np.isnan(v) else f"{v:.{digits}f}"
    return str(v)


def to_markdown(df: pd.DataFrame, digits: int = 2) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(_fmt(r[c], digits) for c in cols) + " |")
    return "\n".join(lines)


def to_latex(
    df: pd.DataFrame, caption: str, label: str, digits: int = 2, bold_min: list[str] | None = None
) -> str:
    """booktabs table; the minimum of each column in ``bold_min`` is bold (best = lowest error)."""
    cols = list(df.columns)
    best = {c: df[c].min() for c in (bold_min or [])}
    body = []
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            s = _fmt(r[c], digits).replace("_", r"\_").replace("%", r"\%")
            if c in best and isinstance(r[c], float | np.floating) and r[c] == best[c]:
                s = rf"\textbf{{{s}}}"
            cells.append(s)
        body.append(" & ".join(cells) + r" \\")
    head = " & ".join(c.replace("_", r"\_") for c in cols) + r" \\"
    spec = "l" + "r" * (len(cols) - 1)
    return (
        "\\begin{table}[t]\n\\centering\n"
        f"\\caption{{{caption}}}\\label{{{label}}}\n"
        "\\footnotesize\\resizebox{\\columnwidth}{!}{%\n"
        f"\\begin{{tabular}}{{{spec}}}\n\\toprule\n{head}\n\\midrule\n"
        + "\n".join(body)
        + "\n\\bottomrule\n\\end{tabular}}\n\\end{table}\n"
    )


def insert_readme_block(readme: Path, markdown: str) -> None:
    text = readme.read_text()
    block = f"{START}\n{markdown}\n{END}"
    new = re.sub(re.escape(START) + r".*?" + re.escape(END), lambda _: block, text, flags=re.S)
    readme.write_text(new)
