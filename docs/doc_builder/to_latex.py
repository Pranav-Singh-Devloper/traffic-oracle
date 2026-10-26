"""Convert content.json (the same blocks used for the docx) into a LaTeX document. Usage: to_latex.py content.json out.tex"""
import json, re, sys

blocks = json.load(open(sys.argv[1], encoding="utf8"))

MAP = {  # unicode -> LaTeX (math-safe)
    "±": r"\ensuremath{\pm}", "−": r"\ensuremath{-}", "·": r"\ensuremath{\cdot}", "Σ": r"\ensuremath{\Sigma}",
    "≈": r"\ensuremath{\approx}", "→": r"\ensuremath{\rightarrow}", "ᵀ": r"\ensuremath{^{\top}}", "×": r"\ensuremath{\times}",
    "²": r"\ensuremath{^{2}}", "¹": r"\ensuremath{^{1}}", "ρ": r"\ensuremath{\rho}", "λ": r"\ensuremath{\lambda}",
    "′": r"\ensuremath{'}", "≥": r"\ensuremath{\geq}", "Δ": r"\ensuremath{\Delta}", "σ": r"\ensuremath{\sigma}",
    "…": r"\ldots{}", "‖": r"\ensuremath{\Vert}", "γ": r"\ensuremath{\gamma}", "∞": r"\ensuremath{\infty}",
    "≤": r"\ensuremath{\leq}", "½": r"\ensuremath{\tfrac{1}{2}}", "∂": r"\ensuremath{\partial}", "α": r"\ensuremath{\alpha}",
    "⁻": r"\ensuremath{^{-}}", "π": r"\ensuremath{\pi}", "Ω": r"\ensuremath{\Omega}", "∈": r"\ensuremath{\in}",
    "θ": r"\ensuremath{\theta}", "∧": r"\ensuremath{\wedge}", "∪": r"\ensuremath{\cup}", "β": r"\ensuremath{\beta}",
    "⌈": r"\ensuremath{\lceil}", "⌉": r"\ensuremath{\rceil}", "κ": r"\ensuremath{\kappa}", "ℓ": r"\ensuremath{\ell}",
    "⇒": r"\ensuremath{\Rightarrow}", "√": r"\ensuremath{\sqrt{\ }}", "₁": r"\ensuremath{_{1}}", "₂": r"\ensuremath{_{2}}",
    "ℝ": r"\ensuremath{\mathbb{R}}", "ŷ": r"\ensuremath{\hat{y}}", "̂": "", "̄": "", "–": "--", "—": "---",
    "“": "``", "”": "''",
}
ESC = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_\allowbreak{}",
       "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\^{}", "/": r"/\allowbreak{}"}


def esc(s: str) -> str:
    out = []
    for ch in s:
        out.append(ESC.get(ch) or MAP.get(ch) or ch)
    return "".join(out)


def inline(text: str) -> str:
    parts, last = [], 0
    for m in re.finditer(r"(\*\*[^*]+\*\*|`[^`]+`|\*[^*\s][^*]*\*)", text):
        parts.append(esc(text[last:m.start()]))
        t = m.group(0)
        if t.startswith("**"):
            parts.append(r"\textbf{" + esc(t[2:-2]) + "}")
        elif t.startswith("`"):
            parts.append(r"\texttt{" + esc(t[1:-1]) + "}")
        else:
            parts.append(r"\emph{" + esc(t[1:-1]) + "}")
        last = m.end()
    parts.append(esc(text[last:]))
    return "".join(parts)


def heading(cmd, text):
    return f"\\{cmd}*{{{inline(text)}}}\\addcontentsline{{toc}}{{{ 'section' if cmd=='section' else 'subsection' if cmd=='subsection' else 'subsubsection'}}}{{{inline(text)}}}\n"


o = []
for b in blocks:
    t = b["t"]
    if t == "title":
        o.append("\\begin{titlepage}\\centering\\vspace*{3cm}\n{\\fontsize{40}{44}\\selectfont\\bfseries\\color{accent} " + esc(b["title"]) + "}\\\\[1.2cm]\n"
                 "{\\Large " + esc(b["subtitle"]) + "}\\\\[2cm]\n" + "\\\\[4pt]\n".join("{\\large " + inline(l) + "}" for l in b["lines"]) + "\n\\end{titlepage}\n"
                 "\\setcounter{tocdepth}{2}\\tableofcontents\n\\clearpage\n")
    elif t == "toc":
        pass
    elif t == "h1":
        o.append("\\clearpage\n" if False else "")
        o.append(heading("section", b["text"]))
    elif t == "h2":
        o.append(heading("subsection", b["text"]))
    elif t == "h3":
        o.append(heading("subsubsection", b["text"]))
    elif t == "p":
        o.append(inline(b["text"]) + "\n\n")
    elif t in ("bullets", "numbered"):
        env = "itemize" if t == "bullets" else "enumerate"
        o.append(f"\\begin{{{env}}}[nosep,leftmargin=*]\n" + "".join("\\item " + inline(i) + "\n" for i in b["items"]) + f"\\end{{{env}}}\n\n")
    elif t == "eq":
        o.append("\\begin{eqbox}" + inline(b["text"]) + "\\end{eqbox}\n\n")
    elif t == "callout":
        o.append("\\begin{callout}" + inline(b["text"]) + "\\end{callout}\n\n")
    elif t == "pagebreak":
        o.append("\\clearpage\n")
    elif t == "img":
        w = min(b["width_in"] / 6.3, 1.0)
        o.append("\\begin{figure}[H]\\centering\\includegraphics[width=" + f"{w:.2f}" + "\\linewidth]{" + b["path"] + "}\\\\[2pt]{\\footnotesize\\itshape "
                 + inline(b["caption"]) + "}\\end{figure}\n\n")
    elif t == "table":
        n = len(b["header"])
        rel = b["widths"] if b.get("widths") and len(b["widths"]) == n else [1] * n
        tot = sum(rel)
        usable = 0.97 - 0.0  # fraction of \linewidth after column separation
        cols = "".join(f"p{{{usable * r / tot - 2 * 0.0 :.3f}\\dimexpr\\linewidth-{2 * n}\\tabcolsep\\relax}}".replace("\\dimexpr\\linewidth-" + f"{2*n}" + "\\tabcolsep\\relax", "") if False else
                       ">{\\raggedright\\arraybackslash}p{" + f"{(r / tot):.4f}" + "\\dimexpr\\linewidth-" + f"{2 * n}" + "\\tabcolsep\\relax}" for r in rel)
        size = "\\scriptsize" if b.get("small") else "\\footnotesize"
        hdr = " & ".join("\\textbf{" + inline(h) + "}" for h in b["header"])
        rows = "\n".join(" & ".join(inline(c) for c in r) + " \\\\" for r in b["rows"])
        o.append("{" + size + "\\renewcommand{\\arraystretch}{1.15}\\begin{longtable}{" + cols + "}\n\\toprule\n" + hdr + " \\\\\n\\midrule\n\\endhead\n"
                 "\\bottomrule\n\\endlastfoot\n" + rows + "\n\\end{longtable}}\n{\\footnotesize\\itshape " + inline(b["caption"]) + "}\\par\\medskip\n\n")

pre = r"""\documentclass[10pt,a4paper]{article}
\usepackage[a4paper,margin=2.2cm]{geometry}
\usepackage{fontspec}
\usepackage{amsmath,amssymb,booktabs,longtable,array,graphicx,float,enumitem,xcolor,fancyhdr,tcolorbox}
\usepackage[colorlinks=true,linkcolor=accent,urlcolor=accent]{hyperref}
\definecolor{accent}{HTML}{1F4E79}
\sloppy\emergencystretch=3em
\setlength{\parindent}{0pt}\setlength{\parskip}{5pt}
\usepackage{titlesec}
\titleformat{\section}{\Large\bfseries\color{accent}}{}{0pt}{}
\titleformat{\subsection}{\large\bfseries\color{accent!85!black}}{}{0pt}{}
\titleformat{\subsubsection}{\normalsize\bfseries}{}{0pt}{}
\titlespacing*{\section}{0pt}{16pt}{6pt}\titlespacing*{\subsection}{0pt}{12pt}{4pt}
\pagestyle{fancy}\fancyhf{}\fancyhead[R]{\small\color{gray} Traffic Oracle $\cdot$ Team T042}\fancyfoot[C]{\small\thepage}
\renewcommand{\headrulewidth}{0pt}
\newtcolorbox{callout}{colback=orange!8,colframe=orange!60!black,boxrule=0.4pt,left=6pt,right=6pt,top=4pt,bottom=4pt,fontupper=\small}
\newtcolorbox{eqbox}{colback=black!4,colframe=black!10,boxrule=0.2pt,halign=center,fontupper=\small,left=4pt,right=4pt,top=3pt,bottom=3pt}
\begin{document}
"""
import os
ROOT = "/Users/pranavsingh/Desktop/folder_2/DL_AML_Project"
body = "".join(o).replace(ROOT, "../..")
open(sys.argv[2], "w", encoding="utf8").write(pre + body + "\\end{document}\n")
print("wrote", sys.argv[2], len(o), "chunks")
