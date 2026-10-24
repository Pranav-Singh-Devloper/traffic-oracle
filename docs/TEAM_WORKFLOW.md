# Team T042 workflow: getting real commits from each member

**Why:** the course checks GitHub history for individual contribution. Authorship must be real, so each member commits from their own account. We do not rewrite history or fake authors. This takes about 15 minutes per person.

## One-time setup (each member)
```bash
git clone https://github.com/Pranav-Singh-Devloper/traffic-oracle.git && cd traffic-oracle
git config user.name  "Your Name"
git config user.email "the-email-on-your-github-account"   # must match GitHub or commits are not credited to you
uv venv --python 3.11 .venv && source .venv/bin/activate && uv pip install -e ".[dev]"
make data        # downloads the dataset (needed to run anything)
```
Pranav must add the other four as **collaborators** on the GitHub repo (Settings -> Collaborators).

## See your files
```bash
python scripts/owners.py mine --as <you>     # pranav_singh | kunal | himanshu | kanchan | vaibhav
python scripts/owners.py check               # every file has an owner
```

## Commit only your files, as yourself (the "seamless" part)
After editing any of your files:
```bash
git switch -c <you>/<topic>                                   # e.g. kunal/shap-notes
python scripts/owners.py commit --as <you> "what you did" --dry-run   # preview
python scripts/owners.py commit --as <you> "what you did"
git push -u origin <you>/<topic>                              # then open a pull request on GitHub
```
The script stages **only** your files and commits with **your** git identity. Pranav reviews and merges.

## What each member should genuinely commit (real work, small and quick)
Each item changes only that member's files, so history will show their name on their modules.

| Member | First commits (do these, in your own words) |
|---|---|
| **Pranav Singh** | Re-run `python scripts/train_dl.py --model stgat --debug` and `pytest tests/test_gat_layer.py`; add a short "how I verified PyG equivalence" section to `docs/theory/gat.md`; tune/describe one STGAT setting you can defend. |
| **Kunal** | Run `python scripts/train_ml.py --model xgb_global --debug`; read `ml_models.py` and add comments where you would explain to a grader; extend `docs/theory/xgboost.md` with your own worked example; add one feature test. |
| **Himanshu** | Run `python scripts/evaluate_all.py`; add one hand-computed metric test to `tests/test_metrics_masking.py`; extend `docs/theory/significance_testing.md`; run `python scripts/run_control.py` and add your reading of the results to `docs/control_results.md`. |
| **Kanchan** | Run `python scripts/prepare_data.py` and `make eda`; add one data-integrity test; improve `docs/theory/missing_data.md`; add one more ablation variant to `scripts/run_ablations.py` and run it with `--debug`. |
| **Vaibhav** | Run `make lint test`; fix any README step that fails on your machine; check three `report/refs.bib` entries against the actual papers and mark them; rebuild both PDFs (`make report`). |

Every member should also edit **their own** `docs/viva/<you>.md` (answer each question in your own words) and fill the `@username` lines in `.github/CODEOWNERS`.

## Honest authorship
Most of the initial code and docs were written with an AI assistant and committed from Pranav's account as a baseline import. Do not claim otherwise in the report or viva; check your course's rules on AI-assisted work and disclose accordingly. `python scripts/owners.py stats` shows who has actually committed to which files.
