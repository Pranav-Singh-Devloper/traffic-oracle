# Experiment log

| Time | Run | Git | Config | Command | Metrics |
|---|---|---|---|---|---|
| 2026-10-24 01:25 | baselines | 4a36073+dirty | 64c1268c | `scripts/run_baselines.py` | {"seasonal_fallback_share": 0.12612535213657808} |
| 2026-10-24 01:27 | baselines | 4a36073+dirty | de7d8711 | `scripts/run_baselines.py` | {"seasonal_fallback_share": 0.12612535213657808} |
| 2026-10-24 01:29 | xgb_global | bd99171+dirty | fdbb7f35 | `scripts/train_ml.py --model xgb_global --debug` | {"train_seconds": 1} |
| 2026-10-24 01:29 | xgb_per_segment | bd99171+dirty | bb722e21 | `scripts/train_ml.py --model xgb_per_segment --debug` | {"train_seconds": 3} |
| 2026-10-24 01:30 | rf_multioutput | bd99171+dirty | 78d20159 | `scripts/train_ml.py --model rf --debug` | {"train_seconds": 0} |
| 2026-10-24 01:30 | xgb_spatial | bd99171+dirty | 95415b07 | `scripts/train_ml.py --model xgb_spatial --debug` | {"train_seconds": 1} |
| 2026-10-24 01:30 | xgb_global | bd99171+dirty | fdbb7f35 | `scripts/train_ml.py --model xgb_global --debug` | {"train_seconds": 1} |
| 2026-10-24 01:32 | xgb_global | 294e758+dirty | cd65f821 | `scripts/train_ml.py --model xgb_global --debug --daily-lag` | {"train_seconds": 1} |
| 2026-10-24 01:33 | xgb_spatial | 294e758+dirty | 95415b07 | `scripts/train_ml.py --model xgb_spatial --debug` | {"train_seconds": 2} |
| 2026-10-24 01:35 | stgat_seed0 | 294e758+dirty | c18ea6c9 | `scripts/train_dl.py --model stgat --debug` | {"val_mae": 5.3779, "epochs": 1, "seconds": 7} |
| 2026-10-24 01:35 | gru_nograph_seed0 | 294e758+dirty | dc89e14d | `scripts/train_dl.py --model gru_nograph --debug` | {"val_mae": 5.3093, "epochs": 1, "seconds": 2} |
