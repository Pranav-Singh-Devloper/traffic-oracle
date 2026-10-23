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
