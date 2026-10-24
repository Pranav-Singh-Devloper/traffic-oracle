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
| 2026-10-24 01:42 | xgb_global | 65ad767+dirty | 132622dd | `scripts/train_ml.py --model xgb_global --debug` | {"train_seconds": 1} |
| 2026-10-24 01:57 | xgb_global | 17605f9+dirty | 8642c8ca | `scripts/train_ml.py --model xgb_global` | {"train_seconds": 755} |
| 2026-10-24 02:22 | xgb_spatial | 17605f9+dirty | 185233ac | `scripts/train_ml.py --model xgb_spatial` | {"train_seconds": 1252} |
| 2026-10-24 02:22 | stgat_seed0 | 17605f9+dirty | 41b34d4c | `scripts/train_dl.py --model stgat --debug` | {"val_mae": 5.3779, "epochs": 1, "seconds": 3} |
| 2026-10-24 02:25 | abl_default_seed0 | 77d240b+dirty | 41b34d4c | `scripts/train_dl.py --model stgat --name abl_default --out ablations --seeds 0 --set train.max_epochs=25 --debug` | {"val_mae": 5.3779, "epochs": 1, "seconds": 4} |
| 2026-10-24 02:25 | abl_A4_gatv2_seed0 | 77d240b+dirty | c40983b6 | `scripts/train_dl.py --model stgat --name abl_A4_gatv2 --out ablations --seeds 0 --set train.max_epochs=25 model.attention=gatv2 --debug` | {"val_mae": 5.1551, "epochs": 1, "seconds": 3} |
| 2026-10-24 02:25 | abl_A4_mean_seed0 | 77d240b+dirty | 909f93ec | `scripts/train_dl.py --model stgat --name abl_A4_mean --out ablations --seeds 0 --set train.max_epochs=25 model.attention=mean --debug` | {"val_mae": 5.308, "epochs": 1, "seconds": 2} |
| 2026-10-24 02:25 | abl_A8_no_time_seed0 | 77d240b+dirty | 64d9e33b | `scripts/train_dl.py --model stgat --name abl_A8_no_time --out ablations --seeds 0 --set train.max_epochs=25 model.use_time_features=false --debug` | {"val_mae": 4.8712, "epochs": 1, "seconds": 2} |
| 2026-10-24 02:25 | abl_A9_zero_mask_seed0 | 77d240b+dirty | e642b69d | `scripts/train_dl.py --model stgat --name abl_A9_zero_mask --out ablations --seeds 0 --set train.max_epochs=25 input_imputation=zero_mask --debug` | {"val_mae": 5.4643, "epochs": 1, "seconds": 2} |
| 2026-10-24 02:25 | abl_A11_H6_seed0 | 77d240b+dirty | 02cef8d5 | `scripts/train_dl.py --model stgat --name abl_A11_H6 --out ablations --seeds 0 --set train.max_epochs=25 history_steps=6 --debug` | {"val_mae": 5.3718, "epochs": 1, "seconds": 3} |
| 2026-10-24 02:25 | abl_A11_H24_seed0 | 77d240b+dirty | f5dee4c9 | `scripts/train_dl.py --model stgat --name abl_A11_H24 --out ablations --seeds 0 --set train.max_epochs=25 history_steps=24 --debug` | {"val_mae": 5.657, "epochs": 1, "seconds": 2} |
| 2026-10-24 02:25 | abl_A3_forward_seed0 | 77d240b+dirty | 952ca7f7 | `scripts/train_dl.py --model stgat --name abl_A3_forward --out ablations --seeds 0 --set train.max_epochs=25 graph.variant=forward --debug` | {"val_mae": 5.4698, "epochs": 1, "seconds": 2} |
| 2026-10-24 02:25 | abl_A3_backward_seed0 | 77d240b+dirty | 843a8071 | `scripts/train_dl.py --model stgat --name abl_A3_backward --out ablations --seeds 0 --set train.max_epochs=25 graph.variant=backward --debug` | {"val_mae": 5.36, "epochs": 1, "seconds": 2} |
| 2026-10-24 02:27 | xgb_per_segment | c47669b+dirty | ec42d1cb | `scripts/train_ml.py --model xgb_per_segment` | {"train_seconds": 198} |
| 2026-10-24 02:29 | rf_multioutput | c47669b+dirty | a1424194 | `scripts/train_ml.py --model rf` | {"train_seconds": 56} |
| 2026-10-24 02:57 | xgb_global | 01d73b7+dirty | b9fdae4d | `scripts/train_ml.py --model xgb_global --debug` | {"train_seconds": 1} |
| 2026-10-24 02:57 | stgat_seed0 | 01d73b7+dirty | a25f01cf | `scripts/train_dl.py --model stgat --debug` | {"val_mae": 7.2307, "epochs": 1, "seconds": 5} |
| 2026-10-24 02:57 | hybrid_residual_seed0 | 01d73b7+dirty | cf66c754 | `scripts/train_hybrid.py --kind residual --debug` | {"val_mae": 6.033} |
| 2026-10-24 03:03 | gru_nograph_seed0 | 01d73b7+dirty | 02b75d35 | `scripts/train_dl.py --model gru_nograph --seeds 0 1 2` | {"val_mae": 3.0136, "epochs": 40, "seconds": 490} |
| 2026-10-24 03:11 | gru_nograph_seed1 | 01d73b7+dirty | 02b75d35 | `scripts/train_dl.py --model gru_nograph --seeds 0 1 2` | {"val_mae": 3.0088, "epochs": 40, "seconds": 496} |
| 2026-10-24 03:18 | gru_nograph_seed2 | 01d73b7+dirty | 02b75d35 | `scripts/train_dl.py --model gru_nograph --seeds 0 1 2` | {"val_mae": 3.0162, "epochs": 33, "seconds": 413} |
| 2026-10-24 03:47 | stgat_seed0 | 01d73b7+dirty | 5a846db6 | `scripts/train_dl.py --model stgat --seeds 0 1 2` | {"val_mae": 2.9817, "epochs": 40, "seconds": 1692} |
| 2026-10-24 04:15 | stgat_seed1 | 01d73b7+dirty | 5a846db6 | `scripts/train_dl.py --model stgat --seeds 0 1 2` | {"val_mae": 2.9947, "epochs": 40, "seconds": 1685} |
| 2026-10-24 04:44 | stgat_seed2 | 01d73b7+dirty | 5a846db6 | `scripts/train_dl.py --model stgat --seeds 0 1 2` | {"val_mae": 2.9802, "epochs": 40, "seconds": 1687} |
| 2026-10-24 05:24 | hybrid_residual_seed0 | 01d73b7+dirty | 61765cd9 | `scripts/train_hybrid.py --kind residual --seeds 0 1 2` | {"val_mae": 2.9312} |
| 2026-10-24 05:48 | hybrid_residual_seed1 | 01d73b7+dirty | 61765cd9 | `scripts/train_hybrid.py --kind residual --seeds 0 1 2` | {"val_mae": 2.9332} |
| 2026-10-24 06:11 | hybrid_residual_seed2 | 01d73b7+dirty | 61765cd9 | `scripts/train_hybrid.py --kind residual --seeds 0 1 2` | {"val_mae": 2.9335} |
