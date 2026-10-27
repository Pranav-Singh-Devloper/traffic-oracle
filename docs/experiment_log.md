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
| 2026-10-24 06:16 | baselines | ab8826e+dirty | adf531f3 | `scripts/run_baselines.py --debug` | {"seasonal_fallback_share": 0.22184115523465703} |
| 2026-10-24 06:16 | xgb_per_segment | ab8826e+dirty | 1abf314e | `scripts/train_ml.py --model xgb_per_segment --debug` | {"train_seconds": 1} |
| 2026-10-24 06:16 | xgb_global | ab8826e+dirty | b9fdae4d | `scripts/train_ml.py --model xgb_global --debug` | {"train_seconds": 1} |
| 2026-10-24 06:16 | rf_multioutput | ab8826e+dirty | 5c7e1b9b | `scripts/train_ml.py --model rf --debug` | {"train_seconds": 0} |
| 2026-10-24 06:16 | xgb_spatial | ab8826e+dirty | b9eeeca2 | `scripts/train_ml.py --model xgb_spatial --debug` | {"train_seconds": 1} |
| 2026-10-24 06:16 | gru_nograph_seed0 | ab8826e+dirty | c6b1ca22 | `scripts/train_dl.py --model gru_nograph --debug` | {"val_mae": 8.5053, "epochs": 1, "seconds": 1} |
| 2026-10-24 06:16 | stgat_seed0 | ab8826e+dirty | a25f01cf | `scripts/train_dl.py --model stgat --debug` | {"val_mae": 7.2307, "epochs": 1, "seconds": 1} |
| 2026-10-24 06:17 | baselines | ab8826e+dirty | adf531f3 | `scripts/run_baselines.py --debug` | {"seasonal_fallback_share": 0.22184115523465703} |
| 2026-10-24 06:17 | xgb_per_segment | ab8826e+dirty | 1abf314e | `scripts/train_ml.py --model xgb_per_segment --debug` | {"train_seconds": 1} |
| 2026-10-24 06:17 | xgb_global | ab8826e+dirty | b9fdae4d | `scripts/train_ml.py --model xgb_global --debug` | {"train_seconds": 1} |
| 2026-10-24 06:17 | rf_multioutput | ab8826e+dirty | 5c7e1b9b | `scripts/train_ml.py --model rf --debug` | {"train_seconds": 0} |
| 2026-10-24 06:17 | xgb_spatial | ab8826e+dirty | b9eeeca2 | `scripts/train_ml.py --model xgb_spatial --debug` | {"train_seconds": 1} |
| 2026-10-24 06:17 | gru_nograph_seed0 | ab8826e+dirty | c6b1ca22 | `scripts/train_dl.py --model gru_nograph --debug` | {"val_mae": 8.5053, "epochs": 1, "seconds": 1} |
| 2026-10-24 06:17 | stgat_seed0 | ab8826e+dirty | a25f01cf | `scripts/train_dl.py --model stgat --debug` | {"val_mae": 7.2307, "epochs": 1, "seconds": 1} |
| 2026-10-24 06:23 | abl_default_seed0 | 85c65d3+dirty | 4419d93b | `scripts/train_dl.py --model stgat --name abl_default --out ablations --seeds 0 --set train.max_epochs=12` | {"val_mae": 3.1128, "epochs": 12, "seconds": 509} |
| 2026-10-24 06:29 | abl_A1_identity_seed0 | 85c65d3+dirty | 545cd216 | `scripts/train_dl.py --model stgat --name abl_A1_identity --out ablations --seeds 0 --set train.max_epochs=12 graph.variant=identity` | {"val_mae": 3.1015, "epochs": 12, "seconds": 318} |
| 2026-10-24 06:35 | abl_A2_random_seed0 | 85c65d3+dirty | f6cdd083 | `scripts/train_dl.py --model stgat --name abl_A2_random --out ablations --seeds 0 --set train.max_epochs=12 graph.variant=random` | {"val_mae": 3.1723, "epochs": 12, "seconds": 380} |
| 2026-10-24 06:38 | xgb_global | 85c65d3+dirty | a9318259 | `scripts/train_ml.py --model xgb_global --daily-lag` | {"train_seconds": 1173} |
| 2026-10-24 06:43 | abl_A3_forward_seed0 | 85c65d3+dirty | 4dfe36d7 | `scripts/train_dl.py --model stgat --name abl_A3_forward --out ablations --seeds 0 --set train.max_epochs=12 graph.variant=forward` | {"val_mae": 3.1308, "epochs": 12, "seconds": 417} |
| 2026-10-24 06:49 | abl_A3_backward_seed0 | 85c65d3+dirty | 8b886bec | `scripts/train_dl.py --model stgat --name abl_A3_backward --out ablations --seeds 0 --set train.max_epochs=12 graph.variant=backward` | {"val_mae": 3.135, "epochs": 12, "seconds": 400} |
| 2026-10-24 07:54 | abl_A4_gatv2_seed0 | 85c65d3+dirty | 036c6ecd | `scripts/train_dl.py --model stgat --name abl_A4_gatv2 --out ablations --seeds 0 --set train.max_epochs=12 model.attention=gatv2` | {"val_mae": 3.0871, "epochs": 12, "seconds": 3828} |
| 2026-10-24 07:58 | abl_A4_mean_seed0 | 85c65d3+dirty | 457c0dc4 | `scripts/train_dl.py --model stgat --name abl_A4_mean --out ablations --seeds 0 --set train.max_epochs=12 model.attention=mean` | {"val_mae": 3.1081, "epochs": 12, "seconds": 205} |
| 2026-10-24 08:06 | abl_A5_heads1_seed0 | 85c65d3+dirty | f0314faa | `scripts/train_dl.py --model stgat --name abl_A5_heads1 --out ablations --seeds 0 --set train.max_epochs=12 model.heads=1` | {"val_mae": 3.0983, "epochs": 12, "seconds": 450} |
| 2026-10-24 08:14 | abl_A5_heads2_seed0 | 85c65d3+dirty | 9aed0853 | `scripts/train_dl.py --model stgat --name abl_A5_heads2 --out ablations --seeds 0 --set train.max_epochs=12 model.heads=2` | {"val_mae": 3.0921, "epochs": 12, "seconds": 474} |
| 2026-10-24 08:23 | abl_A5_heads8_seed0 | 85c65d3+dirty | c2d68763 | `scripts/train_dl.py --model stgat --name abl_A5_heads8 --out ablations --seeds 0 --set train.max_epochs=12 model.heads=8` | {"val_mae": 3.123, "epochs": 12, "seconds": 536} |
| 2026-10-24 08:28 | abl_A6_layers1_seed0 | 85c65d3+dirty | 78125f68 | `scripts/train_dl.py --model stgat --name abl_A6_layers1 --out ablations --seeds 0 --set train.max_epochs=12 model.layers=1` | {"val_mae": 3.1446, "epochs": 12, "seconds": 257} |
| 2026-10-24 08:40 | abl_A6_layers3_seed0 | 85c65d3+dirty | de3d2eac | `scripts/train_dl.py --model stgat --name abl_A6_layers3 --out ablations --seeds 0 --set train.max_epochs=12 model.layers=3` | {"val_mae": 3.1094, "epochs": 12, "seconds": 728} |
| 2026-10-24 08:48 | abl_A7_no_node_emb_seed0 | 85c65d3+dirty | ac60c322 | `scripts/train_dl.py --model stgat --name abl_A7_no_node_emb --out ablations --seeds 0 --set train.max_epochs=12 model.node_embedding=false` | {"val_mae": 3.3683, "epochs": 12, "seconds": 486} |
| 2026-10-24 08:57 | abl_A8_no_time_seed0 | 85c65d3+dirty | 71a5f153 | `scripts/train_dl.py --model stgat --name abl_A8_no_time --out ablations --seeds 0 --set train.max_epochs=12 model.use_time_features=false` | {"val_mae": 3.3262, "epochs": 12, "seconds": 490} |
| 2026-10-24 09:05 | abl_A9_zero_mask_seed0 | 85c65d3+dirty | 7c6f5609 | `scripts/train_dl.py --model stgat --name abl_A9_zero_mask --out ablations --seeds 0 --set train.max_epochs=12 input_imputation=zero_mask` | {"val_mae": 3.1106, "epochs": 12, "seconds": 493} |
| 2026-10-24 09:14 | abl_A10_no_dropout_seed0 | 85c65d3+dirty | f377db3a | `scripts/train_dl.py --model stgat --name abl_A10_no_dropout --out ablations --seeds 0 --set train.max_epochs=12 model.dropout=0 model.attn_dropout=0` | {"val_mae": 3.0696, "epochs": 12, "seconds": 485} |
| 2026-10-24 09:22 | abl_A10_no_wd_seed0 | 85c65d3+dirty | 0e378a7d | `scripts/train_dl.py --model stgat --name abl_A10_no_wd --out ablations --seeds 0 --set train.max_epochs=12 train.weight_decay=0` | {"val_mae": 3.0248, "epochs": 12, "seconds": 493} |
| 2026-10-24 09:30 | abl_A11_H6_seed0 | 85c65d3+dirty | 4f12795e | `scripts/train_dl.py --model stgat --name abl_A11_H6 --out ablations --seeds 0 --set train.max_epochs=12 history_steps=6` | {"val_mae": 3.1095, "epochs": 12, "seconds": 453} |
| 2026-10-24 09:40 | abl_A11_H24_seed0 | 85c65d3+dirty | 90a99173 | `scripts/train_dl.py --model stgat --name abl_A11_H24 --out ablations --seeds 0 --set train.max_epochs=12 history_steps=24` | {"val_mae": 3.089, "epochs": 12, "seconds": 576} |
| 2026-10-24 09:48 | abl_A12_edge_bias_seed0 | 85c65d3+dirty | 15ee77ba | `scripts/train_dl.py --model stgat --name abl_A12_edge_bias --out ablations --seeds 0 --set train.max_epochs=12 model.edge_bias=true` | {"val_mae": 3.0839, "epochs": 12, "seconds": 497} |
| 2026-10-24 09:51 | baselines | 85c65d3+dirty | adf531f3 | `scripts/run_baselines.py --debug` | {"seasonal_fallback_share": 0.22184115523465703} |
| 2026-10-24 09:52 | xgb_per_segment | 85c65d3+dirty | 1abf314e | `scripts/train_ml.py --model xgb_per_segment --debug` | {"train_seconds": 1} |
| 2026-10-24 09:52 | xgb_global | 85c65d3+dirty | b9fdae4d | `scripts/train_ml.py --model xgb_global --debug` | {"train_seconds": 1} |
| 2026-10-24 09:52 | rf_multioutput | 85c65d3+dirty | 5c7e1b9b | `scripts/train_ml.py --model rf --debug` | {"train_seconds": 0} |
| 2026-10-24 09:52 | xgb_spatial | 85c65d3+dirty | b9eeeca2 | `scripts/train_ml.py --model xgb_spatial --debug` | {"train_seconds": 1} |
| 2026-10-24 09:52 | gru_nograph_seed0 | 85c65d3+dirty | c6b1ca22 | `scripts/train_dl.py --model gru_nograph --debug` | {"val_mae": 8.5053, "epochs": 1, "seconds": 1} |
| 2026-10-24 09:52 | stgat_seed0 | 85c65d3+dirty | a25f01cf | `scripts/train_dl.py --model stgat --debug` | {"val_mae": 7.2307, "epochs": 1, "seconds": 1} |
| 2026-10-24 11:57 | baselines | d7a9c37+dirty | adf531f3 | `scripts/run_baselines.py --debug` | {"seasonal_fallback_share": 0.22184115523465703} |
| 2026-10-24 11:57 | xgb_per_segment | d7a9c37+dirty | 1abf314e | `scripts/train_ml.py --model xgb_per_segment --debug` | {"train_seconds": 1} |
| 2026-10-24 11:57 | xgb_global | d7a9c37+dirty | b9fdae4d | `scripts/train_ml.py --model xgb_global --debug` | {"train_seconds": 1} |
| 2026-10-24 11:57 | rf_multioutput | d7a9c37+dirty | 5c7e1b9b | `scripts/train_ml.py --model rf --debug` | {"train_seconds": 0} |
| 2026-10-24 11:57 | xgb_spatial | d7a9c37+dirty | b9eeeca2 | `scripts/train_ml.py --model xgb_spatial --debug` | {"train_seconds": 1} |
| 2026-10-24 11:57 | gru_nograph_seed0 | d7a9c37+dirty | c6b1ca22 | `scripts/train_dl.py --model gru_nograph --debug` | {"val_mae": 8.5053, "epochs": 1, "seconds": 2} |
| 2026-10-24 11:57 | stgat_seed0 | d7a9c37+dirty | a25f01cf | `scripts/train_dl.py --model stgat --debug` | {"val_mae": 7.2307, "epochs": 1, "seconds": 1} |
| 2026-10-24 11:59 | baselines | 7704e24+dirty | adf531f3 | `scripts/run_baselines.py --debug` | {"seasonal_fallback_share": 0.22184115523465703} |
| 2026-10-24 11:59 | xgb_per_segment | 7704e24+dirty | 1abf314e | `scripts/train_ml.py --model xgb_per_segment --debug` | {"train_seconds": 1} |
| 2026-10-24 11:59 | xgb_global | 7704e24+dirty | b9fdae4d | `scripts/train_ml.py --model xgb_global --debug` | {"train_seconds": 1} |
| 2026-10-24 11:59 | rf_multioutput | 7704e24+dirty | 5c7e1b9b | `scripts/train_ml.py --model rf --debug` | {"train_seconds": 0} |
| 2026-10-24 11:59 | xgb_spatial | 7704e24+dirty | b9eeeca2 | `scripts/train_ml.py --model xgb_spatial --debug` | {"train_seconds": 1} |
| 2026-10-24 11:59 | gru_nograph_seed0 | 7704e24+dirty | c6b1ca22 | `scripts/train_dl.py --model gru_nograph --debug` | {"val_mae": 8.5053, "epochs": 1, "seconds": 1} |
| 2026-10-24 11:59 | stgat_seed0 | 7704e24+dirty | a25f01cf | `scripts/train_dl.py --model stgat --debug` | {"val_mae": 7.2307, "epochs": 1, "seconds": 1} |
| 2026-10-24 12:01 | baselines | bf2fbc5+dirty | adf531f3 | `scripts/run_baselines.py --debug` | {"seasonal_fallback_share": 0.22184115523465703} |
| 2026-10-24 12:01 | xgb_per_segment | bf2fbc5+dirty | 1abf314e | `scripts/train_ml.py --model xgb_per_segment --debug` | {"train_seconds": 1} |
| 2026-10-24 12:01 | xgb_global | bf2fbc5+dirty | b9fdae4d | `scripts/train_ml.py --model xgb_global --debug` | {"train_seconds": 1} |
| 2026-10-24 12:01 | rf_multioutput | bf2fbc5+dirty | 5c7e1b9b | `scripts/train_ml.py --model rf --debug` | {"train_seconds": 0} |
| 2026-10-24 12:02 | xgb_spatial | bf2fbc5+dirty | b9eeeca2 | `scripts/train_ml.py --model xgb_spatial --debug` | {"train_seconds": 1} |
| 2026-10-24 12:02 | gru_nograph_seed0 | bf2fbc5+dirty | c6b1ca22 | `scripts/train_dl.py --model gru_nograph --debug` | {"val_mae": 8.5053, "epochs": 1, "seconds": 1} |
| 2026-10-24 12:02 | stgat_seed0 | bf2fbc5+dirty | a25f01cf | `scripts/train_dl.py --model stgat --debug` | {"val_mae": 7.2307, "epochs": 1, "seconds": 1} |
| 2026-10-24 12:17 | baselines | 6b57be2+dirty | adf531f3 | `scripts/run_baselines.py --debug` | {"seasonal_fallback_share": 0.22184115523465703} |
| 2026-10-24 12:17 | xgb_per_segment | 6b57be2+dirty | 1abf314e | `scripts/train_ml.py --model xgb_per_segment --debug` | {"train_seconds": 1} |
| 2026-10-24 12:17 | xgb_global | 6b57be2+dirty | b9fdae4d | `scripts/train_ml.py --model xgb_global --debug` | {"train_seconds": 1} |
| 2026-10-24 12:17 | rf_multioutput | 6b57be2+dirty | 5c7e1b9b | `scripts/train_ml.py --model rf --debug` | {"train_seconds": 0} |
| 2026-10-24 12:17 | xgb_spatial | 6b57be2+dirty | b9eeeca2 | `scripts/train_ml.py --model xgb_spatial --debug` | {"train_seconds": 1} |
| 2026-10-24 12:17 | gru_nograph_seed0 | 6b57be2+dirty | c6b1ca22 | `scripts/train_dl.py --model gru_nograph --debug` | {"val_mae": 8.5053, "epochs": 1, "seconds": 2} |
| 2026-10-24 12:17 | stgat_seed0 | 6b57be2+dirty | a25f01cf | `scripts/train_dl.py --model stgat --debug` | {"val_mae": 7.2307, "epochs": 1, "seconds": 1} |
| 2026-10-24 13:21 | baselines | 68ed698+dirty | adf531f3 | `scripts/run_baselines.py --debug` | {"seasonal_fallback_share": 0.22184115523465703} |
| 2026-10-24 13:21 | xgb_per_segment | 68ed698+dirty | 1abf314e | `scripts/train_ml.py --model xgb_per_segment --debug` | {"train_seconds": 1} |
| 2026-10-24 13:21 | xgb_global | 68ed698+dirty | b9fdae4d | `scripts/train_ml.py --model xgb_global --debug` | {"train_seconds": 1} |
| 2026-10-24 13:21 | rf_multioutput | 68ed698+dirty | 5c7e1b9b | `scripts/train_ml.py --model rf --debug` | {"train_seconds": 0} |
| 2026-10-24 13:21 | xgb_spatial | 68ed698+dirty | b9eeeca2 | `scripts/train_ml.py --model xgb_spatial --debug` | {"train_seconds": 1} |
| 2026-10-24 13:21 | gru_nograph_seed0 | 68ed698+dirty | c6b1ca22 | `scripts/train_dl.py --model gru_nograph --debug` | {"val_mae": 8.5053, "epochs": 1, "seconds": 2} |
| 2026-10-24 13:21 | stgat_seed0 | 68ed698+dirty | a25f01cf | `scripts/train_dl.py --model stgat --debug` | {"val_mae": 7.2307, "epochs": 1, "seconds": 1} |
| 2026-10-24 13:26 | baselines | 81281ac+dirty | adf531f3 | `scripts/run_baselines.py --debug` | {"seasonal_fallback_share": 0.22184115523465703} |
| 2026-10-24 13:26 | xgb_per_segment | 81281ac+dirty | 1abf314e | `scripts/train_ml.py --model xgb_per_segment --debug` | {"train_seconds": 1} |
| 2026-10-24 13:26 | xgb_global | 81281ac+dirty | b9fdae4d | `scripts/train_ml.py --model xgb_global --debug` | {"train_seconds": 1} |
| 2026-10-24 13:26 | rf_multioutput | 81281ac+dirty | 5c7e1b9b | `scripts/train_ml.py --model rf --debug` | {"train_seconds": 0} |
| 2026-10-24 13:26 | xgb_spatial | 81281ac+dirty | b9eeeca2 | `scripts/train_ml.py --model xgb_spatial --debug` | {"train_seconds": 1} |
| 2026-10-24 13:26 | gru_nograph_seed0 | 81281ac+dirty | c6b1ca22 | `scripts/train_dl.py --model gru_nograph --debug` | {"val_mae": 8.5053, "epochs": 1, "seconds": 2} |
| 2026-10-24 13:26 | stgat_seed0 | 81281ac+dirty | a25f01cf | `scripts/train_dl.py --model stgat --debug` | {"val_mae": 7.2307, "epochs": 1, "seconds": 1} |
| 2026-10-27 14:02 | baselines | 3d48af1+dirty | adf531f3 | `scripts/run_baselines.py --debug` | {"seasonal_fallback_share": 0.22184115523465703} |
| 2026-10-27 14:02 | xgb_per_segment | 3d48af1+dirty | 1abf314e | `scripts/train_ml.py --model xgb_per_segment --debug` | {"train_seconds": 1} |
| 2026-10-27 14:02 | xgb_global | 3d48af1+dirty | b9fdae4d | `scripts/train_ml.py --model xgb_global --debug` | {"train_seconds": 1} |
| 2026-10-27 14:02 | rf_multioutput | 3d48af1+dirty | 5c7e1b9b | `scripts/train_ml.py --model rf --debug` | {"train_seconds": 0} |
| 2026-10-27 14:02 | xgb_spatial | 3d48af1+dirty | b9eeeca2 | `scripts/train_ml.py --model xgb_spatial --debug` | {"train_seconds": 1} |
| 2026-10-27 14:02 | gru_nograph_seed0 | 3d48af1+dirty | c6b1ca22 | `scripts/train_dl.py --model gru_nograph --debug` | {"val_mae": 8.5053, "epochs": 1, "seconds": 2} |
| 2026-10-27 14:02 | stgat_seed0 | 3d48af1+dirty | a25f01cf | `scripts/train_dl.py --model stgat --debug` | {"val_mae": 7.2307, "epochs": 1, "seconds": 1} |
| 2026-10-27 14:02 | baselines | 52518a0+dirty | adf531f3 | `scripts/run_baselines.py --debug` | {"seasonal_fallback_share": 0.22184115523465703} |
| 2026-10-27 14:02 | xgb_per_segment | 52518a0+dirty | 1abf314e | `scripts/train_ml.py --model xgb_per_segment --debug` | {"train_seconds": 1} |
| 2026-10-27 14:02 | xgb_global | 52518a0+dirty | b9fdae4d | `scripts/train_ml.py --model xgb_global --debug` | {"train_seconds": 1} |
| 2026-10-27 14:02 | rf_multioutput | 52518a0+dirty | 5c7e1b9b | `scripts/train_ml.py --model rf --debug` | {"train_seconds": 0} |
| 2026-10-27 14:02 | xgb_spatial | 52518a0+dirty | b9eeeca2 | `scripts/train_ml.py --model xgb_spatial --debug` | {"train_seconds": 1} |
| 2026-10-27 14:02 | gru_nograph_seed0 | 52518a0+dirty | c6b1ca22 | `scripts/train_dl.py --model gru_nograph --debug` | {"val_mae": 8.5053, "epochs": 1, "seconds": 1} |
| 2026-10-27 14:02 | stgat_seed0 | 52518a0+dirty | a25f01cf | `scripts/train_dl.py --model stgat --debug` | {"val_mae": 7.2307, "epochs": 1, "seconds": 1} |
| 2026-10-27 14:16 | baselines | a10281f+dirty | adf531f3 | `scripts/run_baselines.py --debug` | {"seasonal_fallback_share": 0.22184115523465703} |
| 2026-10-27 14:16 | xgb_per_segment | a10281f+dirty | 1abf314e | `scripts/train_ml.py --model xgb_per_segment --debug` | {"train_seconds": 1} |
| 2026-10-27 14:16 | xgb_global | a10281f+dirty | b9fdae4d | `scripts/train_ml.py --model xgb_global --debug` | {"train_seconds": 1} |
| 2026-10-27 14:16 | rf_multioutput | a10281f+dirty | 5c7e1b9b | `scripts/train_ml.py --model rf --debug` | {"train_seconds": 0} |
| 2026-10-27 14:16 | xgb_spatial | a10281f+dirty | b9eeeca2 | `scripts/train_ml.py --model xgb_spatial --debug` | {"train_seconds": 1} |
| 2026-10-27 14:16 | gru_nograph_seed0 | a10281f+dirty | c6b1ca22 | `scripts/train_dl.py --model gru_nograph --debug` | {"val_mae": 8.5053, "epochs": 1, "seconds": 2} |
| 2026-10-27 14:16 | stgat_seed0 | a10281f+dirty | a25f01cf | `scripts/train_dl.py --model stgat --debug` | {"val_mae": 7.2307, "epochs": 1, "seconds": 1} |
| 2026-10-27 14:25 | baselines | 0cefec6+dirty | adf531f3 | `scripts/run_baselines.py --debug` | {"seasonal_fallback_share": 0.22184115523465703} |
| 2026-10-27 14:25 | xgb_per_segment | 0cefec6+dirty | 1abf314e | `scripts/train_ml.py --model xgb_per_segment --debug` | {"train_seconds": 1} |
| 2026-10-27 14:25 | xgb_global | 0cefec6+dirty | b9fdae4d | `scripts/train_ml.py --model xgb_global --debug` | {"train_seconds": 1} |
| 2026-10-27 14:25 | rf_multioutput | 0cefec6+dirty | 5c7e1b9b | `scripts/train_ml.py --model rf --debug` | {"train_seconds": 0} |
| 2026-10-27 14:25 | xgb_spatial | 0cefec6+dirty | b9eeeca2 | `scripts/train_ml.py --model xgb_spatial --debug` | {"train_seconds": 1} |
| 2026-10-27 14:25 | gru_nograph_seed0 | 0cefec6+dirty | c6b1ca22 | `scripts/train_dl.py --model gru_nograph --debug` | {"val_mae": 8.5053, "epochs": 1, "seconds": 3} |
| 2026-10-27 14:25 | stgat_seed0 | 0cefec6+dirty | a25f01cf | `scripts/train_dl.py --model stgat --debug` | {"val_mae": 7.2307, "epochs": 1, "seconds": 1} |
