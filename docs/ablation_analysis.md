# Ablation analysis (auto-generated table; reduced budget)

One seed, 12 epochs, same budget for the default (A0). Differences below about 0.05 mph at 15/30 minutes are within seed noise of the full runs.

| ablation | variant | 15 | 30 | 60 | d15 | d30 | d60 |
|---|---|---|---|---|---|---|---|
| A0 | default | 2.984 | 3.407 | 3.955 | 0.000 | 0.000 | 0.000 |
| A1 | identity | 2.954 | 3.372 | 3.820 | -0.030 | -0.035 | -0.135 |
| A10 | no_dropout | 2.947 | 3.371 | 3.866 | -0.037 | -0.036 | -0.089 |
| A10 | no_wd | 2.929 | 3.310 | 3.731 | -0.055 | -0.097 | -0.224 |
| A11 | H24 | 2.978 | 3.396 | 3.895 | -0.006 | -0.011 | -0.060 |
| A11 | H6 | 2.975 | 3.397 | 3.934 | -0.009 | -0.010 | -0.021 |
| A12 | edge_bias | 2.951 | 3.372 | 3.899 | -0.033 | -0.035 | -0.056 |
| A2 | random | 3.027 | 3.477 | 4.043 | 0.043 | 0.069 | 0.088 |
| A3 | backward | 2.995 | 3.416 | 3.956 | 0.011 | 0.009 | 0.001 |
| A3 | forward | 2.998 | 3.421 | 3.942 | 0.014 | 0.014 | -0.013 |
| A4 | gatv2 | 2.953 | 3.376 | 3.934 | -0.031 | -0.031 | -0.021 |
| A4 | mean | 2.957 | 3.385 | 3.893 | -0.027 | -0.023 | -0.062 |
| A5 | heads1 | 2.967 | 3.385 | 3.895 | -0.017 | -0.022 | -0.060 |
| A5 | heads2 | 2.965 | 3.381 | 3.919 | -0.019 | -0.026 | -0.036 |
| A5 | heads8 | 2.974 | 3.403 | 3.946 | -0.010 | -0.004 | -0.009 |
| A6 | layers1 | 2.992 | 3.437 | 4.024 | 0.008 | 0.030 | 0.069 |
| A6 | layers3 | 2.976 | 3.408 | 3.963 | -0.008 | 0.001 | 0.008 |
| A7 | no_node_emb | 3.132 | 3.728 | 4.581 | 0.149 | 0.321 | 0.626 |
| A8 | no_time | 3.077 | 3.685 | 4.643 | 0.093 | 0.278 | 0.688 |
| A9 | zero_mask | 3.002 | 3.425 | 3.964 | 0.018 | 0.017 | 0.009 |

**Reading.** See the Ablations section of `report/phase2/main.tex`. In short: under this budget the road graph is not shown to beat the identity graph (A1), a random graph is worse than both (A2), the node embedding (A7) and calendar features (A8) matter most, and everything else is within noise. A10 improving without regularisation reflects under-training at 12 epochs. A13 (hybrid design) is the residual/gate/average comparison in the main table.
