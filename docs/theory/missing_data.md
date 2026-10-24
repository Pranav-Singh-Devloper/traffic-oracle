# Missing data

**Owner:** Kanchan

## Zero is not "no traffic"
In METR-LA a speed of 0 almost always means the loop detector did not report. Treating it as a real speed teaches a model that sensors randomly crash to 0 mph.

## Masking
Targets: never imputed; loss and metrics sum only over observed entries: `L = Σ m·ℓ(y,ŷ) / Σ m`. Gradients come only from real observations, and normalising by Σm keeps the scale independent of the batch's missing rate. The masked loss is an unbiased estimate of the loss over observed entries if missingness is independent of the target given the inputs (an assumption we cannot fully verify: outages cluster in time).

## Inputs
Option (a, default): forward-fill up to 6 steps (30 min) using only the past, then train-only time-of-day means, plus an `is_missing` channel. Option (b, DCRNN): leave as 0 in raw space; we set it to the train mean in scaled space (ADR-003) and rely on the `is_missing` channel. Ablation A9 compares them.

## Facts in this dataset
See `docs/eda_summary.md`: ~8% missing, whole-network outages exist, and the test period has more outages than train.

## Likely viva questions
1. *Why not interpolate the targets?* It would invent data and reward smoothness.
2. *Why a missing flag when imputing?* The model should know the value is not real.
3. *Where could masking bias results?* If outages correlate with congestion.
