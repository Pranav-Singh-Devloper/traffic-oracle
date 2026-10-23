# Chronological splits

## Idea
Traffic speed is strongly autocorrelated and we slide windows with stride 1, so neighboring samples share almost all of their inputs and targets. A random train/test split puts near-copies of test samples in training: the model is rewarded for memorising the immediate past, and the measured error is far lower than what it would achieve on genuinely new days.

## Our protocol (strict)
- Split the **timestamp axis**: first 70% train, next 10% validation, last 20% test (`chronological_split`).
- Build windows **inside** each split. Window input and target never cross a boundary, so `max(train target_ts) < min(val input_ts)` and `max(val target_ts) < min(test input_ts)` (tested).
- Everything learned from data (scaler, time-of-day means for imputation, historical-average table, target encodings, hyperparameters) uses train (or val for tuning) only.

## DCRNN protocol (literature-comparable)
DCRNN builds all windows first, then splits the *samples* 70/10/20 (23974/3425/6850 here). Windows at a boundary overlap across splits (up to 23 steps), a small leak. We use it only to compare with published numbers and label it as such.

## Equations
Purged boundary: with history H and horizon F, windows whose span `[t-H+1, t+F]` is wholly inside one split are the only valid ones, so each split loses `H+F-1` anchors.

## Likely viva questions
1. *Why not k-fold cross validation?* It shuffles time; use forward-chaining (expanding window) if CV is needed (we do this for OOF predictions in the hybrid).
2. *Does forward-fill leak?* No: it only looks backwards in time.
3. *Why report the DCRNN protocol at all?* Comparability with published MAE tables; it is labeled.
4. *What distribution shift exists between our splits?* See `docs/eda_summary.md`: the test period has a higher outage rate.
5. *Why can validation still overfit?* We pick hyperparameters and early-stop on it; test is touched once.
