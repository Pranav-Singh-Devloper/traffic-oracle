"""Optuna tuning on the VALIDATION split only (never touches test)."""

from __future__ import annotations

import time

import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from traffic.models.ml_models import fit_xgb

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _mae(y, yh) -> float:
    return float(np.abs(y - yh).mean())


def tune_xgb(xtr, ytr, xva, yva, categorical, n_trials: int, seed: int, base: dict, es: int):
    """Returns (best_params, trials_df). Objective: val MAE (mph)."""

    def objective(trial: optuna.Trial) -> float:
        params = {
            **base,
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "n_estimators": 800,
            "objective": trial.suggest_categorical(
                "objective", ["reg:squarederror", "reg:absoluteerror", "reg:pseudohubererror"]
            ),
        }
        t0 = time.time()
        m = fit_xgb(xtr, ytr, xva, yva, params, categorical, es, seed=seed)
        v = _mae(yva, m.predict(xva))
        trial.set_user_attr("seconds", time.time() - t0)
        trial.set_user_attr("best_iteration", int(m.best_iteration))
        return v

    st = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
    st.optimize(objective, n_trials=n_trials)
    df = st.trials_dataframe()
    best = {**base, **st.best_params, "n_estimators": 800}
    return best, df


def tune_rf(xtr, ytr, xva, yva, n_trials: int, seed: int):
    """Multi-output RF tuning; objective = mean val MAE over all horizons."""

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 60, 150),
            "max_depth": trial.suggest_int("max_depth", 8, 30),
            "max_features": trial.suggest_categorical("max_features", [0.2, 0.33, 0.5]),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
            "max_samples": 0.5,
        }
        rf = RandomForestRegressor(n_jobs=-1, random_state=seed, **params).fit(xtr, ytr)
        return _mae(yva, rf.predict(xva))

    st = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
    st.optimize(objective, n_trials=n_trials)
    return {**st.best_params, "max_samples": 0.5}, st.trials_dataframe()


def save_best(path, name: str, params: dict, extra: dict | None = None) -> None:
    import yaml

    cfg = {"inherits": "base.yaml", "model": {"name": name, "params": params, **(extra or {})}}
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))


def trials_to_csv(df: pd.DataFrame, path) -> None:
    df.to_csv(path, index=False)
