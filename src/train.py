"""Train the freight-rate model.

Step 1 (validation design): time-based holdout that mirrors the real task —
fit on Jan-Sep 2025, score on Oct 2025 (the final month of labelled data),
comparing three target formulations. All target encodings are fitted on the
training window only; the only same-period statistic used at predict time is
the *feature-only* same-date mean of market_index for imputing missing
values, which mirrors exactly what is available when scoring validation.csv.

Step 2 (final model): refit the winning formulation on all of Jan-Oct with
the iteration count found in step 1 (scaled for the larger training set),
and persist the model + encoding stats for src/predict.py.

Run: python src/train.py   (writes models/final_model.txt, models/stats.pkl,
models/metrics.json)
"""
from __future__ import annotations

import json
import pickle
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import features as F

# Configuration chosen by the holdout sweep in src/experiments.py
# (config D: lowest October RMSE of five candidates, 657.6 vs 660.2 for
# the initial setting).
LGB_PARAMS = dict(
    objective="regression",
    learning_rate=0.02,
    num_leaves=255,
    min_data_in_leaf=60,
    feature_fraction=0.9,
    bagging_fraction=0.9,
    bagging_freq=5,
    lambda_l2=1.0,
    verbose=-1,
    num_threads=2,
    seed=42,
)


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    mape = float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)
    return {"rmse": rmse, "mae": mae, "r2": float(r2_score(y_true, y_pred)),
            "mape_pct": mape}


def fit_predict(train_df, target_mode, test_X, n_rounds=3000, es_X=None, es_y=None):
    """Fit one LightGBM model under a target formulation; return raw-rate preds."""
    y = train_df["posted_rate"].to_numpy()
    if target_mode == "rate":
        y_fit, inv = y, lambda p: p
    elif target_mode == "log_rate":
        y_fit, inv = np.log1p(y), np.expm1
    elif target_mode == "rpm":
        y_fit = y / train_df["distance"].to_numpy()
        inv = lambda p: p * test_X["distance"].to_numpy()
    else:
        raise ValueError(target_mode)
    dtrain = lgb.Dataset(train_df[F.FEATURES], label=y_fit,
                         categorical_feature=F.CATEGORICAL)
    valid_sets, callbacks = None, []
    if es_X is not None:
        y_es = es_y
        if target_mode == "log_rate":
            y_es = np.log1p(es_y)
        elif target_mode == "rpm":
            y_es = es_y / es_X["distance"].to_numpy()
        dval = lgb.Dataset(es_X[F.FEATURES], label=y_es,
                           categorical_feature=F.CATEGORICAL, reference=dtrain)
        valid_sets = [dval]
        callbacks = [lgb.early_stopping(250, verbose=False)]
    model = lgb.train(LGB_PARAMS, dtrain, num_boost_round=n_rounds,
                      valid_sets=valid_sets, callbacks=callbacks)
    pred = inv(model.predict(test_X[F.FEATURES],
                             num_iteration=model.best_iteration or None))
    return model, np.clip(pred, 10.0, None)


def main() -> None:
    t0 = time.time()
    train = pd.read_csv("data/train_test.csv", parse_dates=["date"])
    hold_mask = train["date"] >= "2025-10-01"
    fit_df, hold_df = train[~hold_mask].copy(), train[hold_mask].copy()
    print(f"holdout design: fit {len(fit_df):,} rows "
          f"({fit_df['date'].min().date()}..{fit_df['date'].max().date()}), "
          f"test {len(hold_df):,} rows (Oct 2025)")

    stats = F.EncodingStats().fit(fit_df)
    fit_X = F.build_features(fit_df, stats, F.date_market_means(fit_df))
    fit_X["posted_rate"] = fit_df["posted_rate"].to_numpy()
    fit_X["distance"] = fit_df["distance"].to_numpy()  # rpm target needs raw distance
    hold_X = F.build_features(hold_df, stats, F.date_market_means(hold_df))
    y_hold = hold_df["posted_rate"].to_numpy()

    results = {}
    preds = {}
    best_iters = {}
    for mode in ("rate", "log_rate", "rpm"):
        model, p = fit_predict(fit_X, mode, hold_X, es_X=hold_X, es_y=y_hold)
        results[mode] = metrics(y_hold, p)
        preds[mode] = p
        best_iters[mode] = int(model.best_iteration or 800)
        print(f"  {mode:9s}  RMSE {results[mode]['rmse']:8.1f}  "
              f"MAE {results[mode]['mae']:8.1f}  R2 {results[mode]['r2']:.4f}  "
              f"MAPE {results[mode]['mape_pct']:.2f}%  (best_iter {best_iters[mode]})")

    ens = (preds["rate"] + preds["rpm"]) / 2
    results["ensemble_rate_rpm"] = metrics(y_hold, ens)
    print(f"  ensemble   RMSE {results['ensemble_rate_rpm']['rmse']:8.1f}  "
          f"MAE {results['ensemble_rate_rpm']['mae']:8.1f}  "
          f"R2 {results['ensemble_rate_rpm']['r2']:.4f}  "
          f"MAPE {results['ensemble_rate_rpm']['mape_pct']:.2f}%")

    winner = min(results, key=lambda k: results[k]["rmse"])
    print("winner by holdout RMSE:", winner)

    # ---- final model on all labelled data --------------------------------
    stats_full = F.EncodingStats().fit(train)
    full_X = F.build_features(train, stats_full, F.date_market_means(train))
    full_X["posted_rate"] = train["posted_rate"].to_numpy()
    full_X["distance"] = train["distance"].to_numpy()
    scale = len(train) / len(fit_df)

    saved = {}
    if winner == "ensemble_rate_rpm":
        modes = ("rate", "rpm")
    else:
        modes = (winner,)
    for mode in modes:
        # Scale the holdout-chosen iteration count to the full training set.
        rounds = int(best_iters[mode] * scale * 1.05)
        y = train["posted_rate"].to_numpy()
        if mode == "rate":
            y_fit = y
        elif mode == "log_rate":
            y_fit = np.log1p(y)
        else:
            y_fit = y / train["distance"].to_numpy()
        dall = lgb.Dataset(full_X[F.FEATURES], label=y_fit,
                           categorical_feature=F.CATEGORICAL)
        model = lgb.train(LGB_PARAMS, dall, num_boost_round=rounds)
        path = f"models/final_model_{mode}.txt"
        model.save_model(path)
        saved[mode] = {"path": path, "rounds": rounds,
                       "holdout_best_iter": best_iters[mode]}
        print(f"final {mode}: {rounds} rounds (holdout best_iter "
              f"{best_iters[mode]}) -> {path}")

    with open("models/stats.pkl", "wb") as fh:
        pickle.dump(stats_full, fh)
    with open("models/metrics.json", "w") as fh:
        json.dump({"holdout": results, "winner": winner, "final_models": saved,
                   "holdout_design": "fit 2025-01-01..2025-09-30, test 2025-10-01..2025-10-31",
                   "n_fit": int(len(fit_df)), "n_holdout": int(len(hold_df))},
                  fh, indent=2)
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
