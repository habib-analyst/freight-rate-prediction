"""Holdout hyper-parameter sweep for the winning log-rate formulation.

Same split as train.py (fit Jan-Sep 2025, test Oct 2025). Run once during
development; the chosen configuration is hard-coded in train.py.
"""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import features as F

train = pd.read_csv("data/train_test.csv", parse_dates=["date"])
hold = train["date"] >= "2025-10-01"
fit_df, hold_df = train[~hold].copy(), train[hold].copy()
stats = F.EncodingStats().fit(fit_df)
fit_X = F.build_features(fit_df, stats, F.date_market_means(fit_df))
hold_X = F.build_features(hold_df, stats, F.date_market_means(hold_df))
y_fit = np.log1p(fit_df["posted_rate"].to_numpy())
y_hold = hold_df["posted_rate"].to_numpy()

BASE = dict(objective="regression", verbose=-1, num_threads=2, seed=42,
            feature_pre_filter=False)
CONFIGS = {
    "A_lr05_L63 (train.py)": dict(learning_rate=0.05, num_leaves=63, min_data_in_leaf=40,
                                  feature_fraction=0.9, bagging_fraction=0.9, bagging_freq=5, lambda_l2=1.0),
    "B_lr03_L127": dict(learning_rate=0.03, num_leaves=127, min_data_in_leaf=20,
                        feature_fraction=0.9, bagging_fraction=0.9, bagging_freq=5, lambda_l2=1.0),
    "C_lr03_L63_ff08": dict(learning_rate=0.03, num_leaves=63, min_data_in_leaf=40,
                            feature_fraction=0.8, bagging_fraction=0.85, bagging_freq=5, lambda_l2=1.0),
    "D_lr02_L255": dict(learning_rate=0.02, num_leaves=255, min_data_in_leaf=60,
                        feature_fraction=0.9, bagging_fraction=0.9, bagging_freq=5, lambda_l2=1.0),
    "E_lr03_L127_l2_5": dict(learning_rate=0.03, num_leaves=127, min_data_in_leaf=30,
                             feature_fraction=0.85, bagging_fraction=0.9, bagging_freq=5, lambda_l2=5.0),
}

for name, cfg in CONFIGS.items():
    dtrain = lgb.Dataset(fit_X[F.FEATURES], label=y_fit, categorical_feature=F.CATEGORICAL)
    dval = lgb.Dataset(hold_X[F.FEATURES], label=np.log1p(y_hold),
                       categorical_feature=F.CATEGORICAL, reference=dtrain)
    m = lgb.train({**BASE, **cfg}, dtrain, num_boost_round=4000,
                  valid_sets=[dval], callbacks=[lgb.early_stopping(200, verbose=False)])
    p = np.clip(np.expm1(m.predict(hold_X[F.FEATURES], num_iteration=m.best_iteration)), 10, None)
    rmse = np.sqrt(mean_squared_error(y_hold, p))
    print(f"{name:24s} iter {m.best_iteration:5d}  RMSE {rmse:8.1f}  "
          f"MAE {mean_absolute_error(y_hold, p):7.1f}  R2 {r2_score(y_hold, p):.4f}  "
          f"MAPE {np.mean(np.abs((y_hold - p) / y_hold)) * 100:.2f}%")
    if name.startswith("B_"):
        # seed-averaged variant of the strongest-looking config
        ps = [p]
        for seed in (7, 123):
            dtrain = lgb.Dataset(fit_X[F.FEATURES], label=y_fit, categorical_feature=F.CATEGORICAL)
            m2 = lgb.train({**BASE, **cfg, "seed": seed, "bagging_seed": seed,
                            "feature_fraction_seed": seed}, dtrain,
                           num_boost_round=m.best_iteration)
            ps.append(np.clip(np.expm1(m2.predict(hold_X[F.FEATURES])), 10, None))
        pa = np.mean(ps, axis=0)
        print(f"{'B + 3-seed average':24s} iter {m.best_iteration:5d}  "
              f"RMSE {np.sqrt(mean_squared_error(y_hold, pa)):8.1f}  "
              f"MAE {mean_absolute_error(y_hold, pa):7.1f}  "
              f"R2 {r2_score(y_hold, pa):.4f}  "
              f"MAPE {np.mean(np.abs((y_hold - pa) / y_hold)) * 100:.2f}%")
