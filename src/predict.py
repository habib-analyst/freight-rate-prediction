"""Generate the two submission files with the trained final model(s).

Outputs (repo root):
  * validation_predictions.csv  — 12,000 rows, load_id,predicted_rate
  * december-chart-inputs.csv   — the fixed Lexington -> Fort Wayne lane with
    predicted_rate filled for 2025-12-01..31 (original columns preserved)

December inputs: the chart file fixes every input except the date, so the
date-varying market features come from the December rows of validation.csv
itself (feature-only, per-date means of market_index / quote_signal — the
same information available for any December load). Coordinates for the two
cities come from the training data's fixed per-city coordinates.

Run: python src/predict.py
"""
from __future__ import annotations

import json
import pickle

import lightgbm as lgb
import numpy as np
import pandas as pd

import features as F


def load_models():
    with open("models/metrics.json") as fh:
        meta = json.load(fh)
    with open("models/stats.pkl", "rb") as fh:
        stats = pickle.load(fh)
    models = {mode: lgb.Booster(model_file=info["path"])
              for mode, info in meta["final_models"].items()}
    return meta, stats, models


def predict_frame(models: dict, frame_X: pd.DataFrame) -> np.ndarray:
    preds = []
    for mode, model in models.items():
        p = model.predict(frame_X[F.FEATURES])
        if mode == "log_rate":
            p = np.expm1(p)
        elif mode == "rpm":
            p = p * frame_X["distance"].to_numpy()
        preds.append(p)
    return np.clip(np.mean(preds, axis=0), 10.0, None)


def main() -> None:
    meta, stats, models = load_models()
    valid = pd.read_csv("data/validation.csv", parse_dates=["date"])
    template = pd.read_csv("data/validation_predictions_template.csv")

    valid_X = F.build_features(valid, stats, F.date_market_means(valid))
    valid = valid.assign(pred=predict_frame(models, valid_X))

    out = (template.drop(columns=["predicted_rate"])
           .merge(valid[["load_id", "pred"]], on="load_id", how="left"))
    assert len(out) == 12_000 and out["pred"].notna().all() and (out["pred"] > 0).all()
    out = out.rename(columns={"pred": "predicted_rate"})[["load_id", "predicted_rate"]]
    out["predicted_rate"] = out["predicted_rate"].round(2)
    out.to_csv("validation_predictions.csv", index=False)
    print(f"validation_predictions.csv: {len(out):,} rows, "
          f"rate range ${out['predicted_rate'].min():.0f}-${out['predicted_rate'].max():.0f}")

    # ---- December fixed-lane file ---------------------------------------
    dec = pd.read_csv("data/december_chart_inputs.csv")
    dec["date"] = pd.to_datetime(dec["date"])
    dec_month = valid[valid["date"].dt.month == 12]
    mi_by_date = dec_month.groupby("date")["market_index"].mean()
    qs_by_date = dec_month.groupby("date")["quote_signal"].mean()
    lex = stats.city_coords_["Lexington"]
    ftw = stats.city_coords_["Fort Wayne"]
    rows = dec.copy()
    rows["pickup_lat"], rows["pickup_lon"] = lex["pickup_lat"], lex["pickup_lon"]
    rows["delivery_lat"], rows["delivery_lon"] = ftw["pickup_lat"], ftw["pickup_lon"]
    rows["market_index"] = rows["date"].map(mi_by_date).fillna(
        dec_month["market_index"].mean())
    rows["quote_signal"] = rows["date"].map(qs_by_date).fillna(
        dec_month["quote_signal"].mean())
    rows_X = F.build_features(rows, stats, None)
    dec["predicted_rate"] = predict_frame(models, rows_X).round(2)
    dec["date"] = dec["date"].dt.strftime("%Y-%m-%d")
    dec.to_csv("december-chart-inputs.csv", index=False)
    print("december-chart-inputs.csv: 31 rows, predicted_rate "
          f"${dec['predicted_rate'].min():.0f}-${dec['predicted_rate'].max():.0f}")
    print(dec[["date", "predicted_rate"]].to_string(index=False))


if __name__ == "__main__":
    main()
