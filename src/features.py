"""Shared feature engineering for the Freight Rate Prediction Challenge.

Design notes
------------
* All target encodings are *rate-per-mile* (rpm) means with count-based
  shrinkage toward the global mean. They are always fitted on a past-only
  frame (see ``EncodingStats.fit`` usage in train.py / predict.py), so no
  future information leaks into a row's features.
* Missing ``weight`` is imputed with the equipment-group median; missing
  ``market_index`` with the same-date mean. Both get missingness flags.
* Coordinates are internally consistent in this dataset (one fixed point
  per city; haversine vs. ``distance`` correlation is 0.9995), so the raw
  coordinates plus a haversine feature are safe to use.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

EARTH_R_MI = 3958.8
GLOBAL_FALLBACK_RPM = 2.15  # replaced by the fitted global mean at runtime

CATEGORICAL = ["pickup", "delivery", "equipment"]


def haversine_mi(lat1, lon1, lat2, lon2) -> np.ndarray:
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    return 2 * EARTH_R_MI * np.arcsin(
        np.sqrt(np.sin((lat2 - lat1) / 2) ** 2
                + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    )


class EncodingStats:
    """Rate-per-mile target encodings fitted on one (past-only) frame."""

    def __init__(self, lane_k: float = 20.0, city_k: float = 30.0):
        self.lane_k = lane_k
        self.city_k = city_k

    @staticmethod
    def _shrunk(frame: pd.DataFrame, keys: list[str], k: float, global_rpm: float):
        g = (frame.groupby(keys)
               .agg(sum_rpm=("rpm", "sum"), n=("rpm", "size"))
               .reset_index())
        g["te"] = (g["sum_rpm"] + k * global_rpm) / (g["n"] + k)
        return g

    def fit(self, frame: pd.DataFrame) -> "EncodingStats":
        f = frame.copy()
        f["rpm"] = f["posted_rate"] / f["distance"]
        self.global_rpm_ = float(f["rpm"].mean())
        self.lane_ = self._shrunk(f, ["pickup", "delivery"], self.lane_k, self.global_rpm_)
        self.pickup_ = self._shrunk(f, ["pickup"], self.city_k, self.global_rpm_)
        self.delivery_ = self._shrunk(f, ["delivery"], self.city_k, self.global_rpm_)
        self.lane_counts_ = (f.groupby(["pickup", "delivery"]).size()
                             .rename("lane_n").reset_index())
        self.weight_median_by_equip_ = f.groupby("equipment")["weight"].median().to_dict()
        self.weight_median_ = float(f["weight"].median())
        self.market_index_mean_ = float(f["market_index"].mean())
        self.city_coords_ = (f.groupby("pickup")[["pickup_lat", "pickup_lon"]]
                             .first().to_dict("index"))
        self.city_coords_.update(
            f.groupby("delivery")[["delivery_lat", "delivery_lon"]]
            .first().rename(columns={"delivery_lat": "pickup_lat",
                                     "delivery_lon": "pickup_lon"}).to_dict("index"))
        # Fixed category vocabularies so train/predict codes always align;
        # categories unseen at predict time become NaN (treated as missing).
        self.categories_ = {c: sorted(frame[c].dropna().unique())
                            for c in ("pickup", "delivery", "equipment")}
        return self

    # -- imputation helpers -------------------------------------------------
    def impute(self, frame: pd.DataFrame,
               date_market: pd.Series | None) -> pd.DataFrame:
        f = frame.copy()
        f["weight_missing"] = f["weight"].isna().astype(int)
        med = f["equipment"].map(self.weight_median_by_equip_)
        f["weight"] = f["weight"].fillna(med).fillna(self.weight_median_)
        f["market_index_missing"] = f["market_index"].isna().astype(int)
        if date_market is not None:
            f["market_index"] = f["market_index"].fillna(f["date"].map(date_market))
        f["market_index"] = f["market_index"].fillna(self.market_index_mean_)
        return f


def date_market_means(frame: pd.DataFrame) -> pd.Series:
    """Mean observed market_index per calendar date (feature-only statistic)."""
    return frame.dropna(subset=["market_index"]).groupby("date")["market_index"].mean()


def build_features(frame: pd.DataFrame, stats: EncodingStats,
                   date_market: pd.Series | None = None) -> pd.DataFrame:
    """Return the model-ready feature frame (index preserved)."""
    f = stats.impute(frame, date_market)
    out = pd.DataFrame(index=f.index)
    out["distance"] = f["distance"]
    out["weight"] = f["weight"]
    out["weight_missing"] = f["weight_missing"]
    out["market_index"] = f["market_index"]
    out["market_index_missing"] = f["market_index_missing"]
    out["quote_signal"] = f["quote_signal"]
    out["pickup_lat"] = f["pickup_lat"]
    out["pickup_lon"] = f["pickup_lon"]
    out["delivery_lat"] = f["delivery_lat"]
    out["delivery_lon"] = f["delivery_lon"]
    out["haversine"] = haversine_mi(f["pickup_lat"], f["pickup_lon"],
                                    f["delivery_lat"], f["delivery_lon"])
    out["lat_diff"] = f["delivery_lat"] - f["pickup_lat"]
    out["lon_diff"] = f["delivery_lon"] - f["pickup_lon"]
    d = f["date"]
    out["dayofyear"] = d.dt.dayofyear
    out["dayofweek"] = d.dt.dayofweek
    out["month"] = d.dt.month
    out["is_weekend"] = (d.dt.dayofweek >= 5).astype(int)
    out["dow_sin"] = np.sin(2 * np.pi * d.dt.dayofweek / 7)
    out["dow_cos"] = np.cos(2 * np.pi * d.dt.dayofweek / 7)
    out["doy_sin"] = np.sin(2 * np.pi * d.dt.dayofyear / 365.25)
    out["doy_cos"] = np.cos(2 * np.pi * d.dt.dayofyear / 365.25)

    for name, table, keys in (
        ("lane_rpm_te", stats.lane_, ["pickup", "delivery"]),
        ("pickup_rpm_te", stats.pickup_, ["pickup"]),
        ("delivery_rpm_te", stats.delivery_, ["delivery"]),
    ):
        merged = f[keys].merge(table[keys + ["te"]], on=keys, how="left")
        out[name] = merged["te"].fillna(stats.global_rpm_).to_numpy()
    counts = f[["pickup", "delivery"]].merge(stats.lane_counts_,
                                             on=["pickup", "delivery"], how="left")
    out["lane_count_log"] = np.log1p(counts["lane_n"].fillna(0).to_numpy())

    for c in CATEGORICAL:
        out[c] = pd.Categorical(f[c], categories=stats.categories_[c])
    return out


FEATURES = ["distance", "weight", "weight_missing", "market_index",
            "market_index_missing", "quote_signal", "pickup_lat", "pickup_lon",
            "delivery_lat", "delivery_lon", "haversine", "lat_diff", "lon_diff",
            "dayofyear", "dayofweek", "month", "is_weekend", "dow_sin",
            "dow_cos", "doy_sin", "doy_cos", "lane_rpm_te", "pickup_rpm_te",
            "delivery_rpm_te", "lane_count_log",
            "pickup", "delivery", "equipment"]
