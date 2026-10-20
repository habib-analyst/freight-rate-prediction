"""Exploratory data analysis for the Freight Rate Prediction Challenge.

Prints a structured summary used in the report and SOLUTION.md.
Run: python src/eda.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

pd.set_option("display.width", 160)

ROOT = "."
train = pd.read_csv(f"{ROOT}/data/train_test.csv", parse_dates=["date"])
valid = pd.read_csv(f"{ROOT}/data/validation.csv", parse_dates=["date"])

print("=" * 78)
print("SHAPES")
print("train:", train.shape, " validation:", valid.shape)
print("\nDTYPES / MISSING (train)")
print(train.dtypes)
print(train.isna().sum())
print("\nMISSING (validation)")
print(valid.isna().sum())

print("\n" + "=" * 78)
print("DATE RANGES")
print("train:", train["date"].min(), "->", train["date"].max())
print("valid:", valid["date"].min(), "->", valid["date"].max())
print("loads per month (train):")
print(train.set_index("date").resample("ME")["load_id"].count())
print("loads per month (validation):")
print(valid.set_index("date").resample("ME")["load_id"].count())

print("\n" + "=" * 78)
print("TARGET: posted_rate")
print(train["posted_rate"].describe(percentiles=[.01, .05, .25, .5, .75, .95, .99]))
print("zeros/negatives:", (train["posted_rate"] <= 0).sum())
print("duplicated load_id:", train["load_id"].duplicated().sum())
print("fully duplicated rows:", train.duplicated().sum())

print("\n" + "=" * 78)
print("NUMERIC FEATURES (train)")
print(train[["distance", "weight", "market_index", "quote_signal"]].describe(
    percentiles=[.01, .5, .99]).T)

print("\n" + "=" * 78)
print("EQUIPMENT")
print(train["equipment"].value_counts(dropna=False))
print("validation:\n", valid["equipment"].value_counts(dropna=False))

print("\n" + "=" * 78)
print("CITIES / LANES")
print("unique pickups:", train["pickup"].nunique(), " deliveries:", train["delivery"].nunique())
print("unique lanes:", train.groupby(["pickup", "delivery"]).ngroups)
print("Lexington rows (pickup):", (train["pickup"] == "Lexington").sum(),
      " Fort Wayne rows (delivery):", (train["delivery"] == "Fort Wayne").sum())
lw = train[(train["pickup"] == "Lexington") & (train["delivery"] == "Fort Wayne")]
print("exact Lexington->Fort Wayne rows:", len(lw))
if len(lw):
    print(lw[["distance", "equipment", "weight", "posted_rate"]].describe().T)

print("\n" + "=" * 78)
print("MARKET_INDEX / QUOTE_SIGNAL STRUCTURE")
g = train.groupby("date")[["market_index", "quote_signal"]].agg(["mean", "std", "nunique"])
print("per-date nunique of market_index: min/max =",
      train.groupby("date")["market_index"].nunique().min(),
      train.groupby("date")["market_index"].nunique().max())
print("per-date std of market_index: mean =", train.groupby("date")["market_index"].std().mean())
print("per-date nunique of quote_signal: min/max =",
      train.groupby("date")["quote_signal"].nunique().min(),
      train.groupby("date")["quote_signal"].nunique().max())
print("per-date std of quote_signal: mean =", train.groupby("date")["quote_signal"].std().mean())
daily_mi = train.groupby("date")["market_index"].mean()
print("daily market_index: first 5\n", daily_mi.head())
print("daily market_index: last 5\n", daily_mi.tail())
daily_qs = train.groupby("date")["quote_signal"].mean()
print("daily quote_signal mean: first/last:", daily_qs.iloc[0], daily_qs.iloc[-1])

print("\n" + "=" * 78)
print("CORRELATIONS WITH TARGET (train, pearson / spearman)")
num = train[["distance", "weight", "market_index", "quote_signal",
             "pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon", "posted_rate"]]
print(pd.DataFrame({"pearson": num.corr()["posted_rate"],
                    "spearman": num.corr(method="spearman")["posted_rate"]}).drop("posted_rate"))

print("\nrate per mile:")
train["rpm"] = train["posted_rate"] / train["distance"]
print(train["rpm"].describe(percentiles=[.01, .05, .5, .95, .99]))

print("\n" + "=" * 78)
print("HAVERSINE vs DISTANCE (are coordinates trustworthy?)")
R = 3958.8
lat1, lon1 = np.radians(train["pickup_lat"]), np.radians(train["pickup_lon"])
lat2, lon2 = np.radians(train["delivery_lat"]), np.radians(train["delivery_lon"])
hav = 2 * R * np.arcsin(np.sqrt(np.sin((lat2 - lat1) / 2) ** 2
                                + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2))
print("corr(haversine, distance):", np.corrcoef(hav, train["distance"])[0, 1])
print("median |haversine - distance|:", np.median(np.abs(hav - train["distance"])))
# same city -> consistent coordinates?
city = train.groupby("pickup")[["pickup_lat", "pickup_lon"]].std().mean()
print("mean within-city std of pickup coords (lat, lon):\n", city)

print("\n" + "=" * 78)
print("TEMPORAL TARGET TREND")
m = train.set_index("date").resample("ME")["posted_rate"].agg(["mean", "median", "count"])
print(m)
w = train.set_index("date").resample("W")["posted_rate"].mean()
print("weekly mean posted_rate: first:", round(w.iloc[0], 1), " last:", round(w.iloc[-1], 1),
      " min:", round(w.min(), 1), " max:", round(w.max(), 1))

print("\nTARGET BY EQUIPMENT")
print(train.groupby("equipment")["posted_rate"].agg(["mean", "median", "count"]))
print("\nTARGET BY DISTANCE BAND")
train["dist_band"] = pd.cut(train["distance"], [0, 250, 500, 750, 1000, 1500, 3000])
print(train.groupby("dist_band", observed=True)["posted_rate"].agg(["mean", "median", "count"]))

print("\n" + "=" * 78)
print("VALIDATION vs TRAIN — distribution shift check")
for c in ["distance", "weight", "market_index", "quote_signal"]:
    print(f"{c:14s} train mean {train[c].mean():10.2f}  valid mean {valid[c].mean():10.2f}")
print("valid lanes unseen in train:",
      valid.merge(train[["pickup", "delivery"]].drop_duplicates(),
                  on=["pickup", "delivery"], how="left", indicator=True)["_merge"]
      .eq("left_only").sum(), "of", len(valid))
print("valid pickup cities unseen:", set(valid["pickup"]) - set(train["pickup"]))
print("valid delivery cities unseen:", set(valid["delivery"]) - set(train["delivery"]))
