# Solution Summary — Freight Rate Prediction Challenge

**Author:** Habib Ur Rehman
**Deliverables:** `validation_predictions.csv` (12,000 predictions),
completed `december-chart-inputs.csv`, `report.pdf`, and this repository.

## Approach

The target, `posted_rate`, is driven first by distance (Spearman
correlation 0.98; median $2.15 per mile) and then by lane, equipment and
market conditions. The model is a LightGBM gradient-boosting regressor
trained on `log1p(posted_rate)`, which consistently beat the raw-rate and
rate-per-mile formulations on the holdout (table below). Predictions are
exponentiated back to dollars and clipped at a $10 floor.

**Features** (see `src/features.py`):

- Raw: `distance`, `weight`, `market_index`, `quote_signal`, the four
  coordinates, `equipment`, `pickup`, `delivery`.
- Geography: haversine distance between the coordinate pairs and the
  latitude/longitude differences. (Coordinates are internally consistent:
  one fixed point per city, haversine vs. stated distance correlation
  0.9995.)
- Calendar: day of year, day of week, month, weekend flag, sine/cosine
  encodings of both cycles.
- Target encodings: mean rate-per-mile for the lane, the pickup city and
  the delivery city, each shrunk toward the global mean by its load count
  (k = 20 lanes, k = 30 cities), plus the lane's load count. Unseen
  lanes/cities fall back to the global rate per mile — needed in
  validation, where 1,461 of 12,000 lanes and 8 cities are new.

**Data quality handling:** `weight` is missing on 300 training rows
(165 in validation) and `market_index` on 374 (249) — filled with the
equipment-group median and the same-date mean respectively, each with a
missingness flag. No rows are dropped and no target values are touched.

## Validation design

The split is temporal, mirroring the forecasting task: fit on
**2025-01-01 → 2025-09-30** (43,147 loads), score on **October 2025**
(4,853 loads). All target-derived statistics are fitted on the training
window only, so the estimate is leakage-free. The single same-period
statistic used at prediction time is the feature-only same-date mean of
`market_index` for imputation, which mirrors deployment (validation.csv
ships the full feature set).

Formulation comparison on the October holdout (final configuration:
learning rate 0.02, 255 leaves, min 60 rows per leaf, chosen by the sweep
in `src/experiments.py`):

| Target            | RMSE ($) | MAE ($) | R²     | MAPE  |
|-------------------|----------|---------|--------|-------|
| rate              | 667.4    | 156.7   | 0.809  | 8.66% |
| **log rate**      | **657.6**| **132.6**| **0.815** | **6.16%** |
| rate per mile     | 665.2    | 161.4   | 0.811  | 7.29% |
| ensemble (rate+rpm)| 662.3   | 143.4   | 0.812  | 7.04% |

The final model is refitted on all 48,000 labelled loads (Jan–Oct) with
the holdout iteration count (215) scaled by the training-size ratio →
251 rounds.

## December chart

`december-chart-inputs.csv` fixes Lexington → Fort Wayne, 360 miles,
Dry Van, 32,000 lb; only the date varies. Market features for each
December date are the per-date means of `market_index` / `quote_signal`
over the December loads in `validation.csv` (provided features, no
target involved). Predictions run $819–$856 with a clear weekly cycle
(Sundays lowest), consistent with the 32 historical loads on this lane,
which average $857.

## Scorer output (provided `score.py`, unmodified)

```
Validated 12,000 final predictions.
Validated 31 fixed December predictions.
Created chart: scorer_results/candidate_december.png
Final validation metrics are calculated by Spotter after submission.
```

## Reproduce

```bash
bash run_all.sh
```

or step by step: `python src/eda.py`, `python src/train.py`,
`python src/predict.py`, then the `score.py` command from the readme,
then `python src/make_report.py`. Model dependencies are in
`requirements-model.txt`; the scorer's are in `requirements.txt`.
Runtime for train + predict is about three minutes on two CPU cores.
