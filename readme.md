# Freight Rate Prediction Challenge

See `Freight_Rate_ML_Assessment.pdf` for the assessment instructions.

## What to do

1. Train and validate your model using `data/train_test.csv`.
2. Predict every load in `data/validation.csv`. Each load has a unique `load_id`.
3. Fill the matching `predicted_rate` values in `data/validation_predictions_template.csv` and save it as `validation_predictions.csv`.
4. Predict every row in `data/december_chart_inputs.csv` by filling its `predicted_rate` column.
5. Install the scorer requirements and run:

```bash
python -m pip install -r requirements.txt
python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv
```

The scorer validates both files and creates `scorer_results/candidate_december.png`.

## Submit

- GitHub repository containing your code, dependencies, and run instructions
- `validation_predictions.csv`
- PDF or DOCX report containing your validation, data split approach and `candidate_december.png`
- 2-3 minute Loom link

---

## Solution (Habib Ur Rehman)

A LightGBM regressor trained on `log1p(posted_rate)` with leakage-safe
rate-per-mile target encodings for lane and city effects. Held-out
October 2025 (temporal split, fit on Jan-Sep): **RMSE $657.6,
MAE $132.6, R2 0.815, MAPE 6.16%**. Full write-up in `SOLUTION.md` and
`report.pdf`.

### Layout

- `data/` — the assessment files (`train_test.csv`, `validation.csv`,
  `validation_predictions_template.csv`, `december_chart_inputs.csv`)
- `src/features.py` — feature engineering and target encodings
- `src/eda.py` — exploratory analysis summary
- `src/train.py` — holdout comparison + final model training (`models/`)
- `src/predict.py` — writes `validation_predictions.csv` and the
  completed `december-chart-inputs.csv`
- `src/experiments.py` — hyper-parameter sweep behind the final config
- `src/make_report.py` — builds `report.pdf`
- `score.py` — the supplied validator/chart generator (unmodified)

### Run

```bash
python -m pip install -r requirements.txt        # scorer dependencies
python -m pip install -r requirements-model.txt  # model dependencies
bash run_all.sh
```

or step by step:

```bash
python src/train.py
python src/predict.py
python score.py --predictions validation_predictions.csv --december-predictions december-chart-inputs.csv
python src/make_report.py
```
