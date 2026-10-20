#!/usr/bin/env bash
# Full pipeline: explore -> train -> predict -> score -> report.
# Run from the repository root.
set -euo pipefail

python -m pip install -r requirements.txt        # scorer deps
python -m pip install -r requirements-model.txt  # model deps

python src/eda.py        # exploratory analysis (printed summary)
python src/train.py      # holdout comparison + final model -> models/
python src/predict.py    # validation_predictions.csv + december-chart-inputs.csv
python score.py --predictions validation_predictions.csv \
                --december-predictions december-chart-inputs.csv
python src/make_report.py  # report.pdf (embeds scorer_results/candidate_december.png)
