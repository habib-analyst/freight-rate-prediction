"""Build report.pdf — the written report required by the assessment.

Contents: approach, EDA insights, split/validation design, local validation
metrics, model + feature engineering, the December chart produced by
score.py, and reproducibility instructions.

Run AFTER train.py, predict.py and score.py:
    python src/make_report.py
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (Image, PageBreak, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)

TEAL = colors.HexColor("#064A56")
LIGHT = colors.HexColor("#E8F0F1")


def make_figures() -> tuple[str, str]:
    train = pd.read_csv("data/train_test.csv", parse_dates=["date"])
    monthly = train.set_index("date").resample("ME").agg(
        rate=("posted_rate", "mean"), mi=("market_index", "mean"))
    fig, ax1 = plt.subplots(figsize=(7.4, 2.9), dpi=200)
    ax1.plot(monthly.index, monthly["rate"], color="#064A56", lw=2, marker="o", ms=3)
    ax1.set_ylabel("Mean posted rate ($)")
    ax1.grid(axis="y", color="#D9E2E4")
    ax2 = ax1.twinx()
    ax2.plot(monthly.index, monthly["mi"], color="#C05621", lw=1.6, ls="--")
    ax2.set_ylabel("Mean market index", color="#C05621")
    ax1.set_title("Monthly mean posted rate and market index, Jan-Oct 2025",
                  loc="left", fontsize=10, fontweight="bold")
    fig.tight_layout()
    fig.savefig("report_fig_trend.png", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.9), dpi=200)
    axes[0].hexbin(train["distance"], train["posted_rate"], gridsize=45,
                   cmap="viridis", mincnt=1)
    axes[0].set_xlabel("Distance (mi)")
    axes[0].set_ylabel("Posted rate ($)")
    axes[0].set_title("Rate vs. distance", loc="left", fontsize=10, fontweight="bold")
    rpm = train["posted_rate"] / train["distance"]
    for i, eq in enumerate(["Dry Van", "Reefer", "Flatbed"]):
        axes[1].boxplot(rpm[train["equipment"] == eq], positions=[i],
                        widths=0.55, showfliers=False)
    axes[1].set_xticks([0, 1, 2], ["Dry Van", "Reefer", "Flatbed"])
    axes[1].set_ylabel("Rate per mile ($)")
    axes[1].set_title("Rate per mile by equipment", loc="left", fontsize=10,
                      fontweight="bold")
    axes[1].grid(axis="y", color="#D9E2E4")
    fig.tight_layout()
    fig.savefig("report_fig_dist.png", bbox_inches="tight")
    plt.close(fig)
    return "report_fig_trend.png", "report_fig_dist.png"


def main() -> None:
    with open("models/metrics.json") as fh:
        meta = json.load(fh)
    hold = meta["holdout"]
    winner = meta["winner"]
    w = hold[winner]
    fig_trend, fig_dist = make_figures()

    styles = getSampleStyleSheet()
    h1 = ParagraphStyle("h1", parent=styles["Title"], textColor=TEAL, fontSize=20)
    h2 = ParagraphStyle("h2", parent=styles["Heading2"], textColor=TEAL,
                        fontSize=13, spaceBefore=14)
    body = ParagraphStyle("body", parent=styles["BodyText"], fontSize=10.5,
                          leading=15, alignment=4)
    small = ParagraphStyle("small", parent=body, fontSize=9.5, textColor=colors.grey)

    rows = [[mode.replace("_", " "), f"{v['rmse']:,.0f}", f"{v['mae']:,.0f}",
             f"{v['r2']:.4f}", f"{v['mape_pct']:.2f}%"]
            for mode, v in hold.items()]
    tbl = Table([["Target formulation", "RMSE ($)", "MAE ($)", "R2", "MAPE"]] + rows,
                colWidths=[4.6 * cm, 2.8 * cm, 2.8 * cm, 2.4 * cm, 2.4 * cm])
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), TEAL),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#9DAFB3")),
        ("FONTSIZE", (0, 0), (-1, -1), 9.5),
        ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
    ]))

    story = [
        Paragraph("Freight Rate Prediction Challenge", h1),
        Paragraph("Machine Learning Engineer Assessment — Spotter", body),
        Paragraph("Habib Ur Rehman", body),
        Spacer(1, 6),
        Paragraph("Summary", h2),
        Paragraph(
            "The task is to predict the posted rate for 12,000 freight loads "
            "dated November-December 2025, given 48,000 labelled loads from "
            "January-October 2025, and to produce a daily rate curve for one "
            "fixed lane (Lexington to Fort Wayne) across December 2025. The "
            "submitted model is a LightGBM gradient-boosting regressor "
            "trained on the logarithm of the rate, with leakage-safe "
            "rate-per-mile target encodings for lane and city effects. On a "
            "held-out October 2025 split that reproduces the forecasting "
            f"setup of the task, it scores RMSE ${w['rmse']:,.0f}, "
            f"MAE ${w['mae']:,.0f}, R2 {w['r2']:.3f} and MAPE "
            f"{w['mape_pct']:.2f}%.", body),
        Paragraph("Data and exploration", h2),
        Paragraph(
            "The development file holds 48,000 loads over 64 pickup and 64 "
            "delivery cities (4,014 distinct lanes) and three equipment "
            "types. Distance is the dominant driver of the rate: the Spearman "
            "correlation between distance and posted rate is 0.98, and the "
            "median rate per mile is $2.15. Reefer loads price highest per "
            "mile, then Flatbed, then Dry Van. Rates drift upward through "
            "the year (monthly mean $2,256 in January, $2,497 in June, "
            "$2,379 in October) alongside the provided market index, which "
            "behaves as a daily market level with small per-load noise. "
            "Coordinates are internally consistent: each city has exactly "
            "one coordinate pair and the haversine distance correlates "
            "0.9995 with the stated distance. Three data-quality issues were "
            "found and handled: weight is missing on 300 rows and market "
            "index on 374 rows (imputed, see below); 1,461 of the 12,000 "
            "validation lanes never appear in training, and 8 validation "
            "cities are new, so lane and city effects need shrinkage and a "
            "global fallback; and the validation market index averages 0.93 "
            "against 1.08 in training, i.e. a softer market in the "
            "prediction window.", body),
        Image(fig_trend, width=16.6 * cm, height=6.5 * cm),
        Image(fig_dist, width=16.6 * cm, height=6.5 * cm),
        Paragraph("Validation design", h2),
        Paragraph(
            "Because the deliverable is a forecast, the split is temporal, "
            "not random: the model is fitted on 1 January - 30 September "
            "2025 (43,147 loads) and scored on October 2025 (4,853 loads), "
            "the most recent labelled month. Every target-derived statistic "
            "(lane and city encodings) is fitted on the training window "
            "only. The one same-period statistic used at prediction time is "
            "the mean of the observed market index on the load's own date "
            "for imputing missing market values; that mirrors deployment, "
            "where the full feature set of validation.csv is available. "
            "Three target formulations were compared on this split: the raw "
            "rate, its logarithm, and the rate per mile. Results:", body),
        Spacer(1, 4),
        tbl,
        Spacer(1, 4),
        Paragraph(
            f"The log-rate formulation (winner: {winner.replace('_', ' ')}) "
            "was selected. A small hyper-parameter sweep around it is "
            "reproduced in src/experiments.py. The final model is refitted "
            "on all 48,000 labelled loads with the early-stopping iteration "
            "count scaled by the training-size ratio.", body),
        Paragraph("Features and model", h2),
        Paragraph(
            "Features: distance, weight, market index, quote signal, the "
            "four coordinates, haversine distance, latitude/longitude "
            "differences, calendar features (day of year, day of week, "
            "weekend flag, sine/cosine encodings), and three shrunk "
            "rate-per-mile target encodings (lane, pickup city, delivery "
            "city) plus the lane's historical load count. Missing weights "
            "are filled with the equipment-group median and missing market "
            "index values with the same-date mean; both carry missingness "
            "flags. LightGBM was chosen because the signal is dominated by "
            "nonlinear interactions (distance with lane, equipment and "
            "market level), the data mixes numeric and high-cardinality "
            "categorical inputs, and it trains in about a minute on this "
            "hardware, which keeps the pipeline easy to rerun.", body),
        Paragraph("December chart", h2),
        Paragraph(
            "The chart below is produced by the supplied score.py from the "
            "completed december-chart-inputs.csv. Only the date varies in "
            "that file, so the curve reflects the day-of-week pattern and "
            "the December market conditions: the market index and quote "
            "signal for each December date are taken from the December "
            "loads in validation.csv (per-date means of the provided "
            "features). Reference point: the 32 historical Lexington to "
            "Fort Wayne loads in training average $857 at 364 miles.", body),
        Image("scorer_results/candidate_december.png", width=16.6 * cm,
              height=7.4 * cm),
        Paragraph("Reproducibility", h2),
        Paragraph(
            "python -m pip install -r requirements.txt (scorer) and "
            "-r requirements-model.txt (model); python src/eda.py; "
            "python src/train.py; python src/predict.py; "
            "python score.py --predictions validation_predictions.csv "
            "--december-predictions december-chart-inputs.csv. "
            "Full detail in SOLUTION.md and the project readme.", body),
        Paragraph("Final validation metrics are calculated by Spotter after "
                  "submission, per the assessment instructions.", small),
    ]
    doc = SimpleDocTemplate("report.pdf", pagesize=A4,
                            leftMargin=2 * cm, rightMargin=2 * cm,
                            topMargin=2 * cm, bottomMargin=2 * cm,
                            title="Freight Rate Prediction Challenge — Report",
                            author="Habib Ur Rehman")
    doc.build(story)
    print("report.pdf written")


if __name__ == "__main__":
    main()
