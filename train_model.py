import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")

# ─────────────────────────────
# PATH CONFIG
# ─────────────────────────────
BASE_DIR = Path(__file__).parent

DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "model" / "autogluon_model"
OUTPUT_DIR = BASE_DIR / "output"

DATA_PATH = DATA_DIR / "tourism_cleaned.csv"

TARGET_COL = "visitors_total"

TIME_LIMIT = 120
PRESET = "medium_quality"

DATA_DIR.mkdir(exist_ok=True)
MODEL_DIR.parent.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# ─────────────────────────────
# LOAD DATA
# ─────────────────────────────
print("Loading dataset")

df = pd.read_csv(DATA_PATH, parse_dates=["date"])

print("Records:", len(df))
print("Provinces:", df["province"].nunique())

# ─────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────
print("Feature engineering")

df["province_code"] = df["province"].astype("category").cat.codes
df["region_code"] = df["region"].astype("category").cat.codes

df = df.sort_values(["province", "date"])

df["visitors_lag1"] = df.groupby("province")[TARGET_COL].shift(1)
df["visitors_lag3"] = df.groupby("province")[TARGET_COL].shift(3)

df["quarter"] = df["date"].dt.quarter

df_model = df.dropna()

# ─────────────────────────────
# FEATURE LIST
# ─────────────────────────────
FEATURE_COLS = [
    "year",
    "month",
    "quarter",
    "month_sin",
    "month_cos",
    "is_high_season",
    "province_code",
    "region_code",
    "occupancy_rate",
    "checkin_count",
    "foreign_ratio",
    "revenue_total_mb",
    "revenue_per_visitor",
    "visitors_lag1",
    "visitors_lag3",
]

# ─────────────────────────────
# TRAIN / TEST SPLIT
# ─────────────────────────────
train_df = df_model[df_model["year"] <= 2024]
test_df = df_model[df_model["year"] == 2025]

train_ag = train_df[FEATURE_COLS + [TARGET_COL]]
test_ag = test_df[FEATURE_COLS + [TARGET_COL]]

print("Train rows:", len(train_ag))
print("Test rows:", len(test_ag))

# ─────────────────────────────
# TRAIN MODEL
# ─────────────────────────────
print("Training model")

from autogluon.tabular import TabularPredictor

predictor = TabularPredictor(
    label=TARGET_COL,
    path=str(MODEL_DIR),
    eval_metric="root_mean_squared_error",
).fit(
    train_data=train_ag,
    time_limit=TIME_LIMIT,
    presets=PRESET,
)

# ─────────────────────────────
# EVALUATE (2025)
# ─────────────────────────────
print("Evaluating model")

y_pred = predictor.predict(test_ag.drop(columns=[TARGET_COL]))
y_true = test_ag[TARGET_COL].values

mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
r2 = r2_score(y_true, y_pred)

mape = np.mean(
    np.abs((y_true - y_pred) / np.where(y_true == 0, 1, y_true))
) * 100

print("MAE:", mae)
print("RMSE:", rmse)
print("R2:", r2)
print("MAPE:", mape)

# ─────────────────────────────
# SAVE PREDICTIONS 2025
# ─────────────────────────────
result_df = test_df[["date", "province", "region", TARGET_COL]].copy()

result_df["predicted"] = y_pred.values
result_df["error"] = result_df[TARGET_COL] - result_df["predicted"]

result_df["error_pct"] = abs(
    result_df["error"] / result_df[TARGET_COL].replace(0, np.nan) * 100
)

result_df.to_csv(
    OUTPUT_DIR / "predictions_2025.csv",
    index=False
)

print("Saved predictions_2025.csv")

# ─────────────────────────────
# SAVE METRICS
# ─────────────────────────────
metrics = {
    "mae": round(mae, 2),
    "rmse": round(rmse, 2),
    "r2": round(r2, 4),
    "mape": round(mape, 2),
    "best_model": predictor.model_best,
}

with open(OUTPUT_DIR / "model_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("Saved model_metrics.json")

# ─────────────────────────────
# FORECAST 2026
# ─────────────────────────────
print("Forecasting 2026")

last_rows = df_model.sort_values("date").groupby("province").tail(1)

future_results = []

for step in range(1, 13):

    future_df = last_rows.copy()

    future_df["date"] = future_df["date"] + pd.DateOffset(months=step)

    future_df["year"] = future_df["date"].dt.year
    future_df["month"] = future_df["date"].dt.month
    future_df["quarter"] = future_df["date"].dt.quarter

    future_df["month_sin"] = np.sin(2*np.pi*future_df["month"]/12)
    future_df["month_cos"] = np.cos(2*np.pi*future_df["month"]/12)

    pred = predictor.predict(future_df[FEATURE_COLS])

    future_df["predicted_visitors"] = pred.values

    future_results.append(
        future_df[
            ["date","province","region","predicted_visitors"]
        ]
    )

    # update lag
    last_rows["visitors_lag3"] = last_rows["visitors_lag1"]
    last_rows["visitors_lag1"] = pred.values

future_predictions = pd.concat(future_results)

future_predictions.to_csv(
    OUTPUT_DIR / "future_predictions_2026.csv",
    index=False
)

print("Saved future_predictions_2026.csv")

print("Training complete")
print("Best model:", predictor.model_best)