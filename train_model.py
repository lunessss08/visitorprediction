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
MODEL_DIR = BASE_DIR / "model"
OUTPUT_DIR = BASE_DIR / "output"

DATA_PATH = DATA_DIR / "tourism_cleaned.csv"

TARGET_THAI = "visitors_thai"
TARGET_FOREIGN = "visitors_foreign"

TIME_LIMIT = 120
PRESET = "medium_quality"

DATA_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)
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

df["visitors_thai_lag1"] = df.groupby("province")[TARGET_THAI].shift(1)
df["visitors_thai_lag3"] = df.groupby("province")[TARGET_THAI].shift(3)

df["visitors_foreign_lag1"] = df.groupby("province")[TARGET_FOREIGN].shift(1)
df["visitors_foreign_lag3"] = df.groupby("province")[TARGET_FOREIGN].shift(3)

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
    "visitors_thai_lag1",
    "visitors_thai_lag3",
    "visitors_foreign_lag1",
    "visitors_foreign_lag3",
]

# ─────────────────────────────
# TRAIN / TEST SPLIT
# ─────────────────────────────
train_df = df_model[df_model["year"] <= 2024]
test_df = df_model[df_model["year"] == 2025]

train_thai = train_df[FEATURE_COLS + [TARGET_THAI]]
test_thai = test_df[FEATURE_COLS + [TARGET_THAI]]

train_foreign = train_df[FEATURE_COLS + [TARGET_FOREIGN]]
test_foreign = test_df[FEATURE_COLS + [TARGET_FOREIGN]]

print("Train rows:", len(train_thai))
print("Test rows:", len(test_thai))

# ─────────────────────────────
# TRAIN MODEL
# ─────────────────────────────
print("Training models")

from autogluon.tabular import TabularPredictor

predictor_thai = TabularPredictor(
    label=TARGET_THAI,
    path=str(MODEL_DIR / "thai_model"),
    eval_metric="root_mean_squared_error",
).fit(
    train_data=train_thai,
    time_limit=TIME_LIMIT,
    presets=PRESET,
)

predictor_foreign = TabularPredictor(
    label=TARGET_FOREIGN,
    path=str(MODEL_DIR / "foreign_model"),
    eval_metric="root_mean_squared_error",
).fit(
    train_data=train_foreign,
    time_limit=TIME_LIMIT,
    presets=PRESET,
)

# ─────────────────────────────
# EVALUATE (2025)
# ─────────────────────────────
print("Evaluating models")

pred_thai = predictor_thai.predict(test_thai.drop(columns=[TARGET_THAI]))

pred_foreign = predictor_foreign.predict(test_foreign.drop(columns=[TARGET_FOREIGN]))

pred_total = pred_thai + pred_foreign

y_true = test_df[TARGET_THAI] + test_df[TARGET_FOREIGN]

mae = mean_absolute_error(y_true, pred_total)
rmse = np.sqrt(mean_squared_error(y_true, pred_total))
r2 = r2_score(y_true, pred_total)

mape = np.mean(np.abs((y_true - pred_total) / np.where(y_true == 0, 1, y_true))) * 100

print("MAE:", mae)
print("RMSE:", rmse)
print("R2:", r2)
print("MAPE:", mape)

# ─────────────────────────────
# SAVE PREDICTIONS 2025
# ─────────────────────────────
result_df = test_df[["date", "province", "region"]].copy()

result_df["visitors_thai_pred"] = pred_thai.values
result_df["visitors_foreign_pred"] = pred_foreign.values
result_df["visitors_total_pred"] = pred_total.values

result_df.to_csv(OUTPUT_DIR / "predictions_2025.csv", index=False)

print("Saved predictions_2025.csv")

# ─────────────────────────────
# SAVE METRICS
# ─────────────────────────────
metrics = {
    "mae": round(mae, 2),
    "rmse": round(rmse, 2),
    "r2": round(r2, 4),
    "mape": round(mape, 2),
}

with open(OUTPUT_DIR / "model_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("Saved model_metrics.json")

# ─────────────────────────────
# FORECAST 2026
# ─────────────────────────────


# REFIT MODEL (ใช้ 2024 + 2025)

print("Refitting models with full data (2025 included)")

train_full = df_model[df_model["year"] <= 2025]

predictor_thai = TabularPredictor(
    label=TARGET_THAI,
    path=str(MODEL_DIR / "thai_model_final"),
    eval_metric="root_mean_squared_error",
).fit(
    train_data=train_full[FEATURE_COLS + [TARGET_THAI]],
    time_limit=TIME_LIMIT,
    presets=PRESET,
    refit_full=True
)

predictor_foreign = TabularPredictor(
    label=TARGET_FOREIGN,
    path=str(MODEL_DIR / "foreign_model_final"),
    eval_metric="root_mean_squared_error",
).fit(
    train_data=train_full[FEATURE_COLS + [TARGET_FOREIGN]],
    time_limit=TIME_LIMIT,
    presets=PRESET,
    refit_full=True
)

print("Forecasting 2026")

last_rows = df_model.sort_values("date").groupby("province").tail(1)

future_results = []

for step in range(1, 13):

    future_df = last_rows.copy()

    next_date = last_rows["date"].max() + pd.DateOffset(months=1)
    next_date = pd.to_datetime(next_date).to_period("M").to_timestamp()
    
    future_df["date"] = next_date
    last_rows["date"] = next_date
    
    future_df["year"] = future_df["date"].dt.year
    future_df["month"] = future_df["date"].dt.month
    future_df["quarter"] = future_df["date"].dt.quarter 

    future_df["month_sin"] = np.sin(2 * np.pi * future_df["month"] / 12)
    future_df["month_cos"] = np.cos(2 * np.pi * future_df["month"] / 12)

    pred_thai = predictor_thai.predict(future_df[FEATURE_COLS])
    pred_foreign = predictor_foreign.predict(future_df[FEATURE_COLS])

    future_df["visitors_thai"] = pred_thai.values
    future_df["visitors_foreign"] = pred_foreign.values
    future_df["visitors_total"] = (
        future_df["visitors_thai"] + future_df["visitors_foreign"]
    )

    future_results.append(
        future_df[
            [
                "date",
                "province",
                "region",
                "visitors_thai",
                "visitors_foreign",
                "visitors_total",
            ]
        ]
    )

    last_rows["visitors_thai_lag3"] = last_rows["visitors_thai_lag1"]
    last_rows["visitors_thai_lag1"] = pred_thai.values

    last_rows["visitors_foreign_lag3"] = last_rows["visitors_foreign_lag1"]
    last_rows["visitors_foreign_lag1"] = pred_foreign.values

future_predictions = pd.concat(future_results)

future_predictions.to_csv(OUTPUT_DIR / "future_predictions_2026.csv", index=False)

print("Saved future_predictions_2026.csv")

print("Training complete")
