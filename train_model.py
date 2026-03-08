# """
# ========================================================
#   Tourism Forecasting - AutoGluon Training Script
#   พยากรณ์จำนวนนักท่องเที่ยวรายจังหวัด ประเทศไทย
#   ข้อมูล: ม.ค. 2567 - ธ.ค. 2568 (77 จังหวัด)

#   ✅ Script นี้ทำ Data Cleaning ในตัวเองอัตโนมัติ
#      ไม่ต้องพึ่งไฟล์ tourism_cleaned.csv
# ========================================================
# วิธีรัน:
#   1. วางไฟล์ xlsx ทั้ง 2 ไฟล์ในโฟลเดอร์เดียวกับ script
#   2. pip install autogluon.tabular openpyxl scikit-learn pandas numpy
#   3. python train_model.py
# ========================================================
# """

import os
import json
import glob
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────
DATA_PATH = "tourism_cleaned.csv"
MODEL_DIR = "autogluon_model"
TARGET_COL = "visitors_total"
TIME_LIMIT = 120  # วินาที (เพิ่มได้)
PRESET = "medium_quality"  # best_quality / high_quality / medium_quality / fast


# ══════════════════════════════════════════════════════
#  HELPER: Auto-detect xlsx files
# ══════════════════════════════════════════════════════
def find_xlsx_files():
    """หาไฟล์ xlsx ทั้ง 2 ในโฟลเดอร์ปัจจุบัน"""
    all_xlsx = sorted(glob.glob("*.xlsx"))
    if len(all_xlsx) < 2:
        print(f"  ❌ พบ xlsx แค่ {len(all_xlsx)} ไฟล์: {all_xlsx}")
        print("  กรุณาวางไฟล์ xlsx ทั้ง 2 ไว้ในโฟลเดอร์เดียวกัน")
        exit(1)
    # ไฟล์ปี 2567 มักมี "2567" ในชื่อ, ปี 2568 มี "2568"
    f67 = next((f for f in all_xlsx if "2567" in f), all_xlsx[0])
    f68 = next((f for f in all_xlsx if "2568" in f), all_xlsx[1])
    return f67, f68


# ══════════════════════════════════════════════════════
#  STEP 0: DATA CLEANING
# ══════════════════════════════════════════════════════
def run_cleaning(file_2567, file_2568):
    print("=" * 55)
    print("  STEP 0: Data Cleaning")
    print("=" * 55)
    print(f"  ไฟล์ 2567: {file_2567}")
    print(f"  ไฟล์ 2568: {file_2568}")

    month_map_2567 = {
        "ม.ค.2567(R)": (2024, 1),
        "ก.พ.2567(R)": (2024, 2),
        "มี.ค.2567(R)": (2024, 3),
        "เม.ย.2567(R)": (2024, 4),
        "พ.ค.2567(R)": (2024, 5),
        "มิ.ย.2567(R)": (2024, 6),
        "ก.ค.2567(R)": (2024, 7),
        "ส.ค.2567(R)": (2024, 8),
        "ก.ย.2567(R)": (2024, 9),
        "ต.ค.2567(R)": (2024, 10),
        "พ.ย.2567(R)": (2024, 11),
        "ธ.ค.2567(R)": (2024, 12),
    }
    month_map_2568 = {
        "ม.ค.2568p": (2025, 1),
        "ก.พ.2568p": (2025, 2),
        "มี.ค.2568p": (2025, 3),
        "เม.ย.2568p": (2025, 4),
        "พ.ค.2568p": (2025, 5),
        "มิ.ย.2568p": (2025, 6),
        "ก.ค.2568p": (2025, 7),
        "ส.ค.2568p": (2025, 8),
        "ก.ย.2568p": (2025, 9),
        "ต.ค.2568p": (2025, 10),
        "พ.ย.2568p": (2025, 11),
        "ธ.ค.2568p": (2025, 12),
    }
    province_fix = {"ประจวบศิรีขันธ์": "ประจวบคีรีขันธ์", "ศรีษะเกษ": "ศรีสะเกษ"}
    region_rows = {22, 31, 46, 64, 85, 86}
    region_map = {
        "กรุงเทพมหานคร": "กรุงเทพฯ",
        "กาญจนบุรี": "ภาคกลาง",
        "นครปฐม": "ภาคกลาง",
        "ประจวบคีรีขันธ์": "ภาคกลาง",
        "เพชรบุรี": "ภาคกลาง",
        "ราชบุรี": "ภาคกลาง",
        "สมุทรปราการ": "ภาคกลาง",
        "สมุทรสงคราม": "ภาคกลาง",
        "สมุทรสาคร": "ภาคกลาง",
        "ชัยนาท": "ภาคกลาง",
        "นนทบุรี": "ภาคกลาง",
        "ปทุมธานี": "ภาคกลาง",
        "พระนครศรีอยุธยา": "ภาคกลาง",
        "ลพบุรี": "ภาคกลาง",
        "สระบุรี": "ภาคกลาง",
        "สิงห์บุรี": "ภาคกลาง",
        "สุพรรณบุรี": "ภาคกลาง",
        "อ่างทอง": "ภาคกลาง",
        "จันทบุรี": "ภาคตะวันออก",
        "ฉะเชิงเทรา": "ภาคตะวันออก",
        "ชลบุรี": "ภาคตะวันออก",
        "ตราด": "ภาคตะวันออก",
        "นครนายก": "ภาคตะวันออก",
        "ปราจีนบุรี": "ภาคตะวันออก",
        "ระยอง": "ภาคตะวันออก",
        "สระแก้ว": "ภาคตะวันออก",
        "กระบี่": "ภาคใต้",
        "ชุมพร": "ภาคใต้",
        "นครศรีธรรมราช": "ภาคใต้",
        "พังงา": "ภาคใต้",
        "ภูเก็ต": "ภาคใต้",
        "ระนอง": "ภาคใต้",
        "สุราษฎร์ธานี": "ภาคใต้",
        "ตรัง": "ภาคใต้",
        "นราธิวาส": "ภาคใต้",
        "ปัตตานี": "ภาคใต้",
        "พัทลุง": "ภาคใต้",
        "ยะลา": "ภาคใต้",
        "สงขลา": "ภาคใต้",
        "สตูล": "ภาคใต้",
        "เชียงราย": "ภาคเหนือ",
        "เชียงใหม่": "ภาคเหนือ",
        "น่าน": "ภาคเหนือ",
        "พะเยา": "ภาคเหนือ",
        "แพร่": "ภาคเหนือ",
        "แม่ฮ่องสอน": "ภาคเหนือ",
        "ลำปาง": "ภาคเหนือ",
        "ลำพูน": "ภาคเหนือ",
        "กำแพงเพชร": "ภาคเหนือ",
        "ตาก": "ภาคเหนือ",
        "นครสวรรค์": "ภาคเหนือ",
        "พิจิตร": "ภาคเหนือ",
        "พิษณุโลก": "ภาคเหนือ",
        "เพชรบูรณ์": "ภาคเหนือ",
        "สุโขทัย": "ภาคเหนือ",
        "อุตรดิตถ์": "ภาคเหนือ",
        "อุทัยธานี": "ภาคเหนือ",
        "กาฬสินธุ์": "ภาคตะวันออกเฉียงเหนือ",
        "ขอนแก่น": "ภาคตะวันออกเฉียงเหนือ",
        "นครพนม": "ภาคตะวันออกเฉียงเหนือ",
        "บึงกาฬ": "ภาคตะวันออกเฉียงเหนือ",
        "มหาสารคาม": "ภาคตะวันออกเฉียงเหนือ",
        "เลย": "ภาคตะวันออกเฉียงเหนือ",
        "สกลนคร": "ภาคตะวันออกเฉียงเหนือ",
        "หนองคาย": "ภาคตะวันออกเฉียงเหนือ",
        "หนองบัวลำภู": "ภาคตะวันออกเฉียงเหนือ",
        "อุดรธานี": "ภาคตะวันออกเฉียงเหนือ",
        "ชัยภูมิ": "ภาคตะวันออกเฉียงเหนือ",
        "นครราชสีมา": "ภาคตะวันออกเฉียงเหนือ",
        "บุรีรัมย์": "ภาคตะวันออกเฉียงเหนือ",
        "มุกดาหาร": "ภาคตะวันออกเฉียงเหนือ",
        "ยโสธร": "ภาคตะวันออกเฉียงเหนือ",
        "ร้อยเอ็ด": "ภาคตะวันออกเฉียงเหนือ",
        "ศรีสะเกษ": "ภาคตะวันออกเฉียงเหนือ",
        "สุรินทร์": "ภาคตะวันออกเฉียงเหนือ",
        "อำนาจเจริญ": "ภาคตะวันออกเฉียงเหนือ",
        "อุบลราชธานี": "ภาคตะวันออกเฉียงเหนือ",
    }

    def extract_sheet(filepath, sheet_name, year, month):
        d = pd.read_excel(filepath, sheet_name=sheet_name, header=None)
        recs = []
        for i in range(4, 87):
            if i in region_rows or pd.isna(d.iloc[i, 1]):
                continue
            prov = province_fix.get(
                str(d.iloc[i, 1]).strip(), str(d.iloc[i, 1]).strip()
            )
            recs.append(
                {
                    "year": year,
                    "month": month,
                    "province": prov,
                    "occupancy_rate": pd.to_numeric(d.iloc[i, 2], errors="coerce"),
                    "checkin_count": pd.to_numeric(d.iloc[i, 5], errors="coerce"),
                    "visitors_total": pd.to_numeric(d.iloc[i, 8], errors="coerce"),
                    "visitors_thai": pd.to_numeric(d.iloc[i, 11], errors="coerce"),
                    "visitors_foreign": pd.to_numeric(d.iloc[i, 14], errors="coerce"),
                    "revenue_total_mb": pd.to_numeric(d.iloc[i, 17], errors="coerce"),
                    "revenue_thai_mb": pd.to_numeric(d.iloc[i, 20], errors="coerce"),
                    "revenue_foreign_mb": pd.to_numeric(d.iloc[i, 23], errors="coerce"),
                }
            )
        return recs

    all_records = []
    for sheet, (y, m) in month_map_2567.items():
        all_records.extend(extract_sheet(file_2567, sheet, y, m))
        print(f"  ✓ {sheet}")
    for sheet, (y, m) in month_map_2568.items():
        all_records.extend(extract_sheet(file_2568, sheet, y, m))
        print(f"  ✓ {sheet}")

    df = pd.DataFrame(all_records)
    df["date"] = pd.to_datetime(df[["year", "month"]].assign(day=1))
    df["region"] = df["province"].map(region_map).fillna("ภาคกลาง")
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["is_high_season"] = df["month"].isin([11, 12, 1, 2, 3]).astype(int)
    df["foreign_ratio"] = df["visitors_foreign"] / df["visitors_total"].replace(
        0, np.nan
    )
    df["revenue_per_visitor"] = (
        df["revenue_total_mb"] * 1_000_000 / df["visitors_total"].replace(0, np.nan)
    )
    df = df.sort_values(["province", "year", "month"]).reset_index(drop=True)

    # บันทึก CSV (utf-8 ไม่มี BOM)
    df.to_csv(DATA_PATH, index=False, encoding="utf-8")
    print(
        f"\n  ✅ {len(df):,} records | {df['province'].nunique()} จังหวัด | Missing: {df.isnull().sum().sum()}"
    )
    print(f"  ✅ Saved → {DATA_PATH}")
    return df


# ══════════════════════════════════════════════════════
#  STEP 1: LOAD / BUILD DATA
# ══════════════════════════════════════════════════════
print("=" * 55)
print("  STEP 1: Loading Data")
print("=" * 55)

need_rebuild = False
if not Path(DATA_PATH).exists():
    print(f"  ไม่พบ {DATA_PATH} → จะสร้างจาก xlsx")
    need_rebuild = True
else:
    try:
        _test = pd.read_csv(DATA_PATH, encoding="utf-8", nrows=5)
        if len(_test) == 0 or "visitors_total" not in _test.columns:
            print(f"  ไฟล์เสียหายหรือว่างเปล่า → จะสร้างใหม่")
            need_rebuild = True
        else:
            print(f"  ✅ พบ {DATA_PATH} ใช้งานได้")
    except Exception as e:
        print(f"  อ่านไฟล์ไม่ได้ ({e}) → จะสร้างใหม่")
        need_rebuild = True

if need_rebuild:
    f67, f68 = find_xlsx_files()
    df = run_cleaning(f67, f68)
else:
    df = pd.read_csv(DATA_PATH, encoding="utf-8", parse_dates=["date"])
    print(f"  ✅ Loaded: {len(df):,} records | {df['province'].nunique()} จังหวัด")

print(f"  Date range : {df['date'].min().date()} → {df['date'].max().date()}")

# ─────────────────────────────────────────────────────
#  STEP 2: FEATURE ENGINEERING
# ─────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("  STEP 2: Feature Engineering")
print("=" * 55)

df["province_code"] = df["province"].astype("category").cat.codes
df["region_code"] = df["region"].astype("category").cat.codes
df = df.sort_values(["province", "date"]).reset_index(drop=True)

df["visitors_lag1"] = df.groupby("province")[TARGET_COL].shift(1)
df["visitors_lag3"] = df.groupby("province")[TARGET_COL].shift(3)
df["visitors_lag12"] = df.groupby("province")[TARGET_COL].shift(12)
df["visitors_roll3"] = df.groupby("province")[TARGET_COL].transform(
    lambda x: x.shift(1).rolling(3, min_periods=1).mean()
)
df["quarter"] = df["date"].dt.quarter
print(f"  ✅ Added: lag1, lag3, lag12, roll3, quarter")

# ─────────────────────────────────────────────────────
#  STEP 3: FEATURE LIST
# ─────────────────────────────────────────────────────
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
    "visitors_thai",
    "visitors_foreign",
    "foreign_ratio",
    "revenue_total_mb",
    "revenue_per_visitor",
    "visitors_lag1",
    "visitors_lag3",
    "visitors_lag12",
    "visitors_roll3",
]
print(f"  ✅ Features: {len(FEATURE_COLS)} columns")

# ─────────────────────────────────────────────────────
#  STEP 4: TRAIN / TEST SPLIT
# ─────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("  STEP 4: Train/Test Split (2024 → Train, 2025 → Test)")
print("=" * 55)

df_model = df.dropna(subset=["visitors_lag1", "visitors_lag3"]).copy()
train_df = df_model[df_model["year"] == 2024].copy()
test_df = df_model[df_model["year"] == 2025].copy()
train_ag = train_df[FEATURE_COLS + [TARGET_COL]].copy()
test_ag = test_df[FEATURE_COLS + [TARGET_COL]].copy()

print(f"  Train (2024): {len(train_ag):,} records")
print(f"  Test  (2025): {len(test_ag):,} records")

# ─────────────────────────────────────────────────────
#  STEP 5: AUTOGLUON
# ─────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("  STEP 5: AutoGluon Training")
print("=" * 55)

try:
    from autogluon.tabular import TabularPredictor

    predictor = TabularPredictor(
        label=TARGET_COL,
        path=MODEL_DIR,
        eval_metric="root_mean_squared_error",
        problem_type="regression",
        verbosity=2,
    ).fit(
        train_data=train_ag,
        time_limit=TIME_LIMIT,
        presets=PRESET,
        excluded_model_types=["FASTAI"],
    )
    print("\n  ✅ Training complete!")

    # ── Evaluate ──
    print("\n" + "=" * 55)
    print("  STEP 6: Evaluation")
    print("=" * 55)
    y_pred = predictor.predict(test_ag.drop(columns=[TARGET_COL]))
    y_true = test_ag[TARGET_COL].values
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / np.where(y_true == 0, 1, y_true))) * 100
    print(f"  MAE  : {mae:>15,.0f} คน")
    print(f"  RMSE : {rmse:>15,.0f} คน")
    print(f"  R²   : {r2:>15.4f}")
    print(f"  MAPE : {mape:>14.2f}%")

    lb = predictor.leaderboard(test_ag, silent=True)
    print("\n  --- Leaderboard ---")
    print(
        lb[["model", "score_test", "score_val", "fit_time"]]
        .head(10)
        .to_string(index=False)
    )

    try:
        fi = predictor.feature_importance(test_ag)
        print("\n  --- Feature Importance (Top 15) ---")
        print(fi.head(15).to_string())
    except Exception:
        pass

    # ── Save outputs ──
    print("\n" + "=" * 55)
    print("  STEP 7: Saving Outputs")
    print("=" * 55)

    result_df = test_df[["date", "province", "region", TARGET_COL]].copy()
    result_df["predicted"] = y_pred.values
    result_df["error"] = result_df[TARGET_COL] - result_df["predicted"]
    result_df["error_pct"] = (
        result_df["error"] / result_df[TARGET_COL].replace(0, np.nan) * 100
    )
    result_df.to_csv("predictions.csv", index=False, encoding="utf-8")
    print("  ✅ predictions.csv")

    metrics = {
        "mae": round(mae, 2),
        "rmse": round(rmse, 2),
        "r2": round(r2, 4),
        "mape": round(mape, 2),
        "best_model": predictor.model_best,
        "target": TARGET_COL,
        "train_size": len(train_ag),
        "test_size": len(test_ag),
        "features": FEATURE_COLS,
    }
    with open("model_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print("  ✅ model_metrics.json")

    lb.to_csv("model_leaderboard.csv", index=False, encoding="utf-8")
    print("  ✅ model_leaderboard.csv")

    # Province/region code maps สำหรับ Dashboard
    with open("province_maps.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "province_code": dict(
                    zip(df["province"].astype(str), df["province_code"].astype(int))
                ),
                "region_code": dict(
                    zip(df["region"].astype(str), df["region_code"].astype(int))
                ),
                "province_region": dict(
                    zip(df["province"].astype(str), df["region"].astype(str))
                ),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print("  ✅ province_maps.json")
    print(f"  ✅ {MODEL_DIR}/")

    print(
        f"""
{'='*55}
  🏆 Best Model : {predictor.model_best}
  📊 R²         : {r2:.4f}
  📉 MAPE       : {mape:.2f}%
{'='*55}
  ขั้นตอนต่อไป:
    python app.py
    เปิดเบราว์เซอร์: http://127.0.0.1:8050
{'='*55}
    """
    )

except ImportError:
    print(
        """
  ⚠️  AutoGluon ยังไม่ได้ติดตั้ง!
  รันคำสั่งนี้ก่อน:
    pip install autogluon.tabular
  แล้วรัน train_model.py ใหม่
  """
    )
except Exception as e:
    import traceback

    print(f"\n  ❌ Error: {e}")
    traceback.print_exc()
