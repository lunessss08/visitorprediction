# """
# ╔══════════════════════════════════════════════════════════╗
# ║   🇹🇭  Thailand Tourism Forecast Dashboard               ║
# ║   พยากรณ์จำนวนนักท่องเที่ยวรายจังหวัด ปี 2567-2568     ║
# ║   Powered by AutoGluon + Dash                            ║
# ╚══════════════════════════════════════════════════════════╝

# วิธีติดตั้งและรัน:
#   pip install dash dash-bootstrap-components plotly pandas numpy scikit-learn autogluon.tabular
#   python app.py
#   เปิดเบราว์เซอร์ที่: http://127.0.0.1:8050
# """

import json
import warnings
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import dash
from dash import dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════
#   CONFIG & LOAD DATA
# ══════════════════════════════════════════════════════════
DATA_PATH  = "tourism_cleaned.csv"
MODEL_DIR  = "autogluon_model"
PRED_PATH  = "predictions.csv"

THAI_MONTHS = {
    1:"ม.ค.", 2:"ก.พ.", 3:"มี.ค.", 4:"เม.ย.",
    5:"พ.ค.", 6:"มิ.ย.", 7:"ก.ค.", 8:"ส.ค.",
    9:"ก.ย.", 10:"ต.ค.", 11:"พ.ย.", 12:"ธ.ค."
}

# Color palette (Thai-inspired)
COLORS = {
    "primary":    "#1A6B4A",   # deep green
    "secondary":  "#F4A826",   # golden yellow
    "accent":     "#C0392B",   # Thai red
    "bg":         "#0D1117",   # dark background
    "card":       "#161B22",   # card background
    "card2":      "#1C2333",   # lighter card
    "text":       "#E6EDF3",
    "muted":      "#8B949E",
    "border":     "#30363D",
    "thai":       "#F4A826",
    "foreign":    "#58A6FF",
    "revenue":    "#3FB950",
    "predict":    "#FF7B72",
}

# ── Load Data ──
df = pd.read_csv(DATA_PATH, encoding="utf-8-sig", parse_dates=["date"])
df["month_label"] = df["month"].map(THAI_MONTHS)

# ── Try load predictions ──
try:
    pred_df = pd.read_csv(PRED_PATH, encoding="utf-8-sig", parse_dates=["date"])
    HAS_PRED = True
except FileNotFoundError:
    HAS_PRED = False
    pred_df = pd.DataFrame()

# ── Try load model metrics ──
try:
    with open("model_metrics.json", encoding="utf-8") as f:
        metrics = json.load(f)
    HAS_METRICS = True
except FileNotFoundError:
    HAS_METRICS = False
    metrics = {}

# ── Try load AutoGluon model ──
try:
    from autogluon.tabular import TabularPredictor
    predictor = TabularPredictor.load(MODEL_DIR)
    HAS_MODEL = True
except Exception:
    HAS_MODEL = False
    predictor = None

# ── Precompute ──
provinces    = sorted(df["province"].unique())
regions      = sorted(df["region"].unique())
all_years    = sorted(df["year"].unique())
national_monthly = (
    df.groupby(["year","month"])[["visitors_total","visitors_thai","visitors_foreign","revenue_total_mb"]]
    .sum().reset_index()
)
national_monthly["month_label"] = national_monthly["month"].map(THAI_MONTHS)
national_monthly["date"] = pd.to_datetime(
    national_monthly[["year","month"]].assign(day=1)
)
province_annual = (
    df.groupby("province")[["visitors_total","visitors_foreign","revenue_total_mb"]]
    .sum().reset_index().sort_values("visitors_total", ascending=False)
)

# Province → code mapping for model input
prov_code_map = dict(zip(
    df["province"].astype("category").cat.categories,
    range(df["province"].astype("category").cat.codes.max()+1)
))
region_code_map = dict(zip(
    df["region"].astype("category").cat.categories,
    range(df["region"].astype("category").cat.codes.max()+1)
))

# ══════════════════════════════════════════════════════════
#   APP INIT
# ══════════════════════════════════════════════════════════
app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.CYBORG,
        "https://fonts.googleapis.com/css2?family=Sarabun:wght@300;400;600;700&family=Kanit:wght@400;600;700&display=swap",
    ],
    suppress_callback_exceptions=True,
    title="🇹🇭 Thailand Tourism Forecast",
)

# ══════════════════════════════════════════════════════════
#   CSS
# ══════════════════════════════════════════════════════════
CUSTOM_CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }

body {
    background: #0D1117;
    color: #E6EDF3;
    font-family: 'Sarabun', sans-serif;
    font-size: 15px;
}

/* ── Sidebar ── */
.sidebar {
    width: 240px;
    min-height: 100vh;
    background: #0D1117;
    border-right: 1px solid #30363D;
    position: fixed;
    top: 0; left: 0;
    z-index: 100;
    display: flex;
    flex-direction: column;
    padding: 0;
}
.sidebar-logo {
    padding: 24px 20px 20px;
    border-bottom: 1px solid #30363D;
    background: linear-gradient(135deg, #1A6B4A22, #0D1117);
}
.sidebar-logo h2 {
    font-family: 'Kanit', sans-serif;
    font-size: 1.1rem;
    font-weight: 700;
    color: #F4A826;
    line-height: 1.3;
    margin-top: 8px;
}
.sidebar-logo p { color: #8B949E; font-size: 0.75rem; margin-top: 4px; }

.nav-item {
    display: flex; align-items: center; gap: 10px;
    padding: 12px 20px;
    color: #8B949E;
    cursor: pointer;
    transition: all 0.2s;
    border-left: 3px solid transparent;
    font-size: 0.9rem;
    text-decoration: none;
}
.nav-item:hover { background: #161B22; color: #E6EDF3; border-left-color: #30363D; }
.nav-item.active { background: #1A6B4A22; color: #3FB950; border-left-color: #1A6B4A; font-weight: 600; }
.nav-icon { font-size: 1.1rem; width: 20px; text-align: center; }

/* ── Main content ── */
.main-content {
    margin-left: 240px;
    padding: 28px 32px;
    min-height: 100vh;
}

/* ── Page header ── */
.page-header {
    margin-bottom: 24px;
    padding-bottom: 16px;
    border-bottom: 1px solid #30363D;
}
.page-header h1 {
    font-family: 'Kanit', sans-serif;
    font-size: 1.6rem;
    font-weight: 700;
    color: #E6EDF3;
}
.page-header p { color: #8B949E; font-size: 0.88rem; margin-top: 4px; }

/* ── KPI Cards ── */
.kpi-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-bottom: 24px; }
.kpi-card {
    background: #161B22;
    border: 1px solid #30363D;
    border-radius: 10px;
    padding: 18px 20px;
    position: relative;
    overflow: hidden;
    transition: transform 0.2s, box-shadow 0.2s;
}
.kpi-card:hover { transform: translateY(-2px); box-shadow: 0 8px 24px #00000040; }
.kpi-card::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; height: 3px;
}
.kpi-card.green::before  { background: #1A6B4A; }
.kpi-card.yellow::before { background: #F4A826; }
.kpi-card.blue::before   { background: #58A6FF; }
.kpi-card.red::before    { background: #FF7B72; }

.kpi-label { color: #8B949E; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 8px; }
.kpi-value { font-family: 'Kanit', sans-serif; font-size: 1.7rem; font-weight: 700; color: #E6EDF3; line-height: 1; }
.kpi-sub   { color: #8B949E; font-size: 0.78rem; margin-top: 6px; }
.kpi-icon  { position: absolute; top: 16px; right: 16px; font-size: 1.6rem; opacity: 0.2; }

/* ── Chart Cards ── */
.chart-card {
    background: #161B22;
    border: 1px solid #30363D;
    border-radius: 10px;
    padding: 20px;
    margin-bottom: 20px;
}
.chart-title {
    font-family: 'Kanit', sans-serif;
    font-size: 1rem;
    font-weight: 600;
    color: #E6EDF3;
    margin-bottom: 4px;
}
.chart-subtitle { color: #8B949E; font-size: 0.8rem; margin-bottom: 16px; }
.chart-row { display: grid; gap: 20px; margin-bottom: 20px; }
.chart-row-2 { grid-template-columns: 1fr 1fr; }
.chart-row-3 { grid-template-columns: 1fr 1fr 1fr; }

/* ── Controls ── */
.control-card {
    background: #161B22;
    border: 1px solid #30363D;
    border-radius: 10px;
    padding: 20px;
    margin-bottom: 20px;
}
.control-label { color: #8B949E; font-size: 0.8rem; margin-bottom: 6px; font-weight: 600; letter-spacing: 0.04em; text-transform: uppercase; }
.control-row { display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; align-items: end; }

/* ── Dropdown styling ── */
.Select-control { background: #0D1117 !important; border-color: #30363D !important; color: #E6EDF3 !important; border-radius: 6px !important; }
.Select-menu-outer { background: #161B22 !important; border-color: #30363D !important; }
.Select-option { background: #161B22 !important; color: #E6EDF3 !important; }
.Select-option:hover, .Select-option.is-focused { background: #1C2333 !important; color: #F4A826 !important; }
.Select-value-label { color: #E6EDF3 !important; }
.Select-placeholder { color: #8B949E !important; }

/* ── Prediction result ── */
.pred-result {
    background: linear-gradient(135deg, #1A6B4A33, #1C2333);
    border: 1px solid #1A6B4A;
    border-radius: 10px;
    padding: 24px;
    text-align: center;
    margin-top: 16px;
}
.pred-number {
    font-family: 'Kanit', sans-serif;
    font-size: 3rem;
    font-weight: 700;
    color: #3FB950;
    line-height: 1;
}
.pred-label { color: #8B949E; font-size: 0.9rem; margin-top: 8px; }

/* ── Metric badges ── */
.metric-row { display: flex; gap: 12px; flex-wrap: wrap; margin-top: 16px; }
.metric-badge {
    background: #1C2333;
    border: 1px solid #30363D;
    border-radius: 8px;
    padding: 10px 16px;
    flex: 1;
    min-width: 100px;
    text-align: center;
}
.metric-badge .val { font-family: 'Kanit', sans-serif; font-size: 1.3rem; font-weight: 700; color: #F4A826; }
.metric-badge .lbl { color: #8B949E; font-size: 0.72rem; margin-top: 4px; text-transform: uppercase; }

/* ── Badge ── */
.tag {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.05em;
}
.tag-green  { background: #1A6B4A33; color: #3FB950; border: 1px solid #1A6B4A66; }
.tag-yellow { background: #F4A82633; color: #F4A826; border: 1px solid #F4A82666; }
.tag-blue   { background: #58A6FF33; color: #58A6FF; border: 1px solid #58A6FF66; }
.tag-red    { background: #FF7B7233; color: #FF7B72; border: 1px solid #FF7B7266; }

/* ── Table ── */
.data-table { width: 100%; border-collapse: collapse; font-size: 0.85rem; }
.data-table th { color: #8B949E; font-weight: 600; text-transform: uppercase; font-size: 0.72rem; letter-spacing: 0.05em; padding: 8px 12px; border-bottom: 1px solid #30363D; text-align: left; }
.data-table td { padding: 10px 12px; border-bottom: 1px solid #1C2333; color: #E6EDF3; }
.data-table tr:hover td { background: #1C2333; }

/* ── Slider ── */
.rc-slider-track { background-color: #1A6B4A !important; }
.rc-slider-handle { border-color: #1A6B4A !important; background: #1A6B4A !important; }

/* ── Input ── */
.dash-input {
    background: #0D1117 !important;
    border: 1px solid #30363D !important;
    border-radius: 6px !important;
    color: #E6EDF3 !important;
    padding: 8px 12px !important;
    width: 100%;
    font-family: 'Sarabun', sans-serif;
}
.dash-input:focus { border-color: #1A6B4A !important; outline: none !important; }

/* ── Button ── */
.btn-primary {
    background: #1A6B4A;
    border: none;
    border-radius: 8px;
    color: white;
    padding: 10px 24px;
    font-family: 'Kanit', sans-serif;
    font-size: 0.95rem;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.2s;
    width: 100%;
}
.btn-primary:hover { background: #228B5E; transform: translateY(-1px); box-shadow: 0 4px 12px #1A6B4A44; }

/* ── Tabs ── */
.module-tabs { display: flex; gap: 0; margin-bottom: 24px; border-bottom: 1px solid #30363D; }
.module-tab {
    padding: 12px 24px;
    cursor: pointer;
    color: #8B949E;
    font-family: 'Kanit', sans-serif;
    font-size: 0.9rem;
    font-weight: 600;
    border-bottom: 2px solid transparent;
    transition: all 0.2s;
}
.module-tab:hover { color: #E6EDF3; }
.module-tab.active { color: #3FB950; border-bottom-color: #1A6B4A; }

/* ── Status dot ── */
.status-dot { width: 8px; height: 8px; border-radius: 50%; display: inline-block; margin-right: 6px; }
.status-dot.online  { background: #3FB950; box-shadow: 0 0 6px #3FB950; }
.status-dot.offline { background: #F4A826; box-shadow: 0 0 6px #F4A826; }
"""

# ══════════════════════════════════════════════════════════
#   HELPER: Plotly theme
# ══════════════════════════════════════════════════════════
def dark_theme(fig, height=320):
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Sarabun, sans-serif", color="#8B949E", size=11),
        margin=dict(l=0, r=0, t=10, b=0),
        legend=dict(bgcolor="rgba(0,0,0,0)", font_size=11),
        xaxis=dict(gridcolor="#1C2333", linecolor="#30363D", tickcolor="#30363D"),
        yaxis=dict(gridcolor="#1C2333", linecolor="#30363D", tickcolor="#30363D"),
    )
    return fig

# ══════════════════════════════════════════════════════════
#   LAYOUT HELPERS
# ══════════════════════════════════════════════════════════
def kpi_card(label, value, sub, color="green", icon=""):
    return html.Div([
        html.Div(icon, className="kpi-icon"),
        html.Div(label, className="kpi-label"),
        html.Div(value, className="kpi-value"),
        html.Div(sub,   className="kpi-sub"),
    ], className=f"kpi-card {color}")

def chart_card(title, subtitle, children, extra=None):
    return html.Div([
        html.Div([
            html.Div(title, className="chart-title"),
            html.Div(subtitle, className="chart-subtitle"),
            *(extra or []),
        ]),
        *children,
    ], className="chart-card")

def section_header(title, subtitle=""):
    return html.Div([
        html.H1(title, style={"fontFamily":"Kanit,sans-serif","fontSize":"1.5rem","fontWeight":"700","color":"#E6EDF3"}),
        html.P(subtitle, style={"color":"#8B949E","fontSize":"0.85rem","marginTop":"4px"}),
    ], className="page-header")

# ══════════════════════════════════════════════════════════
#   SIDEBAR
# ══════════════════════════════════════════════════════════
sidebar = html.Div([
    html.Div([
        html.Div("🇹🇭", style={"fontSize":"2rem"}),
        html.H2("Tourism\nForecast"),
        html.P("Dashboard 2567-2568"),
    ], className="sidebar-logo"),

    html.Div([
        html.A([html.Span("📊", className="nav-icon"), "ภาพรวมประเทศ"],   href="#", id="nav-1", className="nav-item active"),
        html.A([html.Span("🤖", className="nav-icon"), "พยากรณ์ AI"],     href="#", id="nav-2", className="nav-item"),
        html.A([html.Span("🎛️", className="nav-icon"), "ทดสอบพยากรณ์"],   href="#", id="nav-3", className="nav-item"),
        html.A([html.Span("🗺️", className="nav-icon"), "วิเคราะห์ภาค"],   href="#", id="nav-4", className="nav-item"),
    ], style={"paddingTop":"8px", "flex":"1"}),

    html.Div([
        html.Div([
            html.Span(className=f"status-dot {'online' if HAS_MODEL else 'offline'}"),
            html.Span("Model " + ("Loaded" if HAS_MODEL else "Not Found"), style={"fontSize":"0.75rem","color":"#8B949E"}),
        ], style={"display":"flex","alignItems":"center","padding":"16px 20px","borderTop":"1px solid #30363D"}),
    ]),
], className="sidebar")

# ══════════════════════════════════════════════════════════
#   MODULE 1: ภาพรวมประเทศ
# ══════════════════════════════════════════════════════════
# KPIs
total_2024 = df[df["year"]==2024]["visitors_total"].sum()
total_2025 = df[df["year"]==2025]["visitors_total"].sum()
total_rev  = df["revenue_total_mb"].sum()
avg_occ    = df["occupancy_rate"].mean()
top_prov   = province_annual.iloc[0]["province"]

def build_module1():
    return html.Div([
        section_header("📊 ภาพรวมการท่องเที่ยวประเทศไทย",
                       "สถิติรายจังหวัด ปี 2567-2568 | ข้อมูล: กระทรวงการท่องเที่ยวและกีฬา"),

        # KPIs
        html.Div([
            kpi_card("นักท่องเที่ยว 2567", f"{total_2024/1e6:.1f}M", "คน (รวมทุกจังหวัด)", "green", "🧳"),
            kpi_card("นักท่องเที่ยว 2568", f"{total_2025/1e6:.1f}M", "คน (รวมทุกจังหวัด)", "yellow", "✈️"),
            kpi_card("รายได้รวม", f"฿{total_rev/1e6:.1f}T", "ล้านบาท (2 ปีรวม)", "blue", "💰"),
            kpi_card("Occupancy เฉลี่ย", f"{avg_occ:.1f}%", f"จังหวัดนำ: {top_prov}", "red", "🏨"),
        ], className="kpi-grid"),

        # Chart row 1: National trend + Bar chart
        html.Div([
            chart_card("📈 แนวโน้มนักท่องเที่ยวรายเดือน", "เปรียบเทียบปี 2567 vs 2568", [
                dcc.Graph(id="chart-national-trend", config={"displayModeBar":False}),
            ]),
            chart_card("🏆 Top 10 จังหวัดยอดนิยม", "ยอดรวมตลอด 2 ปี", [
                dcc.Graph(id="chart-top10", config={"displayModeBar":False}),
            ]),
        ], className="chart-row chart-row-2"),

        # Chart row 2: Heatmap
        chart_card("🌡️ Heatmap นักท่องเที่ยว Top 15 จังหวัด", "แสดงความหนาแน่นรายเดือน ปี 2567", [
            dcc.Graph(id="chart-heatmap", config={"displayModeBar":False}),
        ]),

        # Chart row 3: Thai vs Foreign + Revenue
        html.Div([
            chart_card("🧑‍🤝‍🧑 ไทย vs ต่างชาติ", "สัดส่วนนักท่องเที่ยวรายเดือน 2567", [
                dcc.Graph(id="chart-thai-foreign", config={"displayModeBar":False}),
            ]),
            chart_card("💵 รายได้รายภาค", "รายได้รวมแยกตามภาค (ล้านบาท)", [
                dcc.Graph(id="chart-region-revenue", config={"displayModeBar":False}),
            ]),
        ], className="chart-row chart-row-2"),
    ])

# ══════════════════════════════════════════════════════════
#   MODULE 2: ผลพยากรณ์ AI
# ══════════════════════════════════════════════════════════
def build_module2():
    # model metrics display
    if HAS_METRICS:
        metric_section = html.Div([
            html.Div([
                html.Div(f"{metrics.get('r2','-')}", className="val"),
                html.Div("R² Score", className="lbl"),
            ], className="metric-badge"),
            html.Div([
                html.Div(f"{metrics.get('mape','-')}%", className="val"),
                html.Div("MAPE", className="lbl"),
            ], className="metric-badge"),
            html.Div([
                html.Div(f"{int(metrics.get('mae',0)):,}", className="val"),
                html.Div("MAE (คน)", className="lbl"),
            ], className="metric-badge"),
            html.Div([
                html.Div(f"{int(metrics.get('rmse',0)):,}", className="val"),
                html.Div("RMSE (คน)", className="lbl"),
            ], className="metric-badge"),
            html.Div([
                html.Div(metrics.get('best_model','-'), className="val", style={"fontSize":"0.9rem"}),
                html.Div("Best Model", className="lbl"),
            ], className="metric-badge"),
        ], className="metric-row")
    else:
        metric_section = html.Div([
            html.Span("⚠️ ยังไม่มีไฟล์ model_metrics.json — รัน train_model.py ก่อนนะครับ",
                      style={"color":"#F4A826","fontSize":"0.85rem"})
        ])

    if HAS_PRED:
        pred_content = [
            html.Div([
                chart_card("🎯 Actual vs Predicted (2568)", "เปรียบเทียบค่าจริงกับค่าพยากรณ์รายจังหวัด", [
                    html.Div([
                        html.Div([
                            html.Label("เลือกจังหวัด", className="control-label"),
                            dcc.Dropdown(
                                id="pred-province-select",
                                options=[{"label":p,"value":p} for p in sorted(pred_df["province"].unique())],
                                value="กรุงเทพมหานคร",
                                style={"background":"#0D1117"},
                                className="dark-dropdown",
                            ),
                        ], style={"width":"240px","marginBottom":"16px"}),
                    ]),
                    dcc.Graph(id="chart-actual-vs-pred", config={"displayModeBar":False}),
                ]),
                chart_card("🗺️ แผนที่ Error % รายจังหวัด", "ค่า MAPE เฉลี่ยแต่ละจังหวัด", [
                    dcc.Graph(id="chart-error-bar", config={"displayModeBar":False}),
                ]),
            ], className="chart-row chart-row-2"),

            chart_card("📉 Scatter: Actual vs Predicted (ทุกจังหวัด)", "ยิ่งใกล้เส้น diagonal ยิ่งแม่น", [
                dcc.Graph(id="chart-scatter-pred", config={"displayModeBar":False}),
            ]),
        ]
    else:
        pred_content = [
            html.Div([
                html.P("⚠️ ยังไม่มีไฟล์ predictions.csv", style={"color":"#F4A826","textAlign":"center","padding":"40px"}),
                html.P("รัน train_model.py ก่อน จะได้ข้อมูลพยากรณ์สำหรับแสดงผลตรงนี้ครับ",
                       style={"color":"#8B949E","textAlign":"center"}),
            ], className="chart-card"),
        ]

    return html.Div([
        section_header("🤖 ผลการพยากรณ์ด้วย AutoGluon",
                       "เปรียบเทียบค่าจริงกับผลพยากรณ์ | Train: 2567 | Test: 2568"),
        html.Div(className="chart-card", children=[
            html.Div("📐 Model Performance Metrics", className="chart-title"),
            html.Div("ประเมินความแม่นยำของโมเดลบนชุดข้อมูล Test (ปี 2568)", className="chart-subtitle"),
            metric_section,
        ]),
        *pred_content,
    ])

# ══════════════════════════════════════════════════════════
#   MODULE 3: Interactive Prediction
# ══════════════════════════════════════════════════════════
def build_module3():
    return html.Div([
        section_header("🎛️ ทดสอบพยากรณ์แบบ Interactive",
                       "ปรับพารามิเตอร์แล้วกด 'พยากรณ์' เพื่อดูผลทันที"),

        html.Div([
            html.Div("⚙️ กรอกพารามิเตอร์สำหรับพยากรณ์", className="chart-title"),
            html.Div("ปรับค่าด้านล่าง แล้วกดปุ่มพยากรณ์", className="chart-subtitle"),

            html.Div([
                # Row 1
                html.Div([
                    html.Div([
                        html.Label("จังหวัด", className="control-label"),
                        dcc.Dropdown(
                            id="inp-province",
                            options=[{"label":p,"value":p} for p in provinces],
                            value="กรุงเทพมหานคร",
                            style={"background":"#0D1117"},
                        ),
                    ]),
                    html.Div([
                        html.Label("ปี (พ.ศ.)", className="control-label"),
                        dcc.Dropdown(
                            id="inp-year",
                            options=[{"label":str(y+543),"value":y} for y in [2024,2025,2026]],
                            value=2025,
                            style={"background":"#0D1117"},
                        ),
                    ]),
                    html.Div([
                        html.Label("เดือน", className="control-label"),
                        dcc.Dropdown(
                            id="inp-month",
                            options=[{"label":v,"value":k} for k,v in THAI_MONTHS.items()],
                            value=1,
                            style={"background":"#0D1117"},
                        ),
                    ]),
                    html.Div([
                        html.Label("Occupancy Rate (%)", className="control-label"),
                        dcc.Input(id="inp-occ", type="number", value=70, min=0, max=100,
                                  className="dash-input"),
                    ]),
                ], className="control-row"),

                html.Div(style={"height":"14px"}),

                # Row 2
                html.Div([
                    html.Div([
                        html.Label("จำนวนนักท่องเที่ยวไทย (คน)", className="control-label"),
                        dcc.Input(id="inp-thai", type="number", value=500000,
                                  className="dash-input"),
                    ]),
                    html.Div([
                        html.Label("จำนวนนักท่องเที่ยวต่างชาติ (คน)", className="control-label"),
                        dcc.Input(id="inp-foreign", type="number", value=200000,
                                  className="dash-input"),
                    ]),
                    html.Div([
                        html.Label("รายได้รวม (ล้านบาท)", className="control-label"),
                        dcc.Input(id="inp-revenue", type="number", value=5000,
                                  className="dash-input"),
                    ]),
                    html.Div([
                        html.Label("Visitors เดือนก่อน (lag1)", className="control-label"),
                        dcc.Input(id="inp-lag1", type="number", value=600000,
                                  className="dash-input"),
                    ]),
                ], className="control-row"),

                html.Div(style={"height":"20px"}),
                html.Button("🔮 พยากรณ์จำนวนนักท่องเที่ยว", id="btn-predict",
                            className="btn-primary", n_clicks=0),
            ]),
        ], className="chart-card"),

        # Result
        html.Div(id="pred-result-container"),

        # Province history chart
        html.Div(style={"height":"20px"}),
        chart_card("📈 ประวัตินักท่องเที่ยวของจังหวัดที่เลือก", "ข้อมูลรายเดือนย้อนหลัง", [
            dcc.Graph(id="chart-province-history", config={"displayModeBar":False}),
        ]),
    ])

# ══════════════════════════════════════════════════════════
#   MODULE 4: วิเคราะห์ภาค (โมดูลออกแบบเอง)
# ══════════════════════════════════════════════════════════
def build_module4():
    return html.Div([
        section_header("🗺️ วิเคราะห์เปรียบเทียบรายภาค",
                       "โมดูลออกแบบเพิ่มเติม: เปรียบเทียบสมรรถนะการท่องเที่ยวแต่ละภาค"),

        # KPI by region
        html.Div(id="region-kpis", style={"marginBottom":"20px"}),

        html.Div([
            chart_card("📊 เปรียบเทียบนักท่องเที่ยวรายภาค", "แยกไทย vs ต่างชาติ", [
                dcc.Graph(id="chart-region-compare", config={"displayModeBar":False}),
            ]),
            chart_card("💹 รายได้ต่อนักท่องเที่ยว (บาท/คน)", "ภาคไหนสร้างรายได้ต่อหัวสูงสุด", [
                dcc.Graph(id="chart-rev-per-visitor", config={"displayModeBar":False}),
            ]),
        ], className="chart-row chart-row-2"),

        chart_card("📅 ซีซั่นของแต่ละภาค", "เดือนที่นักท่องเที่ยวสูงสุด-ต่ำสุดของแต่ละภาค (2567)", [
            html.Div([
                html.Label("เลือกภาค", className="control-label"),
                dcc.Dropdown(
                    id="sel-region",
                    options=[{"label":r,"value":r} for r in regions],
                    value=regions[0],
                    style={"background":"#0D1117","width":"300px","marginBottom":"16px"},
                ),
            ]),
            dcc.Graph(id="chart-region-season", config={"displayModeBar":False}),
        ]),

        chart_card("🔥 Bubble Chart: Occupancy vs Revenue vs Visitors", "ขนาด bubble = จำนวนผู้เยี่ยมเยือน", [
            html.Div([
                html.Label("เลือกปี", className="control-label"),
                dcc.RadioItems(
                    id="sel-bubble-year",
                    options=[{"label":"2567","value":2024},{"label":"2568","value":2025}],
                    value=2024,
                    inline=True,
                    style={"color":"#8B949E","marginBottom":"16px","fontSize":"0.88rem"},
                ),
            ]),
            dcc.Graph(id="chart-bubble", config={"displayModeBar":False}),
        ]),
    ])

# ══════════════════════════════════════════════════════════
#   MAIN LAYOUT
# ══════════════════════════════════════════════════════════
app.layout = html.Div([
    html.Style(CUSTOM_CSS),
    dcc.Store(id="active-module", data="1"),
    sidebar,
    html.Div([
        html.Div(id="page-content"),
    ], className="main-content"),
])

# ══════════════════════════════════════════════════════════
#   CALLBACKS — Navigation
# ══════════════════════════════════════════════════════════
@app.callback(
    Output("page-content", "children"),
    Output("active-module", "data"),
    Output("nav-1", "className"),
    Output("nav-2", "className"),
    Output("nav-3", "className"),
    Output("nav-4", "className"),
    Input("nav-1", "n_clicks"),
    Input("nav-2", "n_clicks"),
    Input("nav-3", "n_clicks"),
    Input("nav-4", "n_clicks"),
    State("active-module", "data"),
    prevent_initial_call=False,
)
def navigate(n1, n2, n3, n4, current):
    ctx = callback_context
    if not ctx.triggered or ctx.triggered[0]["prop_id"] == ".":
        module = "1"
    else:
        btn = ctx.triggered[0]["prop_id"].split(".")[0]
        module = btn.split("-")[1]

    builders = {"1": build_module1, "2": build_module2, "3": build_module3, "4": build_module4}
    content = builders[module]()
    classes = {str(i): "nav-item active" if str(i)==module else "nav-item" for i in range(1,5)}
    return content, module, classes["1"], classes["2"], classes["3"], classes["4"]

# ══════════════════════════════════════════════════════════
#   CALLBACKS — Module 1 Charts
# ══════════════════════════════════════════════════════════
@app.callback(Output("chart-national-trend","figure"), Input("active-module","data"))
def chart_trend(m):
    fig = go.Figure()
    colors_year = {2024: COLORS["primary"], 2025: COLORS["secondary"]}
    for yr in [2024, 2025]:
        d = national_monthly[national_monthly["year"]==yr].sort_values("month")
        fig.add_trace(go.Scatter(
            x=d["month_label"], y=d["visitors_total"]/1e6,
            name=f"ปี {yr+543}", mode="lines+markers",
            line=dict(color=colors_year[yr], width=2.5),
            marker=dict(size=6),
            fill="tozeroy", fillcolor=colors_year[yr]+"22",
        ))
    fig.update_layout(yaxis_title="ล้านคน", xaxis_title="")
    return dark_theme(fig)

@app.callback(Output("chart-top10","figure"), Input("active-module","data"))
def chart_top10(m):
    top = province_annual.head(10)
    fig = go.Figure(go.Bar(
        x=top["visitors_total"]/1e6,
        y=top["province"],
        orientation="h",
        marker=dict(
            color=top["visitors_total"]/1e6,
            colorscale=[[0,"#0D4A30"],[0.5,"#1A6B4A"],[1,"#3FB950"]],
        ),
        text=[f"{v/1e6:.1f}M" for v in top["visitors_total"]],
        textposition="outside",
        textfont=dict(color="#E6EDF3", size=11),
    ))
    fig.update_layout(yaxis=dict(autorange="reversed"), xaxis_title="ล้านคน")
    return dark_theme(fig)

@app.callback(Output("chart-heatmap","figure"), Input("active-module","data"))
def chart_heatmap(m):
    top15 = province_annual.head(15)["province"].tolist()
    d = df[(df["year"]==2024) & (df["province"].isin(top15))]
    pivot = d.pivot_table(index="province", columns="month", values="visitors_total", aggfunc="sum")
    pivot.columns = [THAI_MONTHS[c] for c in pivot.columns]
    fig = go.Figure(go.Heatmap(
        z=pivot.values/1e6,
        x=pivot.columns.tolist(),
        y=pivot.index.tolist(),
        colorscale=[[0,"#0D1117"],[0.3,"#0D4A30"],[0.7,"#1A6B4A"],[1,"#3FB950"]],
        text=np.round(pivot.values/1e6,1),
        texttemplate="%{text}",
        textfont=dict(size=10),
        hovertemplate="จังหวัด: %{y}<br>เดือน: %{x}<br>%{z:.2f}M คน<extra></extra>",
    ))
    return dark_theme(fig, height=380)

@app.callback(Output("chart-thai-foreign","figure"), Input("active-module","data"))
def chart_thai_foreign(m):
    d = national_monthly[national_monthly["year"]==2024].sort_values("month")
    fig = go.Figure()
    fig.add_trace(go.Bar(name="คนไทย", x=d["month_label"], y=d["visitors_thai"]/1e6,
                         marker_color=COLORS["thai"]+"cc"))
    fig.add_trace(go.Bar(name="ต่างชาติ", x=d["month_label"], y=d["visitors_foreign"]/1e6,
                         marker_color=COLORS["foreign"]+"cc"))
    fig.update_layout(barmode="stack", yaxis_title="ล้านคน", xaxis_title="")
    return dark_theme(fig)

@app.callback(Output("chart-region-revenue","figure"), Input("active-module","data"))
def chart_region_revenue(m):
    d = df.groupby("region")["revenue_total_mb"].sum().reset_index().sort_values("revenue_total_mb")
    colors = [COLORS["primary"],COLORS["secondary"],COLORS["accent"],COLORS["foreign"],COLORS["revenue"],"#A371F7"]
    fig = go.Figure(go.Bar(
        x=d["revenue_total_mb"]/1e3,
        y=d["region"],
        orientation="h",
        marker_color=colors[:len(d)],
        text=[f"฿{v/1e3:.0f}B" for v in d["revenue_total_mb"]],
        textposition="outside",
        textfont=dict(color="#E6EDF3",size=11),
    ))
    fig.update_layout(xaxis_title="พันล้านบาท")
    return dark_theme(fig)

# ══════════════════════════════════════════════════════════
#   CALLBACKS — Module 2 Charts
# ══════════════════════════════════════════════════════════
@app.callback(Output("chart-actual-vs-pred","figure"),
              Input("pred-province-select","value"),
              prevent_initial_call=False)
def chart_actual_vs_pred(province):
    if not HAS_PRED or province is None:
        return go.Figure()
    d = pred_df[pred_df["province"]==province].sort_values("date")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d["date"], y=d["visitors_total"], name="Actual",
                             line=dict(color=COLORS["secondary"], width=2.5),
                             mode="lines+markers", marker_size=6))
    fig.add_trace(go.Scatter(x=d["date"], y=d["predicted"], name="Predicted",
                             line=dict(color=COLORS["predict"], width=2, dash="dash"),
                             mode="lines+markers", marker_size=6))
    fig.update_layout(yaxis_title="คน", xaxis_title="")
    return dark_theme(fig)

@app.callback(Output("chart-error-bar","figure"), Input("active-module","data"))
def chart_error_bar(m):
    if not HAS_PRED:
        return go.Figure()
    d = pred_df.copy()
    d["abs_err_pct"] = np.abs(d["error_pct"])
    err = d.groupby("province")["abs_err_pct"].mean().reset_index().sort_values("abs_err_pct").head(20)
    fig = go.Figure(go.Bar(
        x=err["abs_err_pct"], y=err["province"], orientation="h",
        marker=dict(color=err["abs_err_pct"],
                    colorscale=[[0,"#1A6B4A"],[0.5,"#F4A826"],[1,"#C0392B"]]),
        text=[f"{v:.1f}%" for v in err["abs_err_pct"]],
        textposition="outside", textfont=dict(color="#E6EDF3", size=10),
    ))
    fig.update_layout(xaxis_title="MAPE (%)", yaxis=dict(autorange="reversed"))
    return dark_theme(fig, height=380)

@app.callback(Output("chart-scatter-pred","figure"), Input("active-module","data"))
def chart_scatter(m):
    if not HAS_PRED:
        return go.Figure()
    d = pred_df.copy()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=d["visitors_total"]/1e6, y=d["predicted"]/1e6,
        mode="markers",
        marker=dict(color=COLORS["foreign"], opacity=0.6, size=7),
        text=d["province"], name="จังหวัด",
        hovertemplate="<b>%{text}</b><br>Actual: %{x:.2f}M<br>Pred: %{y:.2f}M<extra></extra>",
    ))
    mx = max(d["visitors_total"].max(), d["predicted"].max()) / 1e6
    fig.add_trace(go.Scatter(x=[0,mx], y=[0,mx], mode="lines",
                             line=dict(color="#FF7B72", dash="dash", width=1.5), name="Perfect Fit"))
    fig.update_layout(xaxis_title="Actual (ล้านคน)", yaxis_title="Predicted (ล้านคน)")
    return dark_theme(fig, height=340)

# ══════════════════════════════════════════════════════════
#   CALLBACKS — Module 3 Interactive Predict
# ══════════════════════════════════════════════════════════
@app.callback(
    Output("pred-result-container","children"),
    Output("chart-province-history","figure"),
    Input("btn-predict","n_clicks"),
    Input("inp-province","value"),
    State("inp-year","value"),
    State("inp-month","value"),
    State("inp-occ","value"),
    State("inp-thai","value"),
    State("inp-foreign","value"),
    State("inp-revenue","value"),
    State("inp-lag1","value"),
    prevent_initial_call=False,
)
def predict_interactive(n_clicks, province, year, month, occ, thai, foreign, revenue, lag1):
    # Province history chart always shown
    prov_d = df[df["province"]==province].sort_values("date") if province else pd.DataFrame()
    hist_fig = go.Figure()
    if not prov_d.empty:
        hist_fig.add_trace(go.Scatter(
            x=prov_d["date"], y=prov_d["visitors_total"]/1e6,
            name="ผู้เยี่ยมเยือนทั้งหมด",
            fill="tozeroy", fillcolor=COLORS["primary"]+"33",
            line=dict(color=COLORS["primary"], width=2.5),
        ))
        hist_fig.add_trace(go.Scatter(
            x=prov_d["date"], y=prov_d["visitors_foreign"]/1e6,
            name="ต่างชาติ",
            line=dict(color=COLORS["foreign"], width=2, dash="dot"),
        ))
    hist_fig.update_layout(yaxis_title="ล้านคน", xaxis_title="")
    dark_theme(hist_fig)

    # Prediction only on button click
    ctx = callback_context
    triggered = ctx.triggered[0]["prop_id"] if ctx.triggered else ""

    if "btn-predict" not in triggered or not n_clicks:
        return html.Div(), hist_fig

    if not HAS_MODEL:
        result = html.Div([
            html.Div([
                html.P("⚠️ โมเดลยังไม่ได้ load", style={"color":"#F4A826","fontSize":"1rem"}),
                html.P("กรุณารัน train_model.py ก่อน จะได้โฟลเดอร์ autogluon_model/",
                       style={"color":"#8B949E","marginTop":"8px"}),
            ], className="pred-result"),
        ])
        return result, hist_fig

    try:
        month_sin = np.sin(2 * np.pi * month / 12)
        month_cos = np.cos(2 * np.pi * month / 12)
        is_high   = 1 if month in [11,12,1,2,3] else 0
        total_v   = (thai or 0) + (foreign or 0)
        f_ratio   = (foreign / total_v) if total_v > 0 else 0
        rev_per   = ((revenue or 0) * 1_000_000 / total_v) if total_v > 0 else 0
        pcode     = prov_code_map.get(province, 0)
        rcode     = region_code_map.get(df[df["province"]==province]["region"].iloc[0] if province else "ภาคกลาง", 0)
        lag3      = df[df["province"]==province]["visitors_total"].mean() if province else lag1
        lag12     = lag1
        roll3     = lag1

        input_df = pd.DataFrame([{
            "year": year, "month": month, "quarter": (month-1)//3+1,
            "month_sin": month_sin, "month_cos": month_cos,
            "is_high_season": is_high,
            "province_code": pcode, "region_code": rcode,
            "occupancy_rate": occ or 70,
            "checkin_count": int(total_v * 0.25),
            "visitors_thai": thai or 0,
            "visitors_foreign": foreign or 0,
            "foreign_ratio": f_ratio,
            "revenue_total_mb": revenue or 0,
            "revenue_per_visitor": rev_per,
            "visitors_lag1": lag1 or 0,
            "visitors_lag3": lag3,
            "visitors_lag12": lag12,
            "visitors_roll3": roll3,
        }])

        pred_val = predictor.predict(input_df)[0]
        month_th = THAI_MONTHS.get(month, "")

        result = html.Div([
            html.Div([
                html.Div(f"🎯 ผลการพยากรณ์", style={"color":"#8B949E","fontSize":"0.85rem","marginBottom":"8px"}),
                html.Div(f"{int(pred_val):,}", className="pred-number"),
                html.Div("คน", className="pred-label"),
                html.Div([
                    html.Span(f"{province}", className="tag tag-green", style={"marginRight":"8px"}),
                    html.Span(f"{month_th} ปี {year+543}", className="tag tag-yellow"),
                ], style={"marginTop":"12px"}),
            ], className="pred-result"),
        ])
    except Exception as e:
        result = html.Div([
            html.Div([
                html.P(f"❌ Error: {str(e)}", style={"color":"#FF7B72"}),
            ], className="pred-result"),
        ])

    return result, hist_fig

# ══════════════════════════════════════════════════════════
#   CALLBACKS — Module 4 Region Analysis
# ══════════════════════════════════════════════════════════
@app.callback(Output("region-kpis","children"), Input("active-module","data"))
def region_kpis(m):
    region_stats = df.groupby("region").agg(
        total=("visitors_total","sum"),
        foreign=("visitors_foreign","sum"),
        revenue=("revenue_total_mb","sum"),
    ).reset_index()
    cards = []
    region_colors = ["green","yellow","blue","red","green","yellow"]
    for i, row in enumerate(region_stats.itertuples()):
        cards.append(kpi_card(
            row.region,
            f"{row.total/1e6:.1f}M",
            f"ต่างชาติ {row.foreign/row.total*100:.0f}% | ฿{row.revenue/1e3:.0f}B",
            region_colors[i % len(region_colors)],
        ))
    return html.Div(cards, style={"display":"grid","gridTemplateColumns":"repeat(3,1fr)","gap":"16px"})

@app.callback(Output("chart-region-compare","figure"), Input("active-module","data"))
def chart_region_compare(m):
    d = df.groupby("region")[["visitors_thai","visitors_foreign"]].sum().reset_index()
    d = d.sort_values("visitors_thai", ascending=False)
    fig = go.Figure()
    fig.add_trace(go.Bar(name="คนไทย", x=d["region"], y=d["visitors_thai"]/1e6,
                         marker_color=COLORS["thai"]+"cc"))
    fig.add_trace(go.Bar(name="ต่างชาติ", x=d["region"], y=d["visitors_foreign"]/1e6,
                         marker_color=COLORS["foreign"]+"cc"))
    fig.update_layout(barmode="group", yaxis_title="ล้านคน", xaxis_title="")
    return dark_theme(fig)

@app.callback(Output("chart-rev-per-visitor","figure"), Input("active-module","data"))
def chart_rev_pv(m):
    d = df.groupby("region").agg(rev=("revenue_total_mb","sum"), v=("visitors_total","sum")).reset_index()
    d["rev_per"] = d["rev"] * 1e6 / d["v"]
    d = d.sort_values("rev_per", ascending=True)
    fig = go.Figure(go.Bar(
        x=d["rev_per"], y=d["region"], orientation="h",
        marker=dict(color=d["rev_per"],
                    colorscale=[[0,"#0D4A30"],[0.5,"#F4A826"],[1,"#FF7B72"]]),
        text=[f"฿{v:,.0f}" for v in d["rev_per"]],
        textposition="outside", textfont=dict(color="#E6EDF3",size=11),
    ))
    fig.update_layout(xaxis_title="บาท/คน")
    return dark_theme(fig)

@app.callback(Output("chart-region-season","figure"), Input("sel-region","value"))
def chart_region_season(region):
    d = df[(df["region"]==region) & (df["year"]==2024)]
    d = d.groupby("month")["visitors_total"].sum().reset_index()
    d["month_label"] = d["month"].map(THAI_MONTHS)
    max_m = d.loc[d["visitors_total"].idxmax(),"month_label"]
    fig = go.Figure(go.Bar(
        x=d["month_label"], y=d["visitors_total"]/1e6,
        marker=dict(
            color=[COLORS["secondary"] if ml==max_m else COLORS["primary"] for ml in d["month_label"]],
        ),
        text=[f"{v/1e6:.2f}M" for v in d["visitors_total"]],
        textposition="outside", textfont=dict(color="#E6EDF3",size=11),
    ))
    fig.update_layout(yaxis_title="ล้านคน", xaxis_title="")
    return dark_theme(fig)

@app.callback(Output("chart-bubble","figure"), Input("sel-bubble-year","value"))
def chart_bubble(year):
    d = df[df["year"]==year].groupby("province").agg(
        occ=("occupancy_rate","mean"),
        rev=("revenue_total_mb","sum"),
        vis=("visitors_total","sum"),
        region=("region","first"),
    ).reset_index()
    fig = px.scatter(
        d, x="occ", y="rev", size="vis", color="region",
        text="province", size_max=50,
        color_discrete_map={
            "กรุงเทพฯ":         COLORS["secondary"],
            "ภาคกลาง":          COLORS["primary"],
            "ภาคตะวันออก":      COLORS["foreign"],
            "ภาคใต้":           COLORS["accent"],
            "ภาคเหนือ":         COLORS["revenue"],
            "ภาคตะวันออกเฉียงเหนือ": "#A371F7",
        },
        hover_data={"vis":":,","occ":":.1f","rev":":.0f"},
        labels={"occ":"Occupancy Rate (%)","rev":"รายได้รวม (ล้านบาท)","vis":"นักท่องเที่ยว"},
    )
    fig.update_traces(textposition="top center", textfont_size=9)
    return dark_theme(fig, height=420)

# ══════════════════════════════════════════════════════════
#   RUN
# ══════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 55)
    print("  🇹🇭 Thailand Tourism Forecast Dashboard")
    print("  เปิดเบราว์เซอร์ที่: http://127.0.0.1:8050")
    print(f"  AutoGluon Model : {'✅ Loaded' if HAS_MODEL else '⚠️  Not found (run train_model.py first)'}")
    print(f"  Predictions CSV : {'✅ Found' if HAS_PRED else '⚠️  Not found'}")
    print(f"  Model Metrics   : {'✅ Found' if HAS_METRICS else '⚠️  Not found'}")
    print("=" * 55)
    app.run(debug=True, host="0.0.0.0", port=8050)
