import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output

import pandas as pd
import plotly.express as px
import json

# ─────────────────────────────
# PROVINCE MAP (TH → EN)
# ─────────────────────────────
province_map = {
    "กระบี่": "Krabi",
    "กรุงเทพมหานคร": "Bangkok Metropolis",
    "กาญจนบุรี": "Kanchanaburi",
    "กาฬสินธุ์": "Kalasin",
    "กำแพงเพชร": "Kamphaeng Phet",
    "ขอนแก่น": "Khon Kaen",
    "จันทบุรี": "Chanthaburi",
    "ฉะเชิงเทรา": "Chachoengsao",
    "ชลบุรี": "Chon Buri",
    "ชัยนาท": "Chai Nat",
    "ชัยภูมิ": "Chaiyaphum",
    "ชุมพร": "Chumphon",
    "เชียงราย": "Chiang Rai",
    "เชียงใหม่": "Chiang Mai",
    "ตรัง": "Trang",
    "ตราด": "Trat",
    "ตาก": "Tak",
    "นครนายก": "Nakhon Nayok",
    "นครปฐม": "Nakhon Pathom",
    "นครพนม": "Nakhon Phanom",
    "นครราชสีมา": "Nakhon Ratchasima",
    "นครศรีธรรมราช": "Nakhon Si Thammarat",
    "นครสวรรค์": "Nakhon Sawan",
    "นนทบุรี": "Nonthaburi",
    "นราธิวาส": "Narathiwat",
    "น่าน": "Nan",
    "บึงกาฬ": "Bueng Kan",
    "บุรีรัมย์": "Buri Ram",
    "ปทุมธานี": "Pathum Thani",
    "ประจวบคีรีขันธ์": "Prachuap Khiri Khan",
    "ปราจีนบุรี": "Prachin Buri",
    "ปัตตานี": "Pattani",
    "พระนครศรีอยุธยา": "Phra Nakhon Si Ayutthaya",
    "พะเยา": "Phayao",
    "พังงา": "Phangnga",
    "พัทลุง": "Phatthalung",
    "พิจิตร": "Phichit",
    "พิษณุโลก": "Phitsanulok",
    "เพชรบุรี": "Phetchaburi",
    "เพชรบูรณ์": "Phetchabun",
    "แพร่": "Phrae",
    "ภูเก็ต": "Phuket",
    "มหาสารคาม": "Maha Sarakham",
    "มุกดาหาร": "Mukdahan",
    "แม่ฮ่องสอน": "Mae Hong Son",
    "ยโสธร": "Yasothon",
    "ยะลา": "Yala",
    "ร้อยเอ็ด": "Roi Et",
    "ระนอง": "Ranong",
    "ระยอง": "Rayong",
    "ราชบุรี": "Ratchaburi",
    "ลพบุรี": "Lop Buri",
    "ลำปาง": "Lampang",
    "ลำพูน": "Lamphun",
    "เลย": "Loei",
    "ศรีสะเกษ": "Si Sa Ket",
    "สกลนคร": "Sakon Nakhon",
    "สงขลา": "Songkhla",
    "สตูล": "Satun",
    "สมุทรปราการ": "Samut Prakan",
    "สมุทรสงคราม": "Samut Songkhram",
    "สมุทรสาคร": "Samut Sakhon",
    "สระแก้ว": "Sa Kaeo",
    "สระบุรี": "Saraburi",
    "สิงห์บุรี": "Sing Buri",
    "สุโขทัย": "Sukhothai",
    "สุพรรณบุรี": "Suphan Buri",
    "สุราษฎร์ธานี": "Surat Thani",
    "สุรินทร์": "Surin",
    "หนองคาย": "Nong Khai",
    "หนองบัวลำภู": "Nong Bua Lam Phu",
    "อ่างทอง": "Ang Thong",
    "อำนาจเจริญ": "Amnat Charoen",
    "อุดรธานี": "Udon Thani",
    "อุตรดิตถ์": "Uttaradit",
    "อุทัยธานี": "Uthai Thani",
    "อุบลราชธานี": "Ubon Ratchathani",
}

province_reverse = {v: k for k, v in province_map.items()}

# ---------------------------
# LOAD DATA
# ---------------------------
pred_df = pd.read_csv("output/predictions_2025.csv", parse_dates=["date"])
future_df = pd.read_csv("output/future_predictions_2026.csv", parse_dates=["date"])
df = pd.read_csv("data/tourism_cleaned.csv", parse_dates=["date"])

with open("map/thailand_provinces.geojson", encoding="utf-8") as f:
    thai_map = json.load(f)

with open("output/model_metrics.json") as f:
    metrics = json.load(f)

# ---------------------------
# MERGE DATA
# ---------------------------
pred_df = pred_df.merge(
    df[["date", "province", "visitors_thai", "visitors_foreign"]],
    on=["date", "province"],
    how="left",
)

pred_df["visitors_total"] = pred_df["visitors_thai"] + pred_df["visitors_foreign"]
pred_df["predicted"] = pred_df["visitors_total_pred"]

pred_df["error"] = pred_df["visitors_total"] - pred_df["predicted"]
pred_df["error_pct"] = abs(pred_df["error"] / pred_df["visitors_total"]) * 100

# ---------------------------
# MAP DATA
# ---------------------------
future_df["province_en"] = future_df["province"].map(province_map)

map_df = future_df.groupby("province_en")["visitors_total"].sum().reset_index()

# ---------------------------
# แบ่งระดับสี
# ---------------------------
low = map_df["visitors_total"].quantile(0.33)
mid = map_df["visitors_total"].quantile(0.66)


def classify(x):
    if x <= low:
        return "Low"
    elif x <= mid:
        return "Medium"
    else:
        return "High"


map_df["tourism_level"] = map_df["visitors_total"].apply(classify)

default_province = future_df.sort_values("visitors_total", ascending=False).iloc[0][
    "province"
]

# ---------------------------
# DASH APP
# ---------------------------
app = dash.Dash(__name__)

app.layout = html.Div(
    [
        html.H1("Thailand Tourism Forecast Dashboard", style={"textAlign": "center"}),
        html.Div(
            [
                html.Div(
                    [html.H4("R² Score"), html.H2(metrics["r2"])],
                    style={"width": "45%", "textAlign": "center"},
                ),
                html.Div(
                    [html.H4("MAPE"), html.H2(f"{metrics['mape']} %")],
                    style={"width": "45%", "textAlign": "center"},
                ),
            ],
            style={"display": "flex", "justifyContent": "space-between"},
        ),
        html.Br(),
        html.H2("Tourism Map (Click Province)"),
        dcc.Graph(id="tourism_map"),
        html.Br(),
        html.H2(id="province_title", style={"textAlign": "center"}),
        html.H3("Actual vs Predicted (2025)"),
        dcc.Graph(id="actual_chart"),
        html.H3("Tourist Forecast 2026"),
        dcc.Graph(id="forecast_chart"),
        html.H3("Prediction Results"),
        dash_table.DataTable(
            id="result_table",
            columns=[
                {"name": "Date", "id": "date"},
                {"name": "Province", "id": "province"},
                {"name": "Actual Visitors", "id": "visitors_total"},
                {"name": "Predicted", "id": "predicted"},
                {"name": "Error", "id": "error"},
                {"name": "Error %", "id": "error_pct"},
            ],
            page_size=10,
            style_cell={"textAlign": "center"},
        ),
    ],
    style={"maxWidth": "1200px", "margin": "auto", "padding": "20px"},
)


# ---------------------------
# MAP CALLBACK
# ---------------------------
@app.callback(
    Output("tourism_map", "figure"),
    Input("actual_chart", "id"),
)
def update_map(_):

    fig = px.choropleth(
        map_df,
        geojson=thai_map,
        locations="province_en",
        featureidkey="properties.name",
        color="tourism_level",
        color_discrete_map={
            "Low": "rgb(250,234,90)",  # forest green
            "Medium": "rgb(250,160,90)",  # amber
            "High": "rgb(213,97,97)",  # soft red
        },
        hover_name="province_en",
        hover_data={"visitors_total": ":,.0f", "tourism_level": True},
    )

    fig.update_geos(fitbounds="locations", visible=False)

    fig.update_layout(
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        hoverlabel=dict(font_size=22, font_family="Arial"),
    )

    return fig


# ---------------------------
# DASHBOARD UPDATE
# ---------------------------
@app.callback(
    Output("province_title", "children"),
    Output("actual_chart", "figure"),
    Output("forecast_chart", "figure"),
    Output("result_table", "data"),
    Input("tourism_map", "clickData"),
)
def update_dashboard(clickData):

    if clickData is None:
        province = default_province
    else:
        province_en = clickData["points"][0]["location"]
        province = province_reverse[province_en]

    dff = pred_df[pred_df["province"] == province]
    dff_future = future_df[future_df["province"] == province]

    fig1 = px.line(
        dff,
        x="date",
        y=["visitors_total", "predicted"],
        markers=True,
        template="plotly_white",
    )

    fig2 = px.line(
        dff_future,
        x="date",
        y=["visitors_total", "visitors_thai", "visitors_foreign"],
        markers=True,
        template="plotly_white",
    )

    title = f"Province : {province}"

    return title, fig1, fig2, dff.to_dict("records")


# ---------------------------
# RUN
# ---------------------------
if __name__ == "__main__":
    app.run(debug=True)
