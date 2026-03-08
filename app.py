import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output

import pandas as pd
import plotly.express as px
import json

# ─────────────────────────────
# LOAD DATA
# ─────────────────────────────
pred_df = pd.read_csv("output/predictions_2025.csv", parse_dates=["date"])
future_df = pd.read_csv("output/future_predictions_2026.csv", parse_dates=["date"])

with open("output/model_metrics.json") as f:
    metrics = json.load(f)

provinces = sorted(pred_df["province"].unique())

# ─────────────────────────────
# APP
# ─────────────────────────────
app = dash.Dash(__name__)

# ─────────────────────────────
# LAYOUT
# ─────────────────────────────
app.layout = html.Div([

    html.H1("Thailand Tourism Forecast Dashboard",
            style={"textAlign":"center"}),

    # KPI
    html.Div([

        html.Div([
            html.H4("R² Score"),
            html.H2(metrics["r2"])
        ], style={"width":"30%","textAlign":"center"}),

        html.Div([
            html.H4("MAPE"),
            html.H2(f"{metrics['mape']} %")
        ], style={"width":"30%","textAlign":"center"}),

        html.Div([
            html.H4("Best Model"),
            html.H2(metrics["best_model"])
        ], style={"width":"30%","textAlign":"center"}),

    ], style={"display":"flex","justifyContent":"space-between"}),

    html.Br(),

    # Province filter
    html.Div([
        html.Label("Select Province"),

        dcc.Dropdown(
            id="province_filter",
            options=[{"label":i,"value":i} for i in provinces],
            value=provinces[0],
            clearable=False
        )

    ], style={"width":"300px"}),

    html.Br(),

    # Actual vs predicted
    html.H3("Actual vs Predicted (2025)"),

    dcc.Graph(id="actual_chart"),

    # Forecast
    html.H3("Forecast Tourists (2026)"),

    dcc.Graph(id="forecast_chart"),

    # Table
    html.H3("Prediction Results"),

    dash_table.DataTable(

        id="result_table",

        columns=[
            {"name":"Date","id":"date"},
            {"name":"Province","id":"province"},
            {"name":"Actual Visitors","id":"visitors_total"},
            {"name":"Predicted","id":"predicted"},
            {"name":"Error","id":"error"},
            {"name":"Error %","id":"error_pct"},
        ],

        page_size=10,
        style_cell={"textAlign":"center"}
    )

], style={
    "maxWidth":"1200px",
    "margin":"auto",
    "padding":"20px"
})

# ─────────────────────────────
# CALLBACK
# ─────────────────────────────
@app.callback(

    Output("actual_chart","figure"),
    Output("forecast_chart","figure"),
    Output("result_table","data"),

    Input("province_filter","value")

)

def update_dashboard(province):

    dff = pred_df[pred_df["province"] == province]
    dff_future = future_df[future_df["province"] == province]

    # actual vs predicted
    fig1 = px.line(
        dff,
        x="date",
        y=["visitors_total","predicted"],
        markers=True,
        title=f"Actual vs Predicted Visitors : {province}",
        labels={"value":"Visitors","variable":"Type"}
    )

    fig1.update_layout(template="plotly_white")

    # forecast
    fig2 = px.line(
        dff_future,
        x="date",
        y="predicted_visitors",
        markers=True,
        title=f"Tourist Forecast 2026 : {province}"
    )

    fig2.update_layout(template="plotly_white")

    return fig1, fig2, dff.to_dict("records")


# ─────────────────────────────
# RUN
# ─────────────────────────────
if __name__ == "__main__":
    app.run(debug=True)