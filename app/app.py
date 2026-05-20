import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib
import os

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Gold Price Forecasting",
    page_icon="🥇",
    layout="wide",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
        /* Dark background */
        .stApp { background-color: #0e0e0e; color: #f0f0f0; }
        section[data-testid="stSidebar"] { background-color: #1a1a1a; }
        /* Headers */
        h1, h2, h3 { color: #FFD700 !important; }
        /* Dividers */
        hr { border-color: #333333; }
        /* Selectbox / Slider labels */
        label { color: #cccccc !important; }
        /* Warning / info boxes */
        .stAlert { background-color: #1f1f1f; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "monthly.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")

GOLD = "#FFD700"
DARK_BG = "#0e0e0e"
PAPER_BG = "#1a1a1a"

# ── Load data ──────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df["Date"] = pd.to_datetime(df["Date"])
    df["Price"] = df["Price"].astype(float)
    df = df.sort_values("Date").reset_index(drop=True)
    return df

df_all = load_data()
df = df_all[df_all["Date"] >= "1971-01-01"].reset_index(drop=True)

# ── Title ──────────────────────────────────────────────────────────────────────
st.title("Gold Price Forecasting Dashboard")
st.markdown(
    """
    This dashboard analyses and forecasts gold prices using a range of models,
    from simple baselines to deep learning.

    > **Note on pre-1971 data:** The dataset contains 130+ years of historical
    > prices, but pre-1971 data is **statistically unusable** for modern
    > forecasting. The **Zivot-Andrews structural break test** identified
    > **January 1971** as the regime-change point — the moment gold ceased to
    > be a fixed government price and became a free-market asset following the
    > Nixon Shock. All charts and models use post-1971 data only.
    """,
)

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — PRICE HISTORY
# ══════════════════════════════════════════════════════════════════════════════
st.header("1 — Gold Price History (1971–2026)")

EVENTS = {
    1971: "Nixon Shock — Gold Standard ends",
    1980: "Oil crisis + Soviet invasion — Gold hits $850",
    1999: "Long bear market bottom — Gold near $250",
    2008: "Global Financial Crisis",
    2011: "Eurozone crisis — Gold hits $1,900",
    2013: "Gold crash — drops 30%",
    2020: "COVID-19 — Gold breaks $2,000",
    2022: "Ukraine war begins",
    2023: "Israel-Hamas conflict — Gold breaks $2,000 ceiling",
    2024: "Fed rate cuts + Trump tariffs — Gold surges past $2,500",
    2025: "Central bank buying surge — Gold hits $4,000",
    2026: "US-Iran tensions — Gold hits $5,000+",
}

fig_history = go.Figure()

# Main price line
fig_history.add_trace(
    go.Scatter(
        x=df["Date"],
        y=df["Price"],
        mode="lines",
        line=dict(color=GOLD, width=1.8),
        name="Gold Price (USD/oz)",
        hovertemplate="%{x|%b %Y}<br>$%{y:,.2f}<extra></extra>",
    )
)

# Event vertical lines
max_price = df["Price"].max()
for year, label in EVENTS.items():
    dt = pd.Timestamp(f"{year}-01-01")
    # Only draw if within data range
    if dt >= df["Date"].min() and dt <= df["Date"].max() + pd.DateOffset(months=6):
        fig_history.add_vline(
            x=dt.timestamp() * 1000,
            line=dict(color="rgba(255,255,255,0.35)", dash="dash", width=1),
        )
        fig_history.add_annotation(
            x=dt,
            y=max_price * 0.98,
            text=f"{year}: {label}",
            showarrow=False,
            textangle=-90,
            font=dict(size=9, color="rgba(255,255,255,0.55)"),
            xanchor="right",
            yanchor="top",
        )

fig_history.update_layout(
    plot_bgcolor=DARK_BG,
    paper_bgcolor=PAPER_BG,
    font=dict(color="#f0f0f0"),
    xaxis=dict(
        title="Date",
        gridcolor="#2a2a2a",
        showgrid=True,
        zeroline=False,
    ),
    yaxis=dict(
        title="Price (USD / troy oz)",
        gridcolor="#2a2a2a",
        showgrid=True,
        zeroline=False,
        tickprefix="$",
        tickformat=",",
    ),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#f0f0f0")),
    margin=dict(l=60, r=20, t=30, b=50),
    height=500,
    hovermode="x unified",
)

st.plotly_chart(fig_history, use_container_width=True)

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — MODEL COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
st.header("2 — Model Comparison")
st.markdown(
    "Mean Absolute Error (MAE) across all 9 models evaluated on the test set. "
    "Lower is better. The red dashed line marks the **Naive baseline** (MAE = $1,129.51)."
)

MODEL_RESULTS = [
    ("Historical Average",     2488.44, "Baseline"),
    ("Window Average 12m",     1184.66, "Baseline"),
    ("XGBoost",                1175.68, "Machine Learning"),
    ("Random Forest",          1158.22, "Machine Learning"),
    ("Seasonal Naive",         1157.09, "Baseline"),
    ("Naive",                  1129.51, "Baseline"),
    ("Exponential Smoothing",  1115.83, "Statistical"),
    ("ARIMA(1,1,2)",           1096.23, "Statistical"),
    ("LSTM",                    387.57, "Deep Learning"),
]

TYPE_COLORS = {
    "Baseline":        "#888888",
    "Statistical":     "#4a90d9",
    "Machine Learning":"#5cb85c",
    "Deep Learning":   "#9b59b6",
}

names  = [r[0] for r in MODEL_RESULTS]
maes   = [r[1] for r in MODEL_RESULTS]
types  = [r[2] for r in MODEL_RESULTS]
colors = [TYPE_COLORS[t] for t in types]

fig_compare = go.Figure()

fig_compare.add_trace(
    go.Bar(
        x=maes,
        y=names,
        orientation="h",
        marker_color=colors,
        text=[f"${m:,.2f}" for m in maes],
        textposition="outside",
        textfont=dict(color="#f0f0f0", size=11),
        hovertemplate="%{y}<br>MAE: $%{x:,.2f}<extra></extra>",
        name="MAE",
    )
)

# Naive baseline reference line
naive_mae = 1129.51
fig_compare.add_vline(
    x=naive_mae,
    line=dict(color="red", dash="dash", width=1.5),
    annotation_text="Naive baseline",
    annotation_position="top right",
    annotation_font=dict(color="red", size=10),
)

# Legend traces (invisible bars for colour key)
for type_name, color in TYPE_COLORS.items():
    fig_compare.add_trace(
        go.Bar(
            x=[None], y=[None],
            orientation="h",
            marker_color=color,
            name=type_name,
            showlegend=True,
        )
    )

fig_compare.update_layout(
    plot_bgcolor=DARK_BG,
    paper_bgcolor=PAPER_BG,
    font=dict(color="#f0f0f0"),
    xaxis=dict(
        title="MAE (USD)",
        gridcolor="#2a2a2a",
        showgrid=True,
        zeroline=False,
        tickprefix="$",
        tickformat=",",
        range=[0, max(maes) * 1.18],
    ),
    yaxis=dict(
        gridcolor="#2a2a2a",
        showgrid=False,
        zeroline=False,
        autorange="reversed",
    ),
    legend=dict(
        bgcolor="rgba(30,30,30,0.8)",
        bordercolor="#444",
        borderwidth=1,
        font=dict(color="#f0f0f0"),
        title=dict(text="Model Type", font=dict(color="#f0f0f0")),
    ),
    barmode="overlay",
    margin=dict(l=160, r=80, t=30, b=60),
    height=420,
)

st.plotly_chart(fig_compare, use_container_width=True)

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — FORECAST
# ══════════════════════════════════════════════════════════════════════════════
st.header("3 — Price Forecast")

MODEL_OPTIONS = [
    "Naive",
    "Historical Average",
    "Window Average",
    "Seasonal Naive",
    "ARIMA",
    "Exponential Smoothing",
    "Random Forest",
    "XGBoost",
] + (["LSTM"] if TF_AVAILABLE else [])

col1, col2 = st.columns([2, 1])
with col1:
    selected_model = st.selectbox("Select forecasting model", MODEL_OPTIONS)
with col2:
    horizon = st.slider("Forecast horizon (months)", min_value=1, max_value=24, value=12)

# ── Helper: build future date index ───────────────────────────────────────────
def future_dates(last_date, n):
    return pd.date_range(start=last_date + pd.DateOffset(months=1), periods=n, freq="MS")


# ── Helper: confidence interval ───────────────────────────────────────────────
def confidence_band(forecast_values, df_post):
    rolling_std = df_post["Price"].rolling(12).std().iloc[-1]
    margin = 1.5 * rolling_std
    upper = np.array(forecast_values) + margin
    lower = np.array(forecast_values) - margin
    return upper, lower


# ── Helper: ML lag features ───────────────────────────────────────────────────
def build_lag_features(series: pd.Series, date_index: pd.DatetimeIndex) -> pd.DataFrame:
    df_feat = pd.DataFrame({"Price": series.values}, index=date_index)
    df_feat["lag_1"]  = df_feat["Price"].shift(1)
    df_feat["lag_2"]  = df_feat["Price"].shift(2)
    df_feat["lag_3"]  = df_feat["Price"].shift(3)
    df_feat["lag_6"]  = df_feat["Price"].shift(6)
    df_feat["lag_12"] = df_feat["Price"].shift(12)
    df_feat["month"]  = df_feat.index.month
    df_feat["year"]   = df_feat.index.year
    df_feat["rolling_mean_3"] = df_feat["Price"].shift(1).rolling(3).mean()
    df_feat["rolling_mean_6"] = df_feat["Price"].shift(1).rolling(6).mean()
    df_feat["rolling_std_3"]  = df_feat["Price"].shift(1).rolling(3).std()
    return df_feat

LAG_COLS = ["lag_1", "lag_2", "lag_3", "lag_6", "lag_12",
            "month", "year", "rolling_mean_3", "rolling_mean_6", "rolling_std_3"]


# ── Model loading (cached) ─────────────────────────────────────────────────────
@st.cache_resource
def load_arima():
    return joblib.load(os.path.join(MODELS_DIR, "arima.pkl"))

@st.cache_resource
def load_exp_smoothing():
    return joblib.load(os.path.join(MODELS_DIR, "exp_smoothing.pkl"))

@st.cache_resource
def load_rf():
    return joblib.load(os.path.join(MODELS_DIR, "random_forest.pkl"))

@st.cache_resource
def load_xgb():
    return joblib.load(os.path.join(MODELS_DIR, "xgboost.pkl"))

@st.cache_resource
def load_lstm():
    model = tf.keras.models.load_model(os.path.join(MODELS_DIR, "lstm_model.keras"))
    return model


# ── Forecast generation ────────────────────────────────────────────────────────
def generate_forecast(model_name: str, n: int, df_post: pd.DataFrame):
    prices = df_post["Price"]
    last_date = df_post["Date"].iloc[-1]
    fdates = future_dates(last_date, n)

    if model_name == "Naive":
        last_price = prices.iloc[-1]
        return fdates, np.full(n, last_price)

    if model_name == "Historical Average":
        avg = prices.mean()
        return fdates, np.full(n, avg)

    if model_name == "Window Average":
        avg = prices.iloc[-12:].mean()
        return fdates, np.full(n, avg)

    if model_name == "Seasonal Naive":
        preds = []
        for i in range(n):
            idx = -(12 - (i % 12))
            preds.append(prices.iloc[idx])
        return fdates, np.array(preds)

    if model_name == "ARIMA":
        arima_model = load_arima()
        preds = arima_model.predict(n_periods=n)
        return fdates, np.array(preds)

    if model_name == "Exponential Smoothing":
        es_model = load_exp_smoothing()
        preds = es_model.forecast(n)
        return fdates, np.array(preds)

    if model_name in ("Random Forest", "XGBoost"):
        model_fn = load_rf if model_name == "Random Forest" else load_xgb
        ml_model = model_fn()

        # Build rolling historical series extended with predictions
        hist_prices = list(prices.values)
        hist_dates  = list(df_post["Date"].values)
        preds = []

        for i in range(n):
            series = pd.Series(hist_prices)
            date_idx = pd.DatetimeIndex(hist_dates)
            feat_df  = build_lag_features(series, date_idx)
            last_row = feat_df[LAG_COLS].iloc[-1].values.reshape(1, -1)
            pred = float(ml_model.predict(last_row)[0])
            preds.append(pred)
            hist_prices.append(pred)
            hist_dates.append(fdates[i])

        return fdates, np.array(preds)

    if model_name == "LSTM":
        from sklearn.preprocessing import MinMaxScaler
        lstm_model = load_lstm()

        scaler = MinMaxScaler()
        scaler.fit(prices.values.reshape(-1, 1))
        scaled = scaler.transform(prices.values.reshape(-1, 1))

        # Determine sequence length from model input shape
        seq_len = lstm_model.input_shape[1] if lstm_model.input_shape[1] else 12

        window = list(scaled[-seq_len:].flatten())
        preds_scaled = []

        for _ in range(n):
            x = np.array(window[-seq_len:]).reshape(1, seq_len, 1)
            pred_s = float(lstm_model.predict(x, verbose=0)[0][0])
            preds_scaled.append(pred_s)
            window.append(pred_s)

        preds = scaler.inverse_transform(np.array(preds_scaled).reshape(-1, 1)).flatten()
        return fdates, preds

    raise ValueError(f"Unknown model: {model_name}")


# ── Run forecast ───────────────────────────────────────────────────────────────
if selected_model == "LSTM":
    st.info(
        "LSTM is a deep learning model and may take 10–15 seconds to load.",
        icon="ℹ️",
    )

with st.spinner("Loading model… this may take a moment"):
    try:
        fdates, forecast_vals = generate_forecast(selected_model, horizon, df)
        forecast_success = True
    except Exception as e:
        forecast_success = False
        st.error(f"Failed to generate forecast: {e}")

if forecast_success:
    # Last 24 months of actuals for context
    context = df.tail(24)
    upper, lower = confidence_band(forecast_vals, df)
    last_actual_date = df["Date"].iloc[-1]

    fig_fc = go.Figure()

    # Actual context window (white)
    fig_fc.add_trace(
        go.Scatter(
            x=context["Date"],
            y=context["Price"],
            mode="lines",
            line=dict(color="white", width=1.8),
            name="Actual (last 24 months)",
            hovertemplate="%{x|%b %Y}<br>$%{y:,.2f}<extra></extra>",
        )
    )

    # Confidence band (shaded)
    fig_fc.add_trace(
        go.Scatter(
            x=list(fdates) + list(fdates[::-1]),
            y=list(upper) + list(lower[::-1]),
            fill="toself",
            fillcolor="rgba(255,215,0,0.15)",
            line=dict(color="rgba(0,0,0,0)"),
            name="Confidence interval (±1.5σ)",
            hoverinfo="skip",
        )
    )

    # Forecast line (gold)
    fig_fc.add_trace(
        go.Scatter(
            x=fdates,
            y=forecast_vals,
            mode="lines+markers",
            line=dict(color=GOLD, width=2, dash="dot"),
            marker=dict(color=GOLD, size=5),
            name=f"{selected_model} forecast",
            hovertemplate="%{x|%b %Y}<br>$%{y:,.2f}<extra></extra>",
        )
    )

    # Vertical line at forecast start
    fig_fc.add_vline(
        x=last_actual_date.timestamp() * 1000,
        line=dict(color="rgba(255,255,255,0.5)", dash="dash", width=1.5),
        annotation_text="Forecast starts",
        annotation_position="top left",
        annotation_font=dict(color="rgba(255,255,255,0.7)", size=10),
    )

    fig_fc.update_layout(
        plot_bgcolor=DARK_BG,
        paper_bgcolor=PAPER_BG,
        font=dict(color="#f0f0f0"),
        title=dict(
            text=f"{selected_model} — {horizon}-month forecast",
            font=dict(color=GOLD, size=15),
        ),
        xaxis=dict(
            title="Date",
            gridcolor="#2a2a2a",
            showgrid=True,
            zeroline=False,
        ),
        yaxis=dict(
            title="Price (USD / troy oz)",
            gridcolor="#2a2a2a",
            showgrid=True,
            zeroline=False,
            tickprefix="$",
            tickformat=",",
        ),
        legend=dict(
            bgcolor="rgba(30,30,30,0.8)",
            bordercolor="#444",
            borderwidth=1,
            font=dict(color="#f0f0f0"),
        ),
        margin=dict(l=60, r=20, t=50, b=50),
        height=450,
        hovermode="x unified",
    )

    st.plotly_chart(fig_fc, use_container_width=True)

    # Forecast table
    fc_df = pd.DataFrame({
        "Month": [d.strftime("%b %Y") for d in fdates],
        "Forecast (USD)": [f"${v:,.2f}" for v in forecast_vals],
        "Lower bound":    [f"${v:,.2f}" for v in lower],
        "Upper bound":    [f"${v:,.2f}" for v in upper],
    })
    with st.expander("View forecast table"):
        st.dataframe(fc_df, use_container_width=True, hide_index=True)

    # Disclaimer
    st.warning(
        "**Disclaimer:** These forecasts are generated by statistical and machine "
        "learning models trained on historical price data only. They do not account "
        "for geopolitical events, macroeconomic shifts, or market sentiment. "
        "**This is not financial advice.**"
    )
