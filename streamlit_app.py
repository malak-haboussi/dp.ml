# ==========================================================
# STREAMLIT TIME SERIES FORECASTING APPLICATION
# Dataset: Store Item Demand Forecasting (Kaggle)
# ==========================================================

import streamlit as st
import pandas as pd
import numpy as np
import json
from datetime import datetime
import matplotlib.pyplot as plt

from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.holtwinters import (
    SimpleExpSmoothing,
    Holt,
    ExponentialSmoothing
)
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy.stats import shapiro
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="Time Series Forecasting App",
    layout="wide"
)

st.title("📈 Application d’Analyse et de Prévision de Séries Temporelles")

# ==========================================================
# SIDEBAR – CONFIGURATION
# ==========================================================

st.sidebar.header("⚙️ Configuration")

uploaded_file = st.sidebar.file_uploader(
    "Uploader train.csv (Kaggle)",
    type="csv"
)

split_ratio = st.sidebar.selectbox(
    "Découpage Train / Test",
    [0.7, 0.8]
)

forecast_horizon = st.sidebar.slider(
    "Horizon de prévision",
    min_value=7,
    max_value=365,
    value=30
)

run_button = st.sidebar.button("🚀 Lancer l’analyse")

# ==========================================================
# MAIN LOGIC
# ==========================================================

if uploaded_file and run_button:

    # ===============================
    # DATA IMPORT
    # ===============================
    df = pd.read_csv(uploaded_file)
    df["date"] = pd.to_datetime(df["date"])

    st.success(f"Données importées : {len(df)} observations")

    # ===============================
    # SERIES SELECTION
    # ===============================
    store_id = st.sidebar.selectbox("Store", sorted(df["store"].unique()))
    item_id = st.sidebar.selectbox("Item", sorted(df["item"].unique()))

    series = (
        df[(df["store"] == store_id) & (df["item"] == item_id)]
        .sort_values("date")
        .set_index("date")["sales"]
    )

    st.subheader("📊 Série temporelle sélectionnée")

    fig, ax = plt.subplots()
    ax.plot(series)
    ax.set_xlabel("Date")
    ax.set_ylabel("Sales")
    st.pyplot(fig)

    # ===============================
    # EXPLORATORY ANALYSIS
    # ===============================
    st.subheader("🔍 Analyse exploratoire")

    stats = {
        "Moyenne": series.mean(),
        "Variance": series.var(),
        "Skewness": series.skew(),
        "Kurtosis": series.kurt()
    }

    st.write(pd.DataFrame(stats, index=["Valeur"]))

    adf_p = adfuller(series)[1]
    kpss_p = kpss(series, regression="c")[1]

    st.write(f"ADF p-value : {adf_p:.4f}")
    st.write(f"KPSS p-value : {kpss_p:.4f}")

    # ===============================
    # SEASONALITY DETECTION
    # ===============================
    st.subheader("🌊 Détection de la saisonnalité")

    seasonal_period = 365
    st.write(f"Période saisonnière utilisée : {seasonal_period}")

    stl = STL(series, period=seasonal_period)
    res = stl.fit()

    fig_stl = res.plot()
    st.pyplot(fig_stl)

    # ===============================
    # TRAIN / TEST SPLIT
    # ===============================
    split = int(len(series) * split_ratio)
    train = series.iloc[:split]
    test = series.iloc[split:]

    st.write(f"Taille train : {len(train)}")
    st.write(f"Taille test : {len(test)}")

    # ===============================
    # METRICS FUNCTION
    # ===============================
    def compute_metrics(y_true, y_pred):
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        return mse, mae, rmse, mape

    # ===============================
    # MODELS
    # ===============================
    st.subheader("🧠 Modélisation & comparaison")

    results = []

    ses = SimpleExpSmoothing(train).fit(optimized=True)
    ses_f = ses.forecast(len(test))
    results.append(("SES", ses.params, *compute_metrics(test, ses_f)))

    holt = Holt(train).fit(optimized=True)
    holt_f = holt.forecast(len(test))
    results.append(("Holt", holt.params, *compute_metrics(test, holt_f)))

    hw_add = ExponentialSmoothing(
        train,
        trend="add",
        seasonal="add",
        seasonal_periods=seasonal_period
    ).fit(optimized=True)
    hw_add_f = hw_add.forecast(len(test))
    results.append(("Holt-Winters Additif", hw_add.params, *compute_metrics(test, hw_add_f)))

    hw_mul = ExponentialSmoothing(
        train,
        trend="add",
        seasonal="mul",
        seasonal_periods=seasonal_period
    ).fit(optimized=True)
    hw_mul_f = hw_mul.forecast(len(test))
    results.append(("Holt-Winters Multiplicatif", hw_mul.params, *compute_metrics(test, hw_mul_f)))

    results_df = pd.DataFrame(
        results,
        columns=["Modèle", "Paramètres", "MSE", "MAE", "RMSE", "MAPE"]
    )

    st.dataframe(results_df)

    # ===============================
    # BEST MODEL SELECTION
    # ===============================
    best_row = results_df.sort_values("MSE").iloc[0]
    best_model_name = best_row["Modèle"]

    st.success(f"✅ Meilleur modèle sélectionné : {best_model_name}")

    model_map = {
        "SES": (ses, ses_f),
        "Holt": (holt, holt_f),
        "Holt-Winters Additif": (hw_add, hw_add_f),
        "Holt-Winters Multiplicatif": (hw_mul, hw_mul_f)
    }

    best_model, best_forecast_test = model_map[best_model_name]

    # ===============================
    # RESIDUAL ANALYSIS
    # ===============================
    st.subheader("🧪 Analyse des résidus")

    residuals = test - best_forecast_test

    shapiro_p = shapiro(residuals).pvalue
    ljung_p = acorr_ljungbox(residuals, lags=[10], return_df=True)["lb_pvalue"].values[0]

    st.write(f"Shapiro-Wilk p-value : {shapiro_p:.4f}")
    st.write(f"Ljung-Box p-value : {ljung_p:.4f}")

    # ===============================
    # FINAL FORECAST
    # ===============================
    st.subheader("🔮 Prévisions futures")

    future_forecast = best_model.forecast(forecast_horizon)
    future_dates = pd.date_range(
        series.index[-1],
        periods=forecast_horizon + 1,
        freq="D"
    )[1:]

    fig_f, ax_f = plt.subplots()
    ax_f.plot(series, label="Historique")
    ax_f.plot(future_dates, future_forecast, label="Prévision")
    ax_f.legend()
    st.pyplot(fig_f)

    forecast_df = pd.DataFrame({
        "date": future_dates,
        "forecast": future_forecast.values
    })

    st.download_button(
        label="📥 Télécharger les prévisions (CSV)",
        data=forecast_df.to_csv(index=False),
        file_name="forecasts.csv",
        mime="text/csv"
    )

# ==========================================================
# FOOTER
# ==========================================================

st.markdown("---")
st.markdown(
    "Application de prévision de séries temporelles – "
    "Lissage exponentiel, Holt, Holt-Winters | MSE, MAE, RMSE, MAPE"
)
