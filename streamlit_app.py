import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import io
from datetime import datetime

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.api import SimpleExpSmoothing, Holt, ExponentialSmoothing
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy.stats import shapiro

from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression
from numpy import sqrt

# =============================
# 1. FONCTIONS UTILITAIRES
# =============================

def calculate_metrics(y_true, y_pred, resid=None):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = sqrt(mse)

    mape_series = np.abs((y_true - y_pred) / y_true)
    mape = np.mean(mape_series[np.isfinite(mape_series)]) * 100

    ljung_p, shapiro_p = None, None

    if resid is not None and len(resid.dropna()) > 3:
        try:
            ljung_p = acorr_ljungbox(resid.dropna(), lags=[10], return_df=True)['lb_pvalue'].iloc[0]
        except:
            ljung_p = np.nan

        try:
            if len(resid.dropna()) <= 5000:
                shapiro_p = shapiro(resid.dropna()).pvalue
            else:
                shapiro_p = "N/A (taille)"
        except:
            shapiro_p = np.nan

    return {
        "MSE": mse,
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape,
        "Ljung-Box p-value": ljung_p,
        "Shapiro-Wilk p-value": shapiro_p
    }


# =============================
# 2. CONFIG STREAMLIT & LOG
# =============================

st.set_page_config(page_title="Prévision de Séries Temporelles", layout="wide")

if "log" not in st.session_state:
    st.session_state["log"] = ""

def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state["log"] += f"[{ts}] {msg}\n"


st.title("📈 Application d’Analyse et de Prévision de Séries Temporelles")

log("Démarrage de l'application")

# =============================
# 3. BARRE LATÉRALE
# =============================

with st.sidebar:
    st.header("⚙️ Paramètres")

    uploaded_file = st.file_uploader("Importer un dataset (CSV)", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        log(f"Fichier chargé : {uploaded_file.name}")

        date_col = st.selectbox("Colonne Date", df.columns)
        target_col = st.selectbox("Colonne Cible", df.columns)

        resample_freq = st.selectbox("Fréquence", ["D", "W", "M"])
        agg_method = st.selectbox("Agrégation", ["sum", "mean"])

        period = st.number_input("Période saisonnière", min_value=1, value=7)
        train_ratio = st.slider("Train (%)", 60, 90, 80) / 100
        horizon = st.number_input("Horizon de prévision", min_value=1, value=period * 2)

        run = st.button("🚀 Lancer l'analyse")


# =============================
# 4. TRAITEMENT PRINCIPAL
# =============================

if uploaded_file and run and date_col != target_col:

    with st.spinner("Analyse en cours..."):

        # ---------- PRÉTRAITEMENT (CORRIGÉ) ----------
        log("Prétraitement des données")

        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df.dropna(subset=[date_col], inplace=True)

        # 🔥 CORRECTION MAJEURE : groupby AVANT resample
        df_grouped = df.groupby(date_col)[target_col].agg(agg_method)
        log(f"Agrégation initiale : {len(df)} lignes → {len(df_grouped)} dates uniques")

        ts = df_grouped.sort_index()
        ts = ts.resample(resample_freq).agg(agg_method)

        missing = ts.isna().sum()
        ts.interpolate(method="linear", inplace=True)

        log(f"Valeurs manquantes interpolées : {missing}")

        st.success(f"Série prête : {len(ts)} observations")

        # ---------- TRAIN / TEST ----------
        train_size = int(len(ts) * train_ratio)
        train, test = ts[:train_size], ts[train_size:]

        # =============================
        # 5. ANALYSE EXPLORATOIRE
        # =============================

        st.header("1️⃣ Analyse Exploratoire")

        fig, ax = plt.subplots(figsize=(10, 4))
        ts.plot(ax=ax)
        ax.set_title("Série temporelle")
        st.pyplot(fig)

        try:
            decomp = seasonal_decompose(ts, model="additive", period=period)
            st.pyplot(decomp.plot())
            log("Décomposition saisonnière réalisée")
        except Exception as e:
            log(f"Décomposition échouée : {e}")

        adf_p = adfuller(ts)[1]
        kpss_p = kpss(ts)[1]

        st.write(f"ADF p-value : {adf_p:.4f}")
        st.write(f"KPSS p-value : {kpss_p:.4f}")

        # =============================
        # 6. MODÉLISATION
        # =============================

        st.header("2️⃣ Modélisation & Évaluation")

        results = []
        models = {}

        # --- Régression Linéaire ---
        X_train = np.arange(len(train)).reshape(-1, 1)
        X_test = np.arange(len(train), len(ts)).reshape(-1, 1)

        lr = LinearRegression().fit(X_train, train.values)
        pred_lr = pd.Series(lr.predict(X_test), index=test.index)
        metrics_lr = calculate_metrics(test, pred_lr, test - pred_lr)

        results.append({"Modèle": "Régression Linéaire", **metrics_lr})
        models["Régression Linéaire"] = lr

        # --- Lissage ---
        smooth_models = {
            "SES": SimpleExpSmoothing(train, initialization_method="estimated"),
            "Holt": Holt(train, initialization_method="estimated"),
            "HW Additif": ExponentialSmoothing(train, trend="add", seasonal="add", seasonal_periods=period),
            "HW Multiplicatif": ExponentialSmoothing(train, trend="add", seasonal="mul", seasonal_periods=period),
        }

        for name, model in smooth_models.items():
            try:
                fitted = model.fit()
                pred = fitted.forecast(len(test))
                metrics = calculate_metrics(test, pred, fitted.resid)

                results.append({"Modèle": name, **metrics})
                models[name] = fitted

                log(f"Modèle {name} ajusté")
            except Exception as e:
                log(f"Erreur modèle {name} : {e}")

        results_df = pd.DataFrame(results).sort_values("MSE")
        st.dataframe(results_df)

        best_model_name = results_df.iloc[0]["Modèle"]
        st.success(f"🏆 Meilleur modèle : {best_model_name}")

        # =============================
        # 7. PRÉVISIONS
        # =============================

        st.header("3️⃣ Prévisions futures")

        best_model = models[best_model_name]

        if best_model_name == "Régression Linéaire":
            X_future = np.arange(len(ts), len(ts) + horizon).reshape(-1, 1)
            forecast = pd.Series(best_model.predict(X_future))
        else:
            forecast = best_model.forecast(horizon)

        forecast.name = "Prévision"

        future_index = pd.date_range(
            start=ts.index[-1], periods=horizon + 1, freq=ts.index.freq
        )[1:]

        forecast_df = pd.DataFrame(forecast.values, index=future_index, columns=["Prévision"])

        fig, ax = plt.subplots(figsize=(10, 4))
        ts.plot(ax=ax, label="Historique")
        forecast_df["Prévision"].plot(ax=ax, label="Prévision", linestyle="--")
        ax.legend()
        st.pyplot(fig)

        # =============================
        # 8. EXPORTS
        # =============================

        st.download_button(
            "📥 Télécharger les prévisions (CSV)",
            forecast_df.to_csv().encode("utf-8"),
            "previsions.csv",
            "text/csv",
        )

        st.download_button(
            "📥 Télécharger le journal (TXT)",
            st.session_state["log"].encode("utf-8"),
            "audit_log.txt",
            "text/plain",
        )

        st.header("📝 Journal d'exécution")
        st.code(st.session_state["log"])
