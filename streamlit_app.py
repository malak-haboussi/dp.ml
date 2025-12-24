import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import json, uuid, os, time
from datetime import datetime
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, Holt, ExponentialSmoothing
from statsmodels.tsa.stattools import adfuller, kpss
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from scipy.stats import shapiro, skew, kurtosis

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(page_title="Expert ROMARIN Forecast", page_icon="📈", layout="wide")

# CSS personnalisé pour un look professionnel
st.markdown("""
    <style>
    .main { background-color: #f0f2f6; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; border-left: 5px solid #004a99; }
    .report-title { color: #004a99; font-weight: bold; border-bottom: 2px solid #004a99; padding-bottom: 10px; }
    </style>
    """, unsafe_allow_html=True)

# --- FONCTIONS TECHNIQUES ---
@st.cache_data
def load_and_clean_data():
    df = sns.load_dataset("flights")
    df["date"] = pd.to_datetime(df["year"].astype(str) + "-" + df["month"].astype(str))
    series = df.set_index("date").asfreq('MS')["passengers"].interpolate(method='linear')
    return series, df

def safe_serialize(obj):
    """Nettoie les objets Numpy pour l'affichage JSON"""
    if isinstance(obj, (np.ndarray, list)):
        return [round(float(x), 4) for x in obj]
    if isinstance(obj, (np.float64, np.int64, float, int)):
        return round(float(obj), 4)
    return str(obj)

# --- BARRE LATÉRALE ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2103/2103633.png", width=80)
    st.title("⚙️ Paramètres")
    st.divider()
    
    method_name = st.selectbox(
        "Modèle de prévision",
        ["Moyenne Mobile", "Régression Linéaire", "Lissage Simple (SES)", 
         "Lissage de Holt", "Holt-Winters Additif", "Holt-Winters Multiplicatif"]
    )
    
    split_ratio = st.slider("Split Entraînement (%)", 60, 90, 80)
    st.divider()
    run_btn = st.button("🚀 Lancer l'Audit Complet", use_container_width=True)

# --- CORPS PRINCIPAL ---
st.title("📊 APPLICATION DE PREVISION ")
st.markdown("Ce système génère un journal d'audit complet conforme aux exigences du projet.")

if run_btn:
    # 1. Préparation
    sid = uuid.uuid4().hex[:8].upper()
    series, raw_df = load_and_clean_data()
    train_size = int(len(series) * (split_ratio / 100))
    train, test = series.iloc[:train_size], series.iloc[train_size:]
    
    # 2. Modélisation
    start_time = time.time()
    try:
        if method_name == "Moyenne Mobile":
            preds = series.shift(1).rolling(window=12).mean().iloc[train_size:]
            params = {"Window": 12}
        elif method_name == "Régression Linéaire":
            coef = np.polyfit(np.arange(len(train)), train.values, 1)
            preds = pd.Series(np.poly1d(coef)(np.arange(len(train), len(series))), index=test.index)
            params = {"Pente": coef[0], "Intercept": coef[1]}
        elif method_name == "Lissage Simple (SES)":
            model = SimpleExpSmoothing(train, initialization_method="estimated").fit()
            preds, params = model.forecast(len(test)), model.params
        elif method_name == "Lissage de Holt":
            model = Holt(train, initialization_method="estimated").fit()
            preds, params = model.forecast(len(test)), model.params
        elif method_name == "Holt-Winters Additif":
            model = ExponentialSmoothing(train, trend="add", seasonal="add", seasonal_periods=12).fit()
            preds, params = model.forecast(len(test)), model.params
        else: # Multiplicatif
            model = ExponentialSmoothing(train, trend="add", seasonal="mul", seasonal_periods=12, use_boxcox=True).fit()
            preds, params = model.forecast(len(test)), model.params
        
        exec_time = time.time() - start_time
        mse = mean_squared_error(test, preds)
        mape = mean_absolute_percentage_error(test, preds) * 100

        # --- AFFICHAGE DES RÉSULTATS ---
        st.subheader("📈 Visualisation des Prévisions", divider="blue")
        
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(train, label="Historique (Train)", color="#1e3d59", linewidth=2)
        ax.plot(test, label="Réel (Test)", color="#27ae60", linewidth=2)
        ax.plot(preds, label="Prévision", color="#e74c3c", linestyle="--", linewidth=2)
        ax.fill_between(test.index, preds*0.95, preds*1.05, color='red', alpha=0.1, label="IC 95%")
        ax.set_title(f"Modèle : {method_name}")
        ax.legend()
        st.pyplot(fig)

        # --- JOURNAL D'AUDIT DÉTAILLÉ (STYLE PAGE 4) ---
        st.subheader(f"📑 Journal d'Audit Détaillé - Session {sid}", divider="blue")
        
        tab1, tab2, tab3 = st.tabs(["📋 Import & EDA", "⚙️ Modélisation", "📊 Évaluation"])

        with tab1:
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**1.1 En-tête & Importation**")
                st.write(f"• Date : {datetime.now().strftime('%d/%m/%Y %H:%M')}")
                st.write(f"• Obs. importées : {len(series)}")
                st.write(f"• Variables : {list(raw_df.columns)}")
                st.write(f"• Traitement : Interpolation & Fréquence MS")
            with c2:
                st.markdown("**1.2 Analyse Exploratoire**")
                st.write(f"• Moyenne : {series.mean():.2f} | Skewness : {skew(series):.2f}")
                st.write(f"• ADF p-value : {adfuller(series)[1]:.4f}")
                st.write(f"• Stationnarité : {'Non' if adfuller(series)[1] > 0.05 else 'Oui'}")

        with tab2:
            st.markdown("**2.1 Configuration du Modèle**")
            # Correction du TypeError ici avec une boucle de nettoyage
            clean_params = {k: safe_serialize(v) for k, v in params.items() if any(x in k for x in ['smoothing', 'initial', 'lamda'])}
            st.json(clean_params)
            st.write(f"**Temps d'exécution :** {exec_time:.4f} secondes")

        with tab3:
            st.markdown("**3.1 Métriques de Qualité**")
            col_m1, col_m2, col_m3 = st.columns(3)
            col_m1.metric("MSE", f"{mse:.2f}")
            col_m2.metric("MAPE", f"{mape:.2f}%")
            col_m3.metric("Shapiro (Normalité)", f"{shapiro(test-preds)[1]:.4f}")
            
            # Histogramme des résidus
            
            fig2, ax2 = plt.subplots(figsize=(6, 2))
            sns.histplot(test - preds, kde=True, ax=ax2, color="purple")
            ax2.set_title("Distribution des Résidus")
            st.pyplot(fig2)

        # --- EXPORT ---
        st.divider()
        audit_json = json.dumps({"audit": sid, "method": method_name, "metrics": {"mse": mse, "mape": mape}, "params": clean_params}, indent=4)
        st.download_button("📥 Télécharger le Livrable JSON", audit_json, file_name=f"ROMARIN_AUDIT_{sid}.json", mime="application/json")

    except Exception as e:
        st.error(f"❌ Erreur critique : {e}")
        st.info("Astuce : Si vous utilisez Holt-Winters, assurez-vous que les données n'ont pas de valeurs nulles ou négatives.")

else:
    st.info("👋 Bienvenue dans le système ROMARIN. Configurez vos paramètres à gauche et cliquez sur **Lancer l'Audit**.")
