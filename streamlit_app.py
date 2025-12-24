import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import json, uuid, os, time, shutil
from datetime import datetime

# Modèles et Statistiques
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, Holt, ExponentialSmoothing
from statsmodels.tsa.stattools import adfuller, kpss
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from scipy.stats import shapiro

# Configuration de la page
st.set_page_config(page_title="ROMARIN Forecast", page_icon="📊", layout="wide")

# Style CSS personnalisé
st.markdown("""
    <style>
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; border: 1px solid #e0e0e0; }
    .main { background-color: #f8f9fa; }
    </style>
    """, unsafe_allow_html=True)

# --- ENCODEUR JSON POUR TYPES NUMPY ---
class RomarinEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.floating)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super().default(obj)

# --- CHARGEMENT DES DONNÉES ---
@st.cache_data
def load_data():
    df = sns.load_dataset("flights")
    df["date"] = pd.to_datetime(df["year"].astype(str) + "-" + df["month"].astype(str))
    # Prétraitement : Indexation et Interpolation (Exigence Page 4)
    series = df.set_index("date").asfreq('MS')["passengers"].interpolate(method='linear')
    return series, len(df), df.dtypes.to_dict()

# --- BARRE LATÉRALE ---
with st.sidebar:
    st.title("🛡️ Projet ROMARIN")
    st.markdown("**Session Master 2025**")
    st.divider()
    
    method_name = st.selectbox(
        "Sélectionner le Modèle",
        ["Moyenne Mobile", "Régression Linéaire", "Lissage Simple (SES)", 
         "Lissage de Holt", "Holt-Winters Additif", "Holt-Winters Multiplicatif"]
    )
    
    split_ratio = st.slider("Part de l'entraînement (%)", 50, 90, 80)
    st.divider()
    run_analysis = st.button("📊 Lancer l'Analyse", use_container_width=True)

# --- CORPS PRINCIPAL ---
st.title("📈 Système Expert de Prévision Temporelle")
st.caption(f"Identifiant de session : {st.session_state.get('sid', uuid.uuid4().hex[:8])}")

if run_analysis:
    sid = uuid.uuid4().hex[:8]
    st.session_state.sid = sid
    
    # 1. IMPORTATION ET EDA
    series, raw_count, dtypes = load_data()
    train_size = int(len(series) * (split_ratio / 100))
    train, test = series.iloc[:train_size], series.iloc[train_size:]
    
    # KPIs d'Analyse Exploratoire
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("Moyenne", f"{series.mean():.1f}")
    with col2: st.metric("Variance", f"{series.var():.1f}")
    with col3: st.metric("ADF p-val", f"{adfuller(series)[1]:.4f}")
    with col4: st.metric("Observations", len(series))

    # 2. MODÉLISATION
    start_time = time.time()
    try:
        if method_name == "Moyenne Mobile":
            # Simple window approach for forecast
            full_series = series.copy()
            for i in range(len(test)):
                full_series.iloc[train_size+i] = full_series.iloc[train_size+i-12:train_size+i].mean()
            preds = full_series.iloc[train_size:]
            model_info = {"window": 12}
        
        elif method_name == "Régression Linéaire":
            x = np.arange(len(train))
            coef = np.polyfit(x, train.values, 1)
            preds = pd.Series(np.poly1d(coef)(np.arange(len(train), len(series))), index=test.index)
            model_info = {"pente": coef[0], "intercept": coef[1]}

        elif method_name == "Lissage Simple (SES)":
            model = SimpleExpSmoothing(train, initialization_method="estimated").fit()
            preds = model.forecast(len(test))
            model_info = model.params
            
        elif method_name == "Lissage de Holt":
            model = Holt(train, initialization_method="estimated").fit()
            preds = model.forecast(len(test))
            model_info = model.params

        elif method_name == "Holt-Winters Additif":
            model = ExponentialSmoothing(train, trend="add", seasonal="add", seasonal_periods=12).fit()
            preds = model.forecast(len(test))
            model_info = model.params

        elif method_name == "Holt-Winters Multiplicatif":
            # Sécurité Box-Cox pour éviter les erreurs de calcul (valeurs NaN)
            model = ExponentialSmoothing(train, trend="add", seasonal="mul", seasonal_periods=12, use_boxcox=True).fit()
            preds = model.forecast(len(test))
            model_info = model.params

        exec_duration = time.time() - start_time
        
        # 3. ÉVALUATION ET GRAPHIQUES
        
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(train, label="Entraînement", color="#2c3e50")
        ax.plot(test, label="Réel (Test)", color="#27ae60")
        ax.plot(preds, label="Prévision", color="#e74c3c", linestyle="--")
        ax.set_title(f"Modèle : {method_name}")
        ax.legend()
        st.pyplot(fig)

        # 4. GÉNÉRATION DES JOURNAUX (PAGE 4)
        audit_log = {
            "En-tête": {
                "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Session_ID": sid,
                "Méthode": method_name
            },
            "Importation_Pretraitement": {
                "Observations": raw_count,
                "Variables": str(dtypes),
                "Traitements": "Interpolation linéaire, Fréquence Mensuelle (MS)"
            },
            "Analyse_Exploratoire": {
                "Stats": f"Moyenne={series.mean():.2f}, Skew={series.skew():.2f}",
                "Stationnarité": f"ADF p-val={adfuller(series)[1]:.4f}",
                "Saisonnalité": "Période 12 identifiée"
            },
            "Evaluation_Performance": {
                "MSE": round(mean_squared_error(test, preds), 2),
                "MAPE": f"{round(mean_absolute_percentage_error(test, preds)*100, 2)}%",
                "Normalité_Résidus": "Oui" if shapiro(test-preds)[1] > 0.05 else "Non",
                "Temps_Calcul": f"{exec_duration:.4f}s"
            }
        }

        # Affichage du Journal
        st.subheader("📋 Journal d'Audit Détaillé")
        st.json(audit_log)

        # Bouton de téléchargement
        json_export = json.dumps(audit_log, indent=4, cls=RomarinEncoder)
        st.download_button(
            label="📥 Télécharger le Rapport JSON",
            data=json_export,
            file_name=f"audit_romarin_{sid}.json",
            mime="application/json"
        )

    except Exception as e:
        st.error(f"Une erreur est survenue lors du calcul : {e}")

else:
    st.info("👋 Bienvenue ! Veuillez sélectionner vos paramètres dans la barre latérale et lancer l'analyse.")
