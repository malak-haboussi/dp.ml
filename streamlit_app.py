import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# Configuration de la page avec un thème sombre/pro
st.set_page_config(page_title="Sonatrach Predict-Stock", layout="wide", initial_sidebar_state="expanded")

# CSS pour le style (Police Urbanist, bords arrondis, couleurs Sonatrach)
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stMetric { background-color: #ffffff; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.05); }
    .stDataFrame { border-radius: 10px; }
    div[data-testid="stMetricValue"] { color: #00843D; }
    </style>
    """, unsafe_allow_html=True)

# --- CHARGEMENT DES DONNÉES ---
@st.cache_data
def get_data():
    data = {
        'Article': ['538Y042219', '586L015592', '584C110991', '584W011710', '584C110457', '584C113270', '538Y042606', '538Y030905'],
        'Stock': [11707, 1007, 376, 0, 1339, 0, 1154, 526],
        'Prevision_30j': [932.47, 996.17, 1265.39, 1078.42, 984.58, 915.20, 1093.43, 1008.50],
        'Rupture_J': [30, 30, 10, 0, 30, 0, 11, 4]
    }
    return pd.DataFrame(data)

df = get_data()

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/fr/4/4b/Logo_Sonatrach.svg", width=150)
    st.title("Navigation")
    st.info("Utilisez ce tableau de bord pour anticiper les ruptures de stock sur la Classe A.")
    st.markdown("---")
    st.write("👨‍🎓 **Projet de Fin d'Études**")

# --- HEADER ---
st.title("📊 Tableau de Bord Prédictif DAT")
st.markdown("Analyse proactive des stocks par **Deep Learning (LSTM)**")

# --- SECTION 1 : KEY METRICS (Cartes design) ---
col1, col2, col3, col4 = st.columns(4)
ruptures_count = len(df[df['Stock'] == 0])
alertes_count = len(df[(df['Rupture_J'] > 0) & (df['Rupture_J'] <= 7)])

col1.metric("Articles Monitorés", f"{len(df)}")
col2.metric("En Rupture", ruptures_count, delta=f"{ruptures_count}", delta_color="inverse")
col3.metric("Alertes Critiques", alertes_count, delta="-20%", delta_color="normal")
col4.metric("Disponibilité Globale", "84%")

st.markdown("---")

# --- SECTION 2 : GRAPHIQUES INTERACTIFS ---
c_left, c_right = st.columns([1.2, 0.8])

with c_left:
    st.subheader("🗺️ Matrice Temporelle de Disponibilité")
    # Création d'une heatmap avec Plotly (beaucoup plus beau que Seaborn)
    matrice = []
    for r in df['Rupture_J']:
        ligne = [1]*r + [0]*(30-r) if r < 30 else [1]*30
        matrice.append(ligne)
    
    fig_heat = px.imshow(matrice, 
                         labels=dict(x="Jours Futurs", y="Articles", color="Statut"),
                         y=df['Article'],
                         x=[f"J+{i}" for i in range(1,31)],
                         color_continuous_scale=['#FF4B4B', '#00CC96'])
    fig_heat.update_layout(height=450, coloraxis_showscale=False)
    st.plotly_chart(fig_heat, use_container_width=True)

with c_right:
    st.subheader("📦 Niveau de Stock Actuel")
    fig_bar = px.bar(df, x='Article', y='Stock', color='Rupture_J',
                     color_continuous_scale='RdYlGn',
                     title="Volume vs Risque")
    st.plotly_chart(fig_bar, use_container_width=True)

# --- SECTION 3 : TABLEAU DÉTAILLÉ & FILTRES ---
st.subheader("📋 Liste des articles à haute priorité")

def format_statut(val):
    if val == 0: return '🔴 RUPTURE IMMÉDIATE'
    if val <= 7: return f'🟠 ALERTE J+{val}'
    return '🟢 SÉCURISÉ'

df['Statut'] = df['Rupture_J'].apply(format_statut)

# Affichage avec style Streamlit natif amélioré
st.dataframe(df[['Article', 'Stock', 'Prevision_30j', 'Statut']].sort_values('Stock'), 
             use_container_width=True, 
             hide_index=True)

st.success("💡 Conseil : Lancez un bon de commande pour les articles en orange avant la fin de semaine.")
