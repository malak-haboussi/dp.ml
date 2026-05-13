import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Configuration de la page
st.set_page_config(page_title="Dashboard Prédictif Sonatrach", layout="wide")

# --- DONNÉES RÉELLES (FUSION DE VOS STOCKS ET PRÉVISIONS) ---
@st.cache_data
def load_data():
    # Données basées sur vos extractions précédentes
    stocks_reels = {
        '538Y042219': 11707, '586L015592': 1007, '584C110991': 376, '538Y041201': 1359,
        '584W011710': 0, '584C110457': 1339, '584C113270': 0, '584W010711': 0,
        '538Y042606': 1154, '538Y030905': 526, '584C030965': 0, '536Y200600': 0,
        '584C030232': 108, '538Y042632': 628, '588W662595': 23405, '538Y042626': 43,
        '584J250350': 4834, '584C110023': 457, '586M038493': 225, '586L015590': 0,
        '584J250270': 8802, '588W662525': 0, '584W010713': 0
    }
    
    # Simulation des jours de rupture basés sur vos résultats LSTM
    # (Remplacez par vos vraies valeurs si nécessaire)
    lignes = []
    for item, stock in stocks_reels.items():
        # Simulation d'une conso moyenne pour déduire un jour de rupture théorique
        conso_moy_simulee = 40 
        jours_restants = 0 if stock == 0 else int(stock / conso_moy_simulee)
        
        status = "✅ OK"
        if stock == 0: status = "❌ RUPTURE STOCK"
        elif jours_restants <= 7: status = f"⚠️ CRITIQUE (J+{jours_restants})"
        elif jours_restants <= 30: status = f"🟠 ATTENTION (J+{jours_restants})"
        
        lignes.append({
            'Code Article': item,
            'Stock Actuel': stock,
            'Besoin Prévu (30j)': round(conso_moy_simulee * 30, 2),
            'Jours Restants': jours_restants,
            'Statut': status
        })
    return pd.DataFrame(lignes)

df = load_data()

# --- INTERFACE STREAMLIT ---
st.title("🛢️ Sonatrach - Aide à la Décision Logistique")
st.subheader("Direction Approvisionnement et Transport (DAT)")

# Indicateurs rapides
c1, c2, c3 = st.columns(3)
c1.metric("Articles en Rupture", len(df[df['Stock Actuel'] == 0]))
c2.metric("Alertes Critiques (<7j)", len(df[(df['Jours Restants'] > 0) & (df['Jours Restants'] <= 7)]))
c3.metric("Stock Total (Classe A)", f"{df['Stock Actuel'].sum():,.0f}")

# Heatmap Globale
st.write("### 🗺️ Cartographie de Disponibilité (Horizon 30 Jours)")
matrice = []
for idx, row in df.iterrows():
    if row['Jours Restants'] == 0 and row['Stock Actuel'] == 0:
        matrice.append([0]*30)
    elif row['Jours Restants'] > 30:
        matrice.append([1]*30)
    else:
        matrice.append([1]*row['Jours Restants'] + [0]*(30-row['Jours Restants']))

fig, ax = plt.subplots(figsize=(12, 7))
sns.heatmap(matrice, cmap=['#e74c3c', '#2ecc71'], cbar=False, yticklabels=df['Code Article'], xticklabels=range(1, 31))
ax.set_xlabel("Jours futurs")
st.pyplot(fig)

# Tableau détaillé
st.write("### 📋 Détails des articles")
st.dataframe(df.sort_values('Jours Restants'), use_container_width=True)
