import streamlit as st
import pandas as pd
import numpy as np
from pulp import *

# --- 1. CONFIGURATION ET STYLISATION GLOBALE (CSS) ---
st.set_page_config(
    layout="wide", 
    page_title="Système IA + RO", 
    initial_sidebar_state="expanded"
)

# Petit ajout de CSS pour un look plus moderne (style des titres/cartes)
st.markdown("""
    <style>
    .stApp {
        background-color: #f7f9fd; /* Fond clair, doux */
    }
    h1, h2, h3 {
        color: #0b2e59; /* Couleur de titre bleue/marine */
    }
    .stMetric, .stContainer {
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        background-color: white;
    }
    .metric-label {
        font-weight: bold;
        color: #555;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🛡️ Système de Résilience Logistique (Pétrole & Gaz)")
st.header("Analyse Prédictive et Optimisation Prescriptive")
st.write("---")

# --- 2. MENU LATÉRAL : ENTRÉE DE DONNÉES ---
with st.sidebar:
    st.header("⚙️ Configuration des Données")
    st.markdown("---")
    
    # 2.1 Charger les données (Utilisation de votre exemple de ventes)
    if st.button("Charger l'Exemple (Ventes 6 Mois)", help="120, 132, 148, 165, 185, 208"):
        st.session_state['vendu'] = [120, 132, 148, 165, 185, 208]
    
    # 2.2 Entrée manuelle des données
    vendu_str = st.text_area(
        "Saisir les 6 dernières Périodes (k unités, séparées par des virgules) :",
        value=", ".join(map(str, st.session_state.get('vendu', [120, 132, 148, 165, 185, 208]))),
        key='input_ventes'
    )
    
    try:
        ventes_actuelles = [int(x.strip()) for x in vendu_str.split(',') if x.strip()]
        if len(ventes_actuelles) != 6:
            st.error("⚠️ Veuillez saisir exactement 6 valeurs.")
            st.stop()
    except ValueError:
        st.error("⚠️ Veuillez saisir uniquement des nombres entiers séparés par des virgules.")
        st.stop()

    st.markdown("---")
    # Ajout d'une zone pour les paramètres de coût
    st.subheader("Paramètres RO")
    cout_piece = st.number_input("Coût Unitaire du Stock (k€)", value=10.0, min_value=1.0)
    cout_rupture = st.number_input("Coût d'une Rupture (k€)", value=500.0, min_value=10.0)


# --- 3. ANALYSE ET PRÉVISION (MODULE IA) ---
st.markdown("## 🧠 Étape 1 : Analyse et Prévision (Module IA)")

# Calculs pour la prévision simplifiée
df_ventes = pd.DataFrame({
    'Mois': range(1, 7),
    'Ventes (k unités)': ventes_actuelles
})
demande_moyenne = np.mean(ventes_actuelles)
facteur_tendance = (ventes_actuelles[-1] - ventes_actuelles[0]) / 5
prevision_prochain_mois = demande_moyenne + facteur_tendance
probabilite_defaillance_pct = min(80, max(5, int(facteur_tendance * 2))) / 100.0 

# Conteneur pour la visualisation et l'aperçu
col_chart, col_data_preview = st.columns([3, 1])

with col_chart:
    st.markdown("### Évolution de la Demande Historique")
    st.line_chart(df_ventes, x='Mois', y='Ventes (k unités)', use_container_width=True, height=300)

with col_data_preview:
    st.markdown("### Données Brutes")
    # Utilisation d'un expander pour ne pas surcharger la vue
    with st.expander("Voir les 6 périodes", expanded=False):
        st.dataframe(df_ventes, hide_index=True, width=150)


# --- Section des Métriques IA (Cartes) ---
st.markdown("### Synthèse des Prévisions")
col_metrics_ia = st.columns(3)

with col_metrics_ia[0]:
    st.metric(label="Moyenne des Ventes", value=f"{demande_moyenne:.0f} k u.", delta_color="off")

with col_metrics_ia[1]:
    st.metric(label="Prévision Mois 7", value=f"{prevision_prochain_mois:.0f} k u.", delta=f"{facteur_tendance:.1f} k/mois (Tendance)", delta_color="normal")

with col_metrics_ia[2]:
    # La sortie principale du module IA
    st.metric(label="🔥 Probabilité de Rupture Prédite", value=f"{probabilite_defaillance_pct * 100:.1f}%", delta="Risque basé sur la demande", delta_color="inverse")

st.write("---")


# --- 4. OPTIMISATION PRESCRIPTIVE (MODULE RO) ---
st.markdown("## ⚖️ Étape 2 : Optimisation Prescriptive (Module RO)")
st.markdown("Le moteur de Recherche Opérationnelle utilise le risque (IA) pour calculer la décision la plus rentable.")


# --- Modèle RO (Fonction PuLP) ---
@st.cache_data
def optimiser_stock(P_rupture, P_demande, C_piece, C_rupture):
    prob = LpProblem("Optimisation_Stock_Securite", LpMinimize)
    S = LpVariable("Stock_Securite", lowBound=0, upBound=50, cat='Integer')
    
    taux_detention_annuel = 0.10  
    max_stock_hypothetique = 50.0 

    # OBJECTIF : Minimiser Coût Total = Coût de Stockage + Coût du Risque Résiduel
    cout_stockage = S * C_piece * taux_detention_annuel 
    cout_risque_residuel = P_rupture * C_rupture * (1 - S * (1 / max_stock_hypothetique))
    
    prob += cout_stockage + cout_risque_residuel, "Minimisation_Cout_Total"

    # CONTRAINTE : Le stock doit couvrir la prévision moyenne ajustée au risque
    prob += S >= (P_demande / 1000) * (1 + P_rupture * 0.5), "Contrainte_Service_Minimum"

    prob.solve(PULP_CBC_CMD(msg=0)) 

    if LpStatus[prob.status] == "Optimal":
        return value(S), value(prob.objective)
    else:
        return "Échec", 0

# Exécution et Affichage
stock_optimal, cout_total_min = optimiser_stock(
    probabilite_defaillance_pct, 
    prevision_prochain_mois, 
    cout_piece, 
    cout_rupture
)

col_results_ro = st.columns(3)

# 5. AFFICHAGE DES RÉSULTATS (Cartes RO)
if stock_optimal != "Échec":
    with col_results_ro[0]:
        st.metric(
            label="Stock Recommandé (k unités)",
            value=f"{int(stock_optimal):.0f}",
            delta="Décision Optimale (à commander)",
            delta_color="normal"
        )

    with col_results_ro[1]:
        st.metric(
            label="Coût Total Minimum (k€)",
            value=f"{cout_total_min:.2f}",
            delta_color="off"
        )
    
    with col_results_ro[2]:
        # Ajout d'une analyse de la valeur de la décision
        economie_potentielle = (cout_rupture * probabilite_defaillance_pct) - cout_total_min
        st.metric(
            label="Valeur Économique de la Décision",
            value=f"{economie_potentielle:.2f} k€",
            delta="Économie réalisée vs. coût de la rupture",
            delta_color="normal"
        )
        
    st.markdown("---")
    st.info(f"""
    **Synthèse de la Décision :** Pour faire face à une demande croissante (prévision) et un risque de rupture de **{probabilite_defaillance_pct * 100:.1f}%**,
    l'approche optimisée minimise le risque et les coûts, recommandant l'achat de **{int(stock_optimal)} k unités** pour un coût total attendu de **{cout_total_min:.2f} k€**.
    """)
else:
    st.error("Le modèle d'optimisation (RO) n'a pas pu être résolu. Veuillez vérifier les contraintes.")
