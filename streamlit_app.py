import streamlit as st
import pandas as pd
import numpy as np
from pulp import *

# --- 1. CONFIGURATION ET TITRE ---
st.set_page_config(
    layout="wide", 
    page_title="Démo IA + RO Logistique", 
    initial_sidebar_state="expanded" # Ouvre le menu latéral
)

st.title("🛡️ Système Intelligent de Prévision et Optimisation des Risques")
st.subheader("Démonstration : L'IA nourrit la Décision RO")
st.caption("Projet : Optimisation de la Chaîne Logistique dans l'Industrie Pétrolière et Gazière")

# --- MENU LATÉRAL POUR LA SÉLECTION DE DONNÉES ---
with st.sidebar:
    st.header("Paramètres de l'Exemple")
    
    # Bouton pour charger les données de votre exemple (Ventes 6 mois)
    if st.button("Charger l'Exemple 'Ventes sur 6 Mois'", help="Charge les données initiales : 120, 132, 148, 165, 185, 208"):
        st.session_state['vendu'] = [120, 132, 148, 165, 185, 208]
    
    # Entrée manuelle des données (pour plus de flexibilité)
    vendu_str = st.text_area(
        "Saisir les 6 dernières Ventes (séparées par des virgules) :",
        value=", ".join(map(str, st.session_state.get('vendu', [120, 132, 148, 165, 185, 208]))),
        key='input_ventes'
    )
    
    try:
        ventes_actuelles = [int(x.strip()) for x in vendu_str.split(',') if x.strip()]
        if len(ventes_actuelles) != 6:
            st.error("Veuillez saisir exactement 6 valeurs.")
            st.stop()
    except ValueError:
        st.error("Veuillez saisir uniquement des nombres entiers séparés par des virgules.")
        st.stop()

# --- DÉCLENCHEMENT DE L'ANALYSE ---
st.write("---")
st.header("🎯 Analyse des Ventes et Prévision de la Demande")

# 1. ANALYSE ET PRÉVISION (SIMULATION IA)
df_ventes = pd.DataFrame({
    'Mois': range(1, 7),
    'Ventes (k unités)': ventes_actuelles
})

# Calcul de la Demande Moyenne pour la prévision simplifiée
demande_moyenne = np.mean(ventes_actuelles)
facteur_tendance = (ventes_actuelles[-1] - ventes_actuelles[0]) / 5
prevision_prochain_mois = demande_moyenne + facteur_tendance

# SIMULATION IA : Calculer la "Probabilité de forte demande" basée sur la tendance.
# Si la tendance est très forte, le risque de rupture est plus élevé.
probabilite_defaillance_pct = min(80, max(5, int(facteur_tendance * 2))) / 100.0 # Simule le risque

# 2. AFFICHAGE DU TABLEAU DE BORD (Partie IA)
colA, colB = st.columns([1, 2])

with colA:
    st.markdown("#### 📊 Aperçu des Données")
    st.dataframe(df_ventes, hide_index=True, width=350)

    # Métriques de Prévision IA
    st.markdown("#### 🧠 Prévisions et Risque (Module IA)")
    st.metric(label="Moyenne des Ventes Historiques", value=f"{demande_moyenne:.0f} k unités")
    st.metric(label="Prévision du Prochain Mois", value=f"{prevision_prochain_mois:.0f} k unités")
    st.metric(
        label="🔥 Risque de Rupture Prédit (P_rupture)",
        value=f"{probabilite_defaillance_pct * 100:.1f}%",
        delta="Déterminé par la force de la tendance",
        delta_color="normal"
    )

with colB:
    st.markdown("#### Évolution des Ventes")
    st.line_chart(df_ventes, x='Mois', y='Ventes (k unités)', use_container_width=True)


# --- 3. MODULE RO (OPTIMISATION) ---
st.write("---")
st.header("⚖️ Optimisation de la Contre-Mesure (Module RO)")

# --- Paramètres de Coût (Utilisés pour PuLP) ---
cout_piece = 10.0       # Coût unitaire pour simplifier (k unités)
cout_rupture = 500.0    # Coût très élevé de la rupture (k unités)

# Modèle de Recherche Opérationnelle
@st.cache_data
def optimiser_stock(P_rupture, P_demande, C_piece, C_rupture):
    prob = LpProblem("Optimisation_Stock_Securite", LpMinimize)
    S = LpVariable("Stock_Securite", lowBound=0, upBound=50, cat='Integer')
    
    taux_detention_annuel = 0.10  
    max_stock_hypothetique = 50.0 

    # OBJECTIF : Minimiser Coût Total = Coût de Stockage + Coût du Risque
    cout_stockage = S * C_piece * taux_detention_annuel 
    cout_risque_residuel = P_rupture * C_rupture * (1 - S * (1 / max_stock_hypothetique))
    
    prob += cout_stockage + cout_risque_residuel, "Minimisation_Cout_Total"

    # CONTRAINTE : Le stock doit couvrir au moins la prévision moyenne plus 50% du risque.
    prob += S >= (P_demande / 1000) * (1 + P_rupture * 0.5), "Contrainte_Service_Minimum"

    prob.solve(PULP_CBC_CMD(msg=0)) 

    if LpStatus[prob.status] == "Optimal":
        return value(S), value(prob.objective)
    else:
        return "Échec", 0

# Exécution
stock_optimal, cout_total_min = optimiser_stock(
    probabilite_defaillance_pct, 
    prevision_prochain_mois, 
    cout_piece, 
    cout_rupture
)

# 4. AFFICHAGE DES RÉSULTATS (Partie RO)
colC, colD = st.columns(2)

with colC:
    st.metric(
        label="Stock de Sécurité Optimal Recommandé (RO)",
        value=f"{int(stock_optimal):.0f} k unités",
        delta="Décision Prescriptive",
        delta_color="normal"
    )

with colD:
    st.metric(
        label="Coût Total Minimum Attendu",
        value=f"{cout_total_min:.2f} k €",
    )
    
st.write("---")
st.success("""
**Conclusion de la Démo :** Le système a analysé la demande passée (IA), prédit le risque de rupture, et a utilisé la Recherche Opérationnelle (RO) pour fournir la **décision la plus économique** (quantité à commander) afin de couvrir ce risque.
""")
   
