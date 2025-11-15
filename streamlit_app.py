import streamlit as st
from pulp import *

# --- 1. CONFIGURATION ET TITRE ---
st.set_page_config(layout="wide")
st.title("Système Intelligent de Prévision et Optimisation des Risques")
st.subheader("Démo pour votre Encadrant : Intégration IA (Prévision) et RO (Décision)")
st.caption("Projet : Rupture de la Chaîne Logistique dans l'Industrie Pétrolière et Gazière")
st.write("---")

# --- 2. MODULE IA SIMULÉ (PRÉVISION) ---
st.header("🟢 Module IA (Entrée) : Prévision du Risque de Panne")
st.markdown("*(Simule la sortie d'un modèle d'IA (XGBoost ou LSTM) basé sur les données de maintenance prédictive)*")

# Le Slider simule le résultat de votre modèle d'IA
probabilite_defaillance_pct = st.slider(
    "Probabilité Prédite de Défaillance de la Pompe dans les 30 Jours (%)",
    min_value=5, max_value=80, value=25, step=5,
)
probabilite_defaillance = probabilite_defaillance_pct / 100.0  # Convertir en décimal

st.metric(
    label="Probabilité de Défaillance (P_defaillance)",
    value=f"{probabilite_defaillance * 100:.1f}%",
    delta_color="off"
)

# --- 3. DÉFINITION DES PARAMÈTRES RO (COÛTS) ---
# Ces paramètres sont des données fixes de l'entreprise
cout_piece = 1500.0       
cout_rupture = 50000.0    # Coût très élevé pour une rupture de production

st.subheader("Paramètres de Coût (Entrées RO) :")
col1, col2 = st.columns(2)
col1.metric("Coût Unitaire de la Pièce (Stockage)", f"{cout_piece:.0f} €")
col2.metric("Coût d'une Rupture/Panne (Impact)", f"{cout_rupture:.0f} €")

# --- 4. MODULE RO (OPTIMISATION) ---
# Utilisation de @st.cache_data pour ne recalculer que si le slider bouge
@st.cache_data
def optimiser_stock(P_defaillance, C_piece, C_rupture):
    """Calcule le stock de sécurité optimal en minimisant les coûts totaux."""
    
    prob = LpProblem("Optimisation_Stock_Securite", LpMinimize)

    # Variable de décision : Stock de Sécurité (S)
    S = LpVariable("Stock_Securite", lowBound=0, upBound=20, cat='Integer')
    
    # PARAMÈTRES DU MODÈLE RO SIMPLIFIÉ
    taux_detention_annuel = 0.10  
    max_stock_hypothetique = 20.0 

    # OBJECTIF : Minimiser Coût Total = Coût de Stockage + Coût du Risque Résiduel
    
    # Coût de Stockage
    cout_stockage = S * C_piece * taux_detention_annuel 

    # Coût du Risque Résiduel (simplifié pour démo): P_defaillance * Coût Rupture * (1 - S / Max_Stock)
    # Rendu linéaire pour PuLP
    cout_risque_residuel = P_defaillance * C_rupture * (1 - S * (1 / max_stock_hypothetique))
    
    prob += cout_stockage + cout_risque_residuel, "Minimisation_Cout_Total"

    # CONTRAINTES (Le stock minimum augmente avec le risque prédit par l'IA)
    if P_defaillance > 0.30:
         prob += S >= 4, "Contrainte_Service_Haut_Risque"
    elif P_defaillance > 0.10:
         prob += S >= 2, "Contrainte_Service_Moyen_Risque" 
    else:
         prob += S >= 1, "Contrainte_Service_Bas_Risque" 

    # Résolution du problème
    prob.solve(PULP_CBC_CMD(msg=0)) 

    # Retourner le résultat
    if LpStatus[prob.status] == "Optimal":
        return value(S), value(prob.objective)
    else:
        return "Échec", 0

# --- 5. EXÉCUTION ET AFFICHAGE DES RÉSULTATS ---
stock_optimal, cout_total_min = optimiser_stock(probabilite_defaillance, cout_piece, cout_rupture)

st.write("---")
st.header("🔴 Module RO (Sortie) : Décision Optimale (Prescription)")

col3, col4 = st.columns(2)

if stock_optimal != "Échec":
    col3.metric(
        label="Stock de Sécurité Optimal Recommandé",
        value=f"{int(stock_optimal)} unités",
        delta="Décision Prescriptive",
        delta_color="normal"
    )
    col4.metric(
        label="Coût Total Minimisé (Stock + Risque Résiduel)",
        value=f"{cout_total_min:.2f} €",
    )
    
    st.info(f"**Analyse :** Pour un risque de défaillance de **{probabilite_defaillance_pct}%**, le système recommande un stock de **{int(stock_optimal)}** pièces pour minimiser le coût total à **{cout_total_min:.2f} €**.")
else:
    st.error("Le solveur PuLP n'a pas pu trouver une solution optimale. Vérifiez la formulation du modèle.")
