import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# ==================== CSS ET CONFIGURATION GLOBALE ====================
st.set_page_config(
    page_title="Système Intelligent Sonatrach", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# CSS pour un design moderne, des cartes d'information et une meilleure typographie
st.markdown("""
    <style>
    /* Généralités */
    .stApp {
        background-color: #F8F9FA; /* Fond très clair */
    }
    h1, h2, h3, h4 {
        color: #0B2E59; /* Bleu foncé - couleur d'entreprise/industrielle */
        font-family: 'Arial', sans-serif;
    }
    
    /* Style des cartes de saisie/résultats */
    .stContainer {
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 15px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        background-color: white;
    }

    /* Style spécifique pour les métriques de risque */
    .metric-container {
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        margin-top: 10px;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.05);
    }
    .high-risk {
        background-color: #F8D7DA; /* Rouge très léger */
        border: 2px solid #DC3545;
    }
    .medium-risk {
        background-color: #FFF3CD; /* Jaune très léger */
        border: 2px solid #FFC107;
    }
    .low-risk {
        background-color: #D4EDDA; /* Vert très léger */
        border: 2px solid #28A745;
    }
    .stProgress > div > div > div > div {
        background-color: #0B2E59; /* Barre de progression en bleu */
    }
    </style>
    """, unsafe_allow_html=True)


# ==================== DONNÉES D'EXEMPLE ====================
@st.cache_data
def creer_donnees():
    # Données enrichies pour plus de réalisme
    data = {
        'equipement': ['Pompe P-001', 'Compresseur C-245', 'Vanne V-128', 'Pompe P-002', 
                       'Compresseur C-101', 'Pompe P-003', 'Vanne V-056', 'Compresseur C-389',
                       'Pompe P-004', 'Vanne V-201', 'Pompe P-005', 'Compresseur C-412',
                       'Vanne V-078', 'Pompe P-006', 'Compresseur C-225', 'Vanne V-145',
                       'Pompe P-007', 'Compresseur C-331', 'Vanne V-089', 'Pompe P-008'],
        
        'vibration': [4.2, 7.8, 3.1, 5.6, 8.1, 4.8, 2.9, 7.2, 5.1, 3.4, 6.9, 8.3, 3.2, 4.5, 7.5, 3.0, 5.8, 7.9, 2.8, 6.5],
        'temperature': [85, 92, 78, 88, 94, 86, 77, 91, 87, 79, 90, 95, 76, 84, 93, 78, 89, 92, 75, 88],
        'heures_fonctionnement': [1200, 1750, 800, 1450, 1820, 1100, 750, 1680, 1320, 820, 1580, 1900, 780, 1250, 1720, 790, 1420, 1780, 760, 1520],
        
        'stock_pieces': [15, 2, 25, 8, 1, 18, 30, 3, 12, 22, 5, 0, 28, 16, 4, 26, 9, 2, 32, 7],
        'delai_fournisseur': [7, 15, 5, 10, 20, 8, 4, 18, 9, 6, 12, 25, 5, 8, 16, 4, 11, 19, 3, 13],
        
        'risque_panne': [0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1],
        'risque_rupture_stock': [0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1]
    }
    return pd.DataFrame(data)

# ==================== MODÈLES MACHINE LEARNING ====================
# Utilisation de st.cache_resource pour que les modèles ne soient entraînés qu'une seule fois
@st.cache_resource
def entrainer_modeles(df):
    # Modèle pour prédire les pannes
    X_panne = df[['vibration', 'temperature', 'heures_fonctionnement']]
    y_panne = df['risque_panne']
    model_panne = RandomForestClassifier(n_estimators=100, random_state=42)
    model_panne.fit(X_panne, y_panne)
    
    # Modèle pour prédire les ruptures de stock
    X_stock = df[['stock_pieces', 'delai_fournisseur']]
    y_stock = df['risque_rupture_stock']
    model_stock = RandomForestClassifier(n_estimators=100, random_state=42)
    model_stock.fit(X_stock, y_stock)
    
    return model_panne, model_stock

# ==================== FONCTION PRINCIPALE ====================
def main():
    
    # Initialisation des variables d'état pour les scénarios
    if "vibration" not in st.session_state:
        st.session_state.vibration = 5.0
        st.session_state.temperature = 85
        st.session_state.heures = 1000
        st.session_state.stock = 10
        st.session_state.delai = 7
        st.session_state.resultat_panne = None
        st.session_state.resultat_stock = None
    
    # Titre principal et logo (simulé)
    st.markdown("<h1><span style='color:#DC3545;'>🔥</span> SYSTÈME INTELLIGENT DE RÉSILIENCE INDUSTRIELLE</h1>", unsafe_allow_html=True)
    st.markdown("---")
    
    # Chargement des données et entraînement des modèles
    df = creer_donnees()
    model_panne, model_stock = entrainer_modeles(df)
    
    # -----------------------------------------------------------------
    # SECTION 1: SAISIE DES PARAMÈTRES ET PRÉDICTION EN TEMPS RÉEL
    # -----------------------------------------------------------------
    st.markdown("<h2>🎯 Prédiction des Risques en Temps Réel (Module IA)</h2>", unsafe_allow_html=True)
    
    # Utilisation d'un conteneur stylisé pour la saisie
    with st.container(border=True):
        col_input_1, col_input_2 = st.columns(2)
        
        with col_input_1:
            st.markdown("<h4>🔧 Données Conditionnelles de l'Équipement</h4>", unsafe_allow_html=True)
            vibration = st.slider("Niveau de Vibration (0-10)", 0.0, 10.0, st.session_state.vibration, key="vibration_input")
            temperature = st.slider("Température Actuelle (°C)", 50, 120, st.session_state.temperature, key="temperature_input")
            heures = st.slider("Heures Cumulées de Fonctionnement", 0, 2000, st.session_state.heures, key="heures_input")
        
        with col_input_2:
            st.markdown("<h4>📦 Données Logistiques du Rechange</h4>", unsafe_allow_html=True)
            stock = st.number_input("Stock Actuel de Pièces (unités)", 0, 50, st.session_state.stock, key="stock_input")
            delai = st.slider("Délai Moyen Fournisseur (jours)", 0, 30, st.session_state.delai, key="delai_input")

        # Fonction de prédiction
        def prediction_callback():
            # Prédiction panne (Probabilité de panne)
            risque_panne = model_panne.predict_proba([[vibration, temperature, heures]])[0][1]
            
            # Prédiction rupture stock (Probabilité de rupture)
            risque_stock = model_stock.predict_proba([[stock, delai]])[0][1]
            
            st.session_state.resultat_panne = risque_panne
            st.session_state.resultat_stock = risque_stock
            
        st.button("🔍 ANALYSER LES RISQUES & GÉNÉRER LA DÉCISION", type="primary", use_container_width=True, on_click=prediction_callback)
    
    # -----------------------------------------------------------------
    # SECTION 2: RÉSULTATS DE PRÉDICTION & RECOMMANDATIONS (OUTPUT IA)
    # -----------------------------------------------------------------
    if st.session_state.resultat_panne is not None:
        st.markdown("<br><h3>Résultats de l'Analyse IA</h3>", unsafe_allow_html=True)
        
        risque_panne = st.session_state.resultat_panne
        risque_stock = st.session_state.resultat_stock

        col_result_1, col_result_2 = st.columns(2)

        # --- Carte 1: Risque de Panne ---
        with col_result_1:
            if risque_panne > 0.7:
                classe_css = 'high-risk'
                alerte = "🚨 RISQUE CRITIQUE - Panne Probable"
                action = "**MAINTENANCE PRÉVENTIVE URGENTE**"
                priorite = "🔴 URGENT"
            elif risque_panne > 0.4:
                classe_css = 'medium-risk'
                alerte = "⚠️ RISQUE MODÉRÉ - Surveillance Requise"
                action = "**SURVEILLANCE RENFORCÉE** et planification MRO"
                priorite = "🟠 MOYENNE"
            else:
                classe_css = 'low-risk'
                alerte = "✅ RISQUE FAIBLE - Situation Stable"
                action = "**MAINTENANCE PROGRAMMÉE** (Routinière)"
                priorite = "🟢 FAIBLE"

            st.markdown(f"<div class='metric-container {classe_css}'>", unsafe_allow_html=True)
            st.markdown(f"<h4>🔧 {alerte}</h4>", unsafe_allow_html=True)
            st.markdown(f"**Probabilité de Panne:** **{risque_panne:.1%}**")
            st.progress(risque_panne)
            st.markdown(f"<p><strong>Décision IA/RO:</strong> {action}</p>", unsafe_allow_html=True)
            st.markdown(f"<p><strong>Priorité Opérationnelle:</strong> <strong>{priorite}</strong></p>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        # --- Carte 2: Risque de Rupture de Stock ---
        with col_result_2:
            # (Note: ici devrait s'intégrer le module RO, mais pour la démo, on simule la décision)
            if risque_stock > 0.7:
                classe_css = 'high-risk'
                alerte = "🚨 RISQUE CRITIQUE - Rupture de Stock Imminente"
                action = "**COMMANDE URGENTE** (Express Aérien)"
                priorite = "🔴 URGENT"
            elif risque_stock > 0.4:
                classe_css = 'medium-risk'
                alerte = "⚠️ RISQUE MODÉRÉ - Niveau de Stock Dangereux"
                action = "**RÉAPPROVISIONNEMENT RAPIDE** (Planifié)"
                priorite = "🟠 MOYENNE"
            else:
                classe_css = 'low-risk'
                alerte = "✅ RISQUE FAIBLE - Couverture de Stock Optimale"
                action = "**SURVEILLANCE NORMALE** du cycle de stock"
                priorite = "🟢 FAIBLE"

            st.markdown(f"<div class='metric-container {classe_css}'>", unsafe_allow_html=True)
            st.markdown(f"<h4>📦 {alerte}</h4>", unsafe_allow_html=True)
            st.markdown(f"**Probabilité de Rupture:** **{risque_stock:.1%}**")
            st.progress(risque_stock)
            st.markdown(f"<p><strong>Décision IA/RO:</strong> {action}</p>", unsafe_allow_html=True)
            st.markdown(f"<p><strong>Priorité Logistique:</strong> <strong>{priorite}</strong></p>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)


    # -----------------------------------------------------------------
    # SECTION 3: TABLEAU DE BORD GLOBAL ET STATISTIQUES
    # -----------------------------------------------------------------
    st.markdown("<br><br><h2>📋 Vue d'Ensemble des Actifs Opérationnels</h2>", unsafe_allow_html=True)
    
    # Création d'un tableau stylisé
    def style_risque(val):
        if val == 1:
            return 'background-color: #F8D7DA; color: #DC3545; font-weight: bold' # Critique
        else:
            return 'background-color: #D4EDDA; color: #28A745' # Faible

    # Affichage du Tableau de bord
    st.markdown("<h3>Synthèse des Risques par Équipement</h3>", unsafe_allow_html=True)
    styled_df = df.style.applymap(style_risque, subset=['risque_panne', 'risque_rupture_stock'])
    st.dataframe(styled_df, use_container_width=True, height=250)
    
    st.markdown("<h3>Indicateurs Clés de Performance (KPI)</h3>", unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Actifs Surveillés", "20 Unités")
    with col2:
        st.metric("Actifs à Risque Panne", f"{df['risque_panne'].sum()}/20", delta=f"{df['risque_panne'].sum()/20:.1%} du parc", delta_color="inverse")
    with col3:
        st.metric("Actifs à Risque Rupture", f"{df['risque_rupture_stock'].sum()}/20", delta=f"{df['risque_rupture_stock'].sum()/20:.1%} du parc", delta_color="inverse")
    with col4:
        equipements_critiques = len(df[(df['risque_panne'] == 1) & (df['risque_rupture_stock'] == 1)])
        st.metric("Actifs Critiques (Double Risque)", f"{equipements_critiques}/20", delta="Priorité Absolue", delta_color="inverse")

    # -----------------------------------------------------------------
    # SECTION 4: DEMONSTRATION DE SCÉNARIOS
    # -----------------------------------------------------------------
    st.markdown("<br><br><h2>🎭 Scénarios de Démonstration Rapide</h2>", unsafe_allow_html=True)
    st.caption("Cliquez pour charger des jeux de données d'entrée prédéfinis dans la section Prédiction.")

    col_scenario_1, col_scenario_2, col_scenario_3 = st.columns(3)
    
    # Cette fonction met à jour les st.session_state
    def update_session_state(v, t, h, s, d):
        st.session_state.vibration = v
        st.session_state.temperature = t
        st.session_state.heures = h
        st.session_state.stock = s
        st.session_state.delai = d
        st.session_state.resultat_panne = None # Réinitialiser le résultat pour forcer la re-prédiction
        st.rerun()

    with col_scenario_1:
        st.button("🔴 SCÉNARIO 1: URGENCE", use_container_width=True, 
                  on_click=update_session_state, args=(8.5, 95, 1900, 1, 20))
        st.caption("Haute vibration, faible stock, long délai.")
    
    with col_scenario_2:
        st.button("🟠 SCÉNARIO 2: PLANIFICATION", use_container_width=True,
                  on_click=update_session_state, args=(6.0, 88, 1200, 8, 12))
        st.caption("Risque modéré dans les deux domaines.")
    
    with col_scenario_3:
        st.button("🟢 SCÉNARIO 3: SÉCURITÉ", use_container_width=True,
                  on_click=update_session_state, args=(3.5, 78, 500, 25, 5))
        st.caption("Faibles indicateurs de panne, stock élevé.")

    st.markdown("---")
    st.caption("Démo pour votre Encadrant : Intégration IA (Prévision) et RO (Décision) - PFE Sonatrach")

if __name__ == "__main__":
    main()
