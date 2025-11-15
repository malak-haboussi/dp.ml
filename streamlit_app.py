import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# ==================== CSS ET CONFIGURATION GLOBALE ====================
st.set_page_config(
    page_title="Système Intelligent Sonatrach", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# CSS pour un design moderne (identique à la version précédente)
st.markdown("""
    <style>
    /* Généralités */
    .stApp {
        background-color: #F8F9FA;
    }
    h1, h2, h3, h4 {
        color: #0B2E59;
        font-family: 'Arial', sans-serif;
    }
    .stContainer {
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 15px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        background-color: white;
    }
    .metric-container {
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        margin-top: 10px;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.05);
    }
    .high-risk {
        background-color: #F8D7DA;
        border: 2px solid #DC3545;
    }
    .medium-risk {
        background-color: #FFF3CD;
        border: 2px solid #FFC107;
    }
    .low-risk {
        background-color: #D4EDDA;
        border: 2px solid #28A745;
    }
    .stProgress > div > div > div > div {
        background-color: #0B2E59;
    }
    </style>
    """, unsafe_allow_html=True)


# ==================== DONNÉES D'EXEMPLE ET MISE À JOUR ====================

# Utilisation de st.session_state pour stocker les données et simuler la mise à jour
if 'df_historique' not in st.session_state:
    st.session_state.df_historique = pd.DataFrame({
        'equipement': ['Pompe P-001', 'Compresseur C-245', 'Vanne V-128', 'Pompe P-002', 'Compresseur C-101', 'Pompe P-003', 'Vanne V-056', 'Compresseur C-389', 'Pompe P-004', 'Vanne V-201', 'Pompe P-005', 'Compresseur C-412', 'Vanne V-078', 'Pompe P-006', 'Compresseur C-225', 'Vanne V-145', 'Pompe P-007', 'Compresseur C-331', 'Vanne V-089', 'Pompe P-008'],
        'vibration': [4.2, 7.8, 3.1, 5.6, 8.1, 4.8, 2.9, 7.2, 5.1, 3.4, 6.9, 8.3, 3.2, 4.5, 7.5, 3.0, 5.8, 7.9, 2.8, 6.5],
        'temperature': [85, 92, 78, 88, 94, 86, 77, 91, 87, 79, 90, 95, 76, 84, 93, 78, 89, 92, 75, 88],
        'heures_fonctionnement': [1200, 1750, 800, 1450, 1820, 1100, 750, 1680, 1320, 820, 1580, 1900, 780, 1250, 1720, 790, 1420, 1780, 760, 1520],
        'stock_pieces': [15, 2, 25, 8, 1, 18, 30, 3, 12, 22, 5, 0, 28, 16, 4, 26, 9, 2, 32, 7],
        'delai_fournisseur': [7, 15, 5, 10, 20, 8, 4, 18, 9, 6, 12, 25, 5, 8, 16, 4, 11, 19, 3, 13],
        'risque_panne': [0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1],
        'risque_rupture_stock': [0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1]
    })

def mettre_a_jour_donnees():
    """Simule l'ajout de nouvelles données réelles pour améliorer l'entraînement."""
    df_nouveau = pd.DataFrame({
        'equipement': ['Pompe P-009', 'Compresseur C-999'],
        'vibration': [9.0, 4.0],
        'temperature': [98, 80],
        'heures_fonctionnement': [2000, 500],
        'stock_pieces': [0, 30],
        'delai_fournisseur': [25, 5],
        'risque_panne': [1, 0],
        'risque_rupture_stock': [1, 0]
    })
    # Concaténation et mise à jour de l'état de la session
    st.session_state.df_historique = pd.concat([st.session_state.df_historique, df_nouveau], ignore_index=True)
    st.toast("✅ Données historiques mises à jour avec 2 nouveaux enregistrements. Le modèle est ré-entraîné !", icon='🔄')

# ==================== MODÈLES MACHINE LEARNING ====================
# Utilisation de st.cache_resource pour que les modèles ne soient entraînés qu'une seule fois
@st.cache_resource
def entrainer_modeles(df):
    st.info(f"Modèles ré-entraînés sur {len(df)} enregistrements historiques.", icon="🧠")
    
    # Modèle pour prédire les pannes
    X_panne = df[['vibration', 'temperature', 'heures_fonctionnement']]
    y_panne = df['risque_panne']
    model_panne = RandomForestClassifier(n_estimators=100, random_state=42)
    model_panne.fit(X_panne, y_panne)
    
    # Modèle pour prédire les ruptures de stock
    X_stock = df[['stock_pieces', 'delai_fournisseur']]
    y_stock = df.loc[X_stock.index, 'risque_rupture_stock'] # S'assurer que les indices correspondent
    model_stock = RandomForestClassifier(n_estimators=100, random_state=42)
    model_stock.fit(X_stock, y_stock)
    
    return model_panne, model_stock

# ==================== FONCTION PRINCIPALE ====================
def main():
    
    if "vibration_val" not in st.session_state:
        st.session_state.vibration_val = 5.0
        st.session_state.temperature_val = 85
        st.session_state.heures_val = 1000
        st.session_state.stock_val = 10
        st.session_state.delai_val = 7
        st.session_state.resultat_panne = None
        st.session_state.resultat_stock = None
    
    st.markdown("<h1><span style='color:#DC3545;'>🔥</span> SYSTÈME INTELLIGENT DE RÉSILIENCE INDUSTRIELLE</h1>", unsafe_allow_html=True)
    st.markdown("---")
    
    # Le modèle est ré-entraîné chaque fois que les données changent (grâce au @st.cache_resource)
    df = st.session_state.df_historique
    model_panne, model_stock = entrainer_modeles(df)
    
    # -----------------------------------------------------------------
    # OPTION DE MISE À JOUR DES DONNÉES
    # -----------------------------------------------------------------
    st.sidebar.header("Gestion des Données Historiques")
    st.sidebar.write(f"Nombre d'enregistrements actuels : **{len(df)}**")
    
    # Bouton de mise à jour
    if st.sidebar.button("🔄 Mettre à Jour les Données Historiques", use_container_width=True):
        mettre_a_jour_donnees()
        st.rerun() # Recharger l'application pour utiliser les nouvelles données
    st.sidebar.markdown("---")


    # -----------------------------------------------------------------
    # SECTION 1: SAISIE DES PARAMÈTRES ET PRÉDICTION EN TEMPS RÉEL
    # -----------------------------------------------------------------
    st.markdown("<h2>🎯 Prédiction des Risques en Temps Réel (Module IA)</h2>", unsafe_allow_html=True)
    
    with st.container(border=True):
        col_input_1, col_input_2 = st.columns(2)
        
        with col_input_1:
            st.markdown("<h4>🔧 Entrées pour le Risque de Panne</h4>", unsafe_allow_html=True)
            vibration = st.slider("Niveau de Vibration (usure)", 0.0, 10.0, st.session_state.vibration_val, key="vibration_input")
            temperature = st.slider("Température Actuelle (°C)", 50, 120, st.session_state.temperature_val, key="temperature_input")
            heures = st.slider("Heures Cumulées de Fonctionnement (fatigue)", 0, 2000, st.session_state.heures_val, key="heures_input")
        
        with col_input_2:
            st.markdown("<h4>📦 Entrées pour le Risque de Rupture de Stock</h4>", unsafe_allow_html=True)
            stock = st.number_input("Stock Actuel de Pièces (unités)", 0, 50, st.session_state.stock_val, key="stock_input")
            delai = st.slider("Délai Moyen Fournisseur (jours)", 0, 30, st.session_state.delai_val, key="delai_input")

        def prediction_callback():
            # Prédiction panne (Probabilité de panne)
            risque_panne = model_panne.predict_proba([[vibration, temperature, heures]])[0][1]
            # Prédiction rupture stock (Probabilité de rupture)
            risque_stock = model_stock.predict_proba([[stock, delai]])[0][1]
            
            st.session_state.resultat_panne = risque_panne
            st.session_state.resultat_stock = risque_stock
            # Mémoriser les valeurs saisies par l'utilisateur
            st.session_state.vibration_val = vibration
            st.session_state.temperature_val = temperature
            st.session_state.heures_val = heures
            st.session_state.stock_val = stock
            st.session_state.delai_val = delai
            
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
                alerte = "🚨 RISQUE CRITIQUE"
                action = "**MAINTENANCE PRÉVENTIVE URGENTE** (Évite la panne)"
            elif risque_panne > 0.4:
                classe_css = 'medium-risk'
                alerte = "⚠️ RISQUE MODÉRÉ"
                action = "**SURVEILLANCE RENFORCÉE** et planification MRO"
            else:
                classe_css = 'low-risk'
                alerte = "✅ RISQUE FAIBLE"
                action = "**MAINTENANCE PROGRAMMÉE** (Routinière)"

            st.markdown(f"<div class='metric-container {classe_css}'>", unsafe_allow_html=True)
            st.markdown(f"<h4>🔧 {alerte}</h4>", unsafe_allow_html=True)
            st.markdown(f"**Probabilité de Panne:** **{risque_panne:.1%}**")
            st.progress(risque_panne)
            st.markdown(f"<p><strong>Action Recommandée:</strong> {action}</p>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        # --- Carte 2: Risque de Rupture de Stock ---
        with col_result_2:
            if risque_stock > 0.7:
                classe_css = 'high-risk'
                alerte = "🚨 RISQUE CRITIQUE"
                action = "**COMMANDE URGENTE** (Réapprovisionnement immédiat)"
            elif risque_stock > 0.4:
                classe_css = 'medium-risk'
                alerte = "⚠️ RISQUE MODÉRÉ"
                action = "**RÉAPPROVISIONNEMENT PLANIFIÉ** (Gérer le délai)"
            else:
                classe_css = 'low-risk'
                alerte = "✅ RISQUE FAIBLE"
                action = "**SURVEILLANCE NORMALE** du cycle de stock"

            st.markdown(f"<div class='metric-container {classe_css}'>", unsafe_allow_html=True)
            st.markdown(f"<h4>📦 {alerte}</h4>", unsafe_allow_html=True)
            st.markdown(f"**Probabilité de Rupture:** **{risque_stock:.1%}**")
            st.progress(risque_stock)
            st.markdown(f"<p><strong>Action Recommandée:</strong> {action}</p>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

    # -----------------------------------------------------------------
    # SECTION 3: TABLEAU DE BORD GLOBAL
    # -----------------------------------------------------------------
    st.markdown("<br><br><h2>📋 Vue d'Ensemble des Actifs Opérationnels</h2>", unsafe_allow_html=True)
    
    # Tableaux et KPIs (inchangés, mais utilisent les données mises à jour)
    def style_risque(val):
        if val == 1:
            return 'background-color: #F8D7DA; color: #DC3545; font-weight: bold'
        else:
            return 'background-color: #D4EDDA; color: #28A745'
    
    st.dataframe(df.style.applymap(style_risque, subset=['risque_panne', 'risque_rupture_stock']), use_container_width=True, height=250)
    
    st.markdown("---")
    st.caption("Démo pour votre Encadrant : Intégration IA (Prévision) et RO (Décision) - PFE Sonatrach")

if __name__ == "__main__":
    main()
