import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime, timedelta

# --- INTÉGRATION TEMPORAIRE DU MODULE RO (Recherche Opérationnelle) ---
COUT_STOCK_UNITAIRE = 5.0    
COUT_RUPTURE_UNITAIRE = 50.0 
DEMANDE_MOYENNE_JOUR = 0.5   
ECART_TYPE_DEMANDE = 0.1     

def estimer_delai_panne(vibration, temperature, heures_fonctionnement):
    """Estime le délai probable avant panne en fonction des paramètres techniques"""
    
    # Facteurs de risque pondérés
    score_vibration = max(0, (vibration - 3) / 7)  # 0-1, vibration > 3 considérée comme risquée
    score_temperature = max(0, (temperature - 80) / 40)  # 0-1, température > 80°C risquée
    score_heures = min(1, heures_fonctionnement / 2000)  # 0-1, proportionnel aux heures
    
    # Score de risque global
    score_risque = 0.4 * score_vibration + 0.4 * score_temperature + 0.2 * score_heures
    
    # Conversion du score en délai estimé
    if score_risque > 0.8:
        return "DANS 1-7 JOURS", "CRITIQUE"
    elif score_risque > 0.6:
        return "DANS 8-15 JOURS", "ÉLEVÉ"
    elif score_risque > 0.4:
        return "DANS 16-30 JOURS", "MODÉRÉ"
    elif score_risque > 0.2:
        return "DANS 1-2 MOIS", "FAIBLE"
    else:
        return "AU-DELÀ DE 2 MOIS", "NÉGLIGEABLE"

def optimiser_decision_ro(
        stock_actuel: int, 
        delai_fournisseur: int, 
        probabilite_rupture_ia: float, 
        probabilite_panne_ia: float
) -> dict:
    """Calcule la quantité de pièces à commander Q et le stock de sécurité optimal S."""
    
    # 1. Ajustement de la Demande par le Risque de Panne (Lien IA <-> RO)
    facteur_risque = 1.0 + (probabilite_panne_ia * 0.5) 
    demande_ajustee = DEMANDE_MOYENNE_JOUR * facteur_risque

    # 2. Stock Cible 
    stock_cible = demande_ajustee * delai_fournisseur
    
    # 3. Stock de Sécurité Optimal (S)
    buffer_securite = probabilite_rupture_ia * 5 
    stock_securite_optimal = stock_cible + buffer_securite
    
    # 4. Décision Finale (Quantité à commander)
    Q_a_commander = int(max(0, np.ceil(stock_securite_optimal) - stock_actuel))
    
    # 5. Génération de la Recommandation
    reco_finale = ""
    if Q_a_commander > 0:
        reco_finale = f"Commander **{Q_a_commander}** unités."
        if probabilite_rupture_ia > 0.7:
             reco_finale += " (Priorité : URGENTE)"
        elif probabilite_rupture_ia > 0.4:
             reco_finale += " (Priorité : Planifiée)"
    else:
        reco_finale = f"Aucune commande nécessaire. Stock de {stock_actuel} suffisant."

    return {
        "stock_securite_optimal": stock_securite_optimal,
        "quantite_a_commander": Q_a_commander,
        "demande_ajustee_jour": demande_ajustee,
        "recommandation_ro": reco_finale,
    }

# ==================== CSS ET CONFIGURATION GLOBALE ====================
st.set_page_config(
    page_title="Système Intelligent Sonatrach", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
    <style>
    .stApp { background-color: #F8F9FA; }
    h1, h2, h3, h4 { color: #0B2E59; font-family: 'Arial', sans-serif; }
    .stContainer { border-radius: 12px; padding: 20px; margin-bottom: 15px; box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1); background-color: white; }
    .metric-container { padding: 15px; border-radius: 10px; text-align: center; margin-top: 10px; box-shadow: 0 2px 5px rgba(0, 0, 0, 0.05); }
    .high-risk { background-color: #F8D7DA; border: 2px solid #DC3545; }
    .medium-risk { background-color: #FFF3CD; border: 2px solid #FFC107; }
    .low-risk { background-color: #D4EDDA; border: 2px solid #28A745; }
    .ro-card { background-color: #E3F2FD; border: 3px solid #1976D2; padding: 20px; border-radius: 10px; margin-top: 20px; box-shadow: 0 4px 8px rgba(25, 118, 210, 0.2); }
    .stProgress > div > div > div > div { background-color: #0B2E59; }
    .timeline-badge { 
        background-color: #0B2E59; 
        color: white; 
        padding: 5px 10px; 
        border-radius: 15px; 
        font-size: 0.8em;
        margin: 5px 0;
        display: inline-block;
    }
    </style>
    """, unsafe_allow_html=True)

# ==================== DONNÉES D'EXEMPLE ET MISE À JOUR ====================

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
    
    # Initialisation des valeurs par défaut pour les curseurs
    st.session_state.vibration_val = 5.0
    st.session_state.temperature_val = 85
    st.session_state.heures_val = 1000
    st.session_state.stock_val = 10
    st.session_state.delai_val = 7
    st.session_state.resultat_panne = None
    st.session_state.resultat_stock = None
    st.session_state.resultat_ro = None
    st.session_state.delai_panne = "NON CALCULÉ"  # Initialisation ajoutée
    st.session_state.niveau_urgence = "INCONNU"   # Initialisation ajoutée

def mettre_a_jour_donnees():
    """Simule l'ajout de nouvelles données historiques et ré-entraîne le modèle."""
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
    st.session_state.df_historique = pd.concat([st.session_state.df_historique, df_nouveau], ignore_index=True)
    st.toast("✅ Données historiques mises à jour avec 2 nouveaux enregistrements. Le modèle est ré-entraîné !", icon='🔄')

@st.cache_resource
def entrainer_modeles(df):
    """Entraîne les deux modèles (Panne et Stock) sur les données actuelles."""
    st.info(f"Modèles ré-entraînés sur {len(df)} enregistrements historiques.", icon="🧠")
    
    # Modèle Panne (IA 1)
    X_panne = df[['vibration', 'temperature', 'heures_fonctionnement']]
    y_panne = df['risque_panne']
    model_panne = RandomForestClassifier(n_estimators=100, random_state=42)
    model_panne.fit(X_panne, y_panne)
    
    # Modèle Stock (IA 2)
    X_stock = df[['stock_pieces', 'delai_fournisseur']]
    y_stock = df.loc[X_stock.index, 'risque_rupture_stock']
    model_stock = RandomForestClassifier(n_estimators=100, random_state=42)
    model_stock.fit(X_stock, y_stock)
    
    return model_panne, model_stock

def executer_prediction(model_panne, model_stock, vibration, temperature, heures, stock, delai):
    """Effectue la prédiction IA et le calcul RO, puis met à jour l'état de la session."""
    
    # 1. PRÉDICTION IA (Probabilités de risque)
    with st.spinner("Analyse en cours..."):
        risque_panne = model_panne.predict_proba([[vibration, temperature, heures]])[0][1]
        risque_stock = model_stock.predict_proba([[stock, delai]])[0][1]
    
    # 2. ESTIMATION TEMPORELLE DE LA PANNE
    delai_panne, niveau_urgence = estimer_delai_panne(vibration, temperature, heures)
    
    # 3. OPTIMISATION RO (Calcul de la meilleure action)
    resultat_ro = optimiser_decision_ro(
        stock_actuel=stock,
        delai_fournisseur=delai,
        probabilite_rupture_ia=risque_stock,
        probabilite_panne_ia=risque_panne
    )
    
    # Mise à jour de l'état de la session
    st.session_state.resultat_panne = risque_panne
    st.session_state.resultat_stock = risque_stock
    st.session_state.resultat_ro = resultat_ro
    st.session_state.delai_panne = delai_panne
    st.session_state.niveau_urgence = niveau_urgence

# ==================== FONCTION PRINCIPALE ====================
def main():
    
    st.markdown("<h1><span style='color:#DC3545;'>🔥</span> SYSTÈME INTELLIGENT DE RÉSILIENCE INDUSTRIELLE</h1>", unsafe_allow_html=True)
    st.markdown("---")
    
    df = st.session_state.df_historique
    model_panne, model_stock = entrainer_modeles(df)
    
    # -----------------------------------------------------------------
    # OPTION DE MISE À JOUR DES DONNÉES (Sidebar)
    # -----------------------------------------------------------------
    st.sidebar.header("Gestion des Données Historiques")
    st.sidebar.write(f"Nombre d'enregistrements actuels : **{len(df)}**")
    
    if st.sidebar.button("🔄 Mettre à Jour les Données Historiques", use_container_width=True):
        mettre_a_jour_donnees()
        executer_prediction(
            model_panne, model_stock, 
            st.session_state.vibration_val, st.session_state.temperature_val, st.session_state.heures_val,
            st.session_state.stock_val, st.session_state.delai_val
        )
        st.rerun() 
    st.sidebar.markdown("---")

    # -----------------------------------------------------------------
    # SECTION 1: SAISIE DES PARAMÈTRES ET DÉCLENCHEMENT
    # -----------------------------------------------------------------
    st.markdown("<h2>🎯 Prédiction des Risques (Module IA)</h2>", unsafe_allow_html=True)
    
    with st.container(border=True):
        col_input_1, col_input_2 = st.columns(2)
        
        with col_input_1:
            st.markdown("<h4>🔧 Entrées pour le Risque de Panne</h4>", unsafe_allow_html=True)
            vibration = st.slider("Niveau de Vibration", 0.0, 10.0, st.session_state.vibration_val, key="vibration_input")
            temperature = st.slider("Température Actuelle (°C)", 50, 120, st.session_state.temperature_val, key="temperature_input")
            heures = st.slider("Heures Cumulées de Fonctionnement", 0, 2000, st.session_state.heures_val, key="heures_input")
        
        with col_input_2:
            st.markdown("<h4>📦 Entrées pour le Risque de Rupture de Stock</h4>", unsafe_allow_html=True)
            stock = st.number_input("Stock Actuel de Pièces (unités)", 0, 50, st.session_state.stock_val, key="stock_input")
            delai = st.slider("Délai Moyen Fournisseur (jours)", 0, 30, st.session_state.delai_val, key="delai_input")

        # Fonction de rappel pour le bouton
        def on_analyze_click():
            # Mise à jour des valeurs stockées pour qu'elles persistent
            st.session_state.vibration_val = vibration
            st.session_state.temperature_val = temperature
            st.session_state.heures_val = heures
            st.session_state.stock_val = stock
            st.session_state.delai_val = delai
            executer_prediction(model_panne, model_stock, vibration, temperature, heures, stock, delai)
            
        st.button("🔍 ANALYSER LES RISQUES & GÉNÉRER LA DÉCISION OPTIMALE", type="primary", use_container_width=True, on_click=on_analyze_click)
    
    # -----------------------------------------------------------------
    # EXÉCUTION INITIALE LORS DU PREMIER CHARGEMENT (Garantit l'affichage)
    # -----------------------------------------------------------------
    if st.session_state.resultat_panne is None:
        executer_prediction(
            model_panne, model_stock, 
            st.session_state.vibration_val, st.session_state.temperature_val, st.session_state.heures_val,
            st.session_state.stock_val, st.session_state.delai_val
        )
        st.rerun() 

    # -----------------------------------------------------------------
    # SECTION 2: RÉSULTATS DE PRÉDICTION & RECOMMANDATIONS (OUTPUT IA + RO)
    # -----------------------------------------------------------------
    
    risque_panne = st.session_state.resultat_panne
    risque_stock = st.session_state.resultat_stock
    resultat_ro = st.session_state.resultat_ro
    delai_panne = st.session_state.delai_panne
    niveau_urgence = st.session_state.niveau_urgence

    st.markdown("<br><h3>Résultats de l'Analyse IA</h3>", unsafe_allow_html=True)
    
    col_result_1, col_result_2 = st.columns(2)

    # --- Carte 1: Risque de Panne ---
    with col_result_1:
        if risque_panne > 0.7:
            classe_css = 'high-risk'
            alerte = "🚨 RISQUE CRITIQUE"
            action = "Maintenance Préventive URGENTE."
        elif risque_panne > 0.4:
            classe_css = 'medium-risk'
            alerte = "⚠️ RISQUE MODÉRÉ"
            action = "Surveillance et Planification."
        else:
            classe_css = 'low-risk'
            alerte = "✅ RISQUE FAIBLE"
            action = "Maintenance Routinière."

        st.markdown(f"<div class='metric-container {classe_css}'>", unsafe_allow_html=True)
        st.markdown(f"<h4>🔧 Panne Prédite : {alerte}</h4>", unsafe_allow_html=True)
        st.markdown(f"**Probabilité (IA):** **{risque_panne:.1%}**")
        st.progress(risque_panne)
        
        # AFFICHAGE DU DÉLAI ESTIMÉ
        badge_color = "#DC3545" if niveau_urgence == "CRITIQUE" else "#FFC107" if niveau_urgence == "ÉLEVÉ" else "#28A745"
        st.markdown(f"<div class='timeline-badge' style='background-color: {badge_color};'>⏱️ {delai_panne}</div>", unsafe_allow_html=True)
        
        st.markdown(f"<p><strong>Décision Opérationnelle:</strong> {action}</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # --- Carte 2: Risque de Rupture de Stock ---
    with col_result_2:
        if risque_stock > 0.7:
            classe_css = 'high-risk'
            alerte = "🚨 RISQUE CRITIQUE"
        elif risque_stock > 0.4:
            classe_css = 'medium-risk'
            alerte = "⚠️ RISQUE MODÉRÉ"
        else:
            classe_css = 'low-risk'
            alerte = "✅ RISQUE FAIBLE"

        st.markdown(f"<div class='metric-container {classe_css}'>", unsafe_allow_html=True)
        st.markdown(f"<h4>📦 Rupture Prédite : {alerte}</h4>", unsafe_allow_html=True)
        st.markdown(f"**Probabilité (IA):** **{risque_stock:.1%}**")
        st.progress(risque_stock)
        st.markdown(f"<p><strong>Décision Optimale (RO):</strong> {resultat_ro['recommandation_ro']}</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    # --- Carte 3: Résultat Détaillé de l'Optimisation RO ---
    st.markdown("<br><h3>📈 Détails de l'Optimisation Logistique (Recherche Opérationnelle)</h3>", unsafe_allow_html=True)
    st.markdown("<div class='ro-card'>", unsafe_allow_html=True)
    
    col_ro_1, col_ro_2 = st.columns(2)

    with col_ro_1:
         st.metric(
            "Stock de Sécurité Optimal (S)", 
            f"{resultat_ro['stock_securite_optimal']:.1f} Unités", 
            help="Niveau de stock ciblé pour minimiser les coûts."
        )
         st.metric(
            "Quantité à Commander (Q)", 
            f"{resultat_ro['quantite_a_commander']} Unités", 
            help="Quantité calculée par la RO pour atteindre S."
        )
    
    with col_ro_2:
        st.markdown(f"**Facteurs pris en compte par la RO :**")
        st.markdown(f"- Risque de Panne (IA) : {risque_panne:.1%}")
        st.markdown(f"- Risque de Rupture (IA) : {risque_stock:.1%}")
        st.markdown(f"- Demande journalière ajustée : {resultat_ro['demande_ajustee_jour']:.2f} unités.")
        st.markdown(f"- Délai estimé avant panne : **{delai_panne}**")

    st.markdown(f"**Conclusion RO :** **{resultat_ro['recommandation_ro']}**", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # -----------------------------------------------------------------
    # SECTION 3: TABLEAU DE BORD GLOBAL
    # -----------------------------------------------------------------
    st.markdown("<br><br><h2>📋 Vue d'Ensemble des Actifs Opérationnels</h2>", unsafe_allow_html=True)
    
    def style_risque(val):
        if val == 1:
            return 'background-color: #F8D7DA; color: #DC3545; font-weight: bold'
        else:
            return 'background-color: #D4EDDA; color: #28A745'
    
    styled_df = df.style.map(style_risque, subset=['risque_panne', 'risque_rupture_stock'])
    st.dataframe(styled_df, use_container_width=True, height=250)
    
    st.markdown("---")
    st.caption("PFE Sonatrach - Intégration IA et RO")

if __name__ == "__main__":
    main()
