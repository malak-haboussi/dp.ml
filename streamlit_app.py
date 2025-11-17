import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime, timedelta

# --- MODULE RO AVEC OPTIMISATION ÉCONOMIQUE ---
COUT_STOCK_UNITAIRE = 5.0      # Coût de possession par unité par jour
COUT_RUPTURE_UNITAIRE = 50.0   # Coût d'une rupture de stock
DEMANDE_MOYENNE_JOUR = 0.5     # Demande moyenne historique
ECART_TYPE_DEMANDE = 0.2       # Variabilité de la demande

def calculer_cout_total(stock_actuel, stock_cible, probabilite_rupture):
    """Calcule le coût total pour différents scénarios de stock"""
    
    # Coût de possession du stock
    cout_possession = stock_cible * COUT_STOCK_UNITAIRE
    
    # Coût de rupture attendu (espérance mathématique)
    cout_rupture_attendu = probabilite_rupture * COUT_RUPTURE_UNITAIRE
    
    # Coût total
    cout_total = cout_possession + cout_rupture_attendu
    
    return {
        'cout_possession': cout_possession,
        'cout_rupture_attendu': cout_rupture_attendu,
        'cout_total': cout_total
    }

def optimiser_decision_ro(
        stock_actuel: int, 
        delai_fournisseur: int, 
        probabilite_rupture_ia: float, 
        probabilite_panne_ia: float
) -> dict:
    """OPTIMISATION RO AVEC CALCUL ÉCONOMIQUE COMPLET"""
    
    # 1. Ajustement intelligent de la demande basé sur le risque de panne
    facteur_risque_panne = 1.0 + (probabilite_panne_ia * 0.8)  # Impact plus fort
    demande_ajustee = DEMANDE_MOYENNE_JOUR * facteur_risque_panne
    
    # 2. Calcul du besoin pendant le délai de livraison
    besoin_delai = demande_ajustee * delai_fournisseur
    
    # 3. Stock de sécurité basé sur le risque de rupture et la variabilité
    z_score = 1.96  # Pour 95% de niveau de service
    stock_securite = (z_score * ECART_TYPE_DEMANDE * np.sqrt(delai_fournisseur) + 
                     probabilite_rupture_ia * 8)
    
    # 4. Stock cible optimal
    stock_cible_optimal = max(besoin_delai + stock_securite, 5)  # Minimum 5 unités
    
    # 5. CALCUL ÉCONOMIQUE - Comparaison des scénarios
    scenarios = []
    
    # Scénario 1 : Commander pour atteindre le stock optimal
    cout_commande = calculer_cout_total(stock_actuel, stock_cible_optimal, probabilite_rupture_ia)
    
    # Scénario 2 : Ne rien commander (garder stock actuel)
    cout_actuel = calculer_cout_total(stock_actuel, stock_actuel, probabilite_rupture_ia)
    
    # 6. Décision optimale basée sur les coûts
    economie_potentielle = cout_actuel['cout_total'] - cout_commande['cout_total']
    
    if economie_potentielle > 0 and stock_cible_optimal > stock_actuel:
        quantite_commander = int(np.ceil(stock_cible_optimal - stock_actuel))
        recommandation = f"🚀 **Commander {quantite_commander} unités** (Économie: {economie_potentielle:.1f}€)"
        decision_optimale = "COMMANDER"
    else:
        quantite_commander = 0
        recommandation = f"✅ **Maintenir stock actuel** (Optimal économique)"
        decision_optimale = "MAINTENIR"
    
    return {
        "demande_ajustee_jour": demande_ajustee,
        "besoin_delai_livraison": besoin_delai,
        "stock_securite_calcule": stock_securite,
        "stock_cible_optimal": stock_cible_optimal,
        "quantite_a_commander": quantite_commander,
        "recommandation_ro": recommandation,
        "decision_optimale": decision_optimale,
        "analyse_economique": {
            "scenario_commande": cout_commande,
            "scenario_actuel": cout_actuel,
            "economie_potentielle": economie_potentielle
        }
    }

def estimer_delai_panne(vibration, temperature, heures_fonctionnement):
    """Estime le délai probable avant panne en fonction des paramètres techniques"""
    
    # Facteurs de risque pondérés
    score_vibration = max(0, (vibration - 3) / 7)
    score_temperature = max(0, (temperature - 80) / 40)
    score_heures = min(1, heures_fonctionnement / 2000)
    
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

# ==================== CSS ET CONFIGURATION ====================
st.set_page_config(
    page_title="Système Intelligent Sonatrach", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
    <style>
    .stApp { background-color: #F8F9FA; }
    h1, h2, h3, h4 { color: #0B2E59; font-family: 'Arial', sans-serif; }
    .metric-container { padding: 15px; border-radius: 10px; text-align: center; margin-top: 10px; box-shadow: 0 2px 5px rgba(0, 0, 0, 0.05); }
    .high-risk { background-color: #F8D7DA; border: 2px solid #DC3545; }
    .medium-risk { background-color: #FFF3CD; border: 2px solid #FFC107; }
    .low-risk { background-color: #D4EDDA; border: 2px solid #28A745; }
    .ro-card { background-color: #E3F2FD; border: 3px solid #1976D2; padding: 20px; border-radius: 10px; margin-top: 20px; }
    .economie-card { background-color: #E8F5E8; border: 2px solid #28A745; padding: 15px; border-radius: 10px; }
    .timeline-badge { 
        background-color: #0B2E59; color: white; padding: 5px 10px; 
        border-radius: 15px; font-size: 0.8em; margin: 5px 0; display: inline-block;
    }
    .cout-positive { color: #28A745; font-weight: bold; }
    .cout-negative { color: #DC3545; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# ==================== DONNÉES ET INITIALISATION ====================

if 'df_historique' not in st.session_state:
    st.session_state.df_historique = pd.DataFrame({
        'equipement': ['Pompe P-001', 'Compresseur C-245', 'Vanne V-128', 'Pompe P-002', 'Compresseur C-101'],
        'vibration': [4.2, 7.8, 3.1, 5.6, 8.1],
        'temperature': [85, 92, 78, 88, 94],
        'heures_fonctionnement': [1200, 1750, 800, 1450, 1820],
        'stock_pieces': [15, 2, 25, 8, 1],
        'delai_fournisseur': [7, 15, 5, 10, 20],
        'risque_panne': [0, 1, 0, 0, 1],
        'risque_rupture_stock': [0, 1, 0, 1, 1]
    })
    
    # Initialisation
    st.session_state.vibration_val = 5.0
    st.session_state.temperature_val = 85
    st.session_state.heures_val = 1000
    st.session_state.stock_val = 10
    st.session_state.delai_val = 7
    st.session_state.resultat_panne = None
    st.session_state.resultat_stock = None
    st.session_state.resultat_ro = None
    st.session_state.delai_panne = "NON CALCULÉ"
    st.session_state.niveau_urgence = "INCONNU"

def mettre_a_jour_donnees():
    """Simule l'ajout de nouvelles données historiques"""
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
    st.toast("✅ Données historiques mises à jour !", icon='🔄')

@st.cache_resource
def entrainer_modeles(df):
    """Entraîne les modèles IA"""
    # Modèle Panne
    X_panne = df[['vibration', 'temperature', 'heures_fonctionnement']]
    y_panne = df['risque_panne']
    model_panne = RandomForestClassifier(n_estimators=100, random_state=42)
    model_panne.fit(X_panne, y_panne)
    
    # Modèle Stock
    X_stock = df[['stock_pieces', 'delai_fournisseur']]
    y_stock = df['risque_rupture_stock']
    model_stock = RandomForestClassifier(n_estimators=100, random_state=42)
    model_stock.fit(X_stock, y_stock)
    
    return model_panne, model_stock

def executer_prediction(model_panne, model_stock, vibration, temperature, heures, stock, delai):
    """Effectue la prédiction IA et l'optimisation RO"""
    
    with st.spinner("Optimisation économique en cours..."):
        # Prédiction IA
        risque_panne = model_panne.predict_proba([[vibration, temperature, heures]])[0][1]
        risque_stock = model_stock.predict_proba([[stock, delai]])[0][1]
    
    # Estimation temporelle
    delai_panne, niveau_urgence = estimer_delai_panne(vibration, temperature, heures)
    
    # OPTIMISATION RO AVEC CALCUL ÉCONOMIQUE
    resultat_ro = optimiser_decision_ro(stock, delai, risque_stock, risque_panne)
    
    # Mise à jour session
    st.session_state.resultat_panne = risque_panne
    st.session_state.resultat_stock = risque_stock
    st.session_state.resultat_ro = resultat_ro
    st.session_state.delai_panne = delai_panne
    st.session_state.niveau_urgence = niveau_urgence

# ==================== APPLICATION PRINCIPALE ====================
def main():
    
    st.markdown("<h1><span style='color:#DC3545;'>🔥</span> SYSTÈME INTELLIGENT SONATRACH - IA + RO</h1>", unsafe_allow_html=True)
    st.markdown("---")
    
    df = st.session_state.df_historique
    model_panne, model_stock = entrainer_modeles(df)
    
    # Sidebar
    st.sidebar.header("📊 Gestion des Données")
    st.sidebar.write(f"Enregistrements : **{len(df)}**")
    if st.sidebar.button("🔄 Mettre à Jour les Données", use_container_width=True):
        mettre_a_jour_donnees()
        st.rerun()

    # SECTION 1: SAISIE DES PARAMÈTRES
    st.markdown("<h2>🎯 Paramètres d'Entrée</h2>", unsafe_allow_html=True)
    
    with st.container(border=True):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("<h4>🔧 Données Équipement</h4>", unsafe_allow_html=True)
            vibration = st.slider("Vibration", 0.0, 10.0, st.session_state.vibration_val)
            temperature = st.slider("Température (°C)", 50, 120, st.session_state.temperature_val)
            heures = st.slider("Heures Fonctionnement", 0, 2000, st.session_state.heures_val)
        
        with col2:
            st.markdown("<h4>📦 Données Stock</h4>", unsafe_allow_html=True)
            stock = st.number_input("Stock Actuel (unités)", 0, 50, st.session_state.stock_val)
            delai = st.slider("Délai Fournisseur (jours)", 0, 30, st.session_state.delai_val)

        def on_analyze_click():
            st.session_state.vibration_val = vibration
            st.session_state.temperature_val = temperature
            st.session_state.heures_val = heures
            st.session_state.stock_val = stock
            st.session_state.delai_val = delai
            executer_prediction(model_panne, model_stock, vibration, temperature, heures, stock, delai)
            
        st.button("🔍 ANALYSER ET OPTIMISER", type="primary", use_container_width=True, on_click=on_analyze_click)
    
    # EXÉCUTION INITIALE
    if st.session_state.resultat_panne is None:
        executer_prediction(
            model_panne, model_stock, 
            st.session_state.vibration_val, st.session_state.temperature_val, st.session_state.heures_val,
            st.session_state.stock_val, st.session_state.delai_val
        )
        st.rerun()

    # SECTION 2: RÉSULTATS IA
    risque_panne = st.session_state.resultat_panne
    risque_stock = st.session_state.resultat_stock
    resultat_ro = st.session_state.resultat_ro
    delai_panne = st.session_state.delai_panne
    niveau_urgence = st.session_state.niveau_urgence

    st.markdown("<br><h3>📈 Résultats de l'Analyse IA</h3>", unsafe_allow_html=True)
    
    col_result_1, col_result_2 = st.columns(2)

    with col_result_1:
        if risque_panne > 0.7:
            classe_css, alerte, action = 'high-risk', "🚨 RISQUE CRITIQUE", "Maintenance URGENTE"
        elif risque_panne > 0.4:
            classe_css, alerte, action = 'medium-risk', "⚠️ RISQUE MODÉRÉ", "Surveillance Renforcée"
        else:
            classe_css, alerte, action = 'low-risk', "✅ RISQUE FAIBLE", "Maintenance Routinière"

        st.markdown(f"<div class='metric-container {classe_css}'>", unsafe_allow_html=True)
        st.markdown(f"<h4>🔧 Panne Prédite : {alerte}</h4>", unsafe_allow_html=True)
        st.markdown(f"**Probabilité IA:** **{risque_panne:.1%}**")
        st.progress(risque_panne)
        badge_color = "#DC3545" if niveau_urgence == "CRITIQUE" else "#FFC107" if niveau_urgence == "ÉLEVÉ" else "#28A745"
        st.markdown(f"<div class='timeline-badge' style='background-color: {badge_color};'>⏱️ {delai_panne}</div>", unsafe_allow_html=True)
        st.markdown(f"**Action:** {action}")
        st.markdown("</div>", unsafe_allow_html=True)

    with col_result_2:
        if risque_stock > 0.7:
            classe_css, alerte = 'high-risk', "🚨 RISQUE CRITIQUE"
        elif risque_stock > 0.4:
            classe_css, alerte = 'medium-risk', "⚠️ RISQUE MODÉRÉ"
        else:
            classe_css, alerte = 'low-risk', "✅ RISQUE FAIBLE"

        st.markdown(f"<div class='metric-container {classe_css}'>", unsafe_allow_html=True)
        st.markdown(f"<h4>📦 Rupture Prédite : {alerte}</h4>", unsafe_allow_html=True)
        st.markdown(f"**Probabilité IA:** **{risque_stock:.1%}**")
        st.progress(risque_stock)
        st.markdown(f"**Décision RO:** {resultat_ro['recommandation_ro']}")
        st.markdown("</div>", unsafe_allow_html=True)
    
    # SECTION 3: OPTIMISATION RO DÉTAILLÉE
    st.markdown("<br><h3>💰 Optimisation Économique (Recherche Opérationnelle)</h3>", unsafe_allow_html=True)
    
    with st.container(border=True):
        st.markdown("<div class='ro-card'>", unsafe_allow_html=True)
        
        col_ro1, col_ro2 = st.columns(2)
        
        with col_ro1:
            st.metric("Stock Cible Optimal", f"{resultat_ro['stock_cible_optimal']:.1f} unités")
            st.metric("Quantité à Commander", f"{resultat_ro['quantite_a_commander']} unités")
            st.metric("Décision Optimale", resultat_ro['decision_optimale'])
            
        with col_ro2:
            st.markdown("**Calculs RO :**")
            st.markdown(f"- Demande ajustée: {resultat_ro['demande_ajustee_jour']:.2f} unités/jour")
            st.markdown(f"- Besoin délai: {resultat_ro['besoin_delai_livraison']:.1f} unités")
            st.markdown(f"- Stock sécurité: {resultat_ro['stock_securite_calcule']:.1f} unités")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # SECTION 4: ANALYSE ÉCONOMIQUE DÉTAILLÉE
    st.markdown("<br><h4>📊 Analyse Économique Détaillée</h4>", unsafe_allow_html=True)
    
    with st.container(border=True):
        col_eco1, col_eco2 = st.columns(2)
        
        with col_eco1:
            st.markdown("**Scénario Actuel (Ne rien changer):**")
            cout_actuel = resultat_ro['analyse_economique']['scenario_actuel']
            st.markdown(f"- Coût possession: {cout_actuel['cout_possession']:.1f}€")
            st.markdown(f"- Coût rupture attendu: {cout_actuel['cout_rupture_attendu']:.1f}€")
            st.markdown(f"- **Coût total: {cout_actuel['cout_total']:.1f}€**")
        
        with col_eco2:
            st.markdown("**Scénario Optimal (Commander):**")
            cout_commande = resultat_ro['analyse_economique']['scenario_commande']
            st.markdown(f"- Coût possession: {cout_commande['cout_possession']:.1f}€")
            st.markdown(f"- Coût rupture attendu: {cout_commande['cout_rupture_attendu']:.1f}€")
            st.markdown(f"- **Coût total: {cout_commande['cout_total']:.1f}€**")
        
        economie = resultat_ro['analyse_economique']['economie_potentielle']
        if economie > 0:
            st.markdown(f"<div class='economie-card'>", unsafe_allow_html=True)
            st.markdown(f"### 💰 ÉCONOMIE POTENTIELLE: {economie:.1f}€")
            st.markdown("**La décision RO vous fait économiser de l'argent !**")
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div style='padding: 15px; border-radius: 10px; background-color: #E3F2FD;'>", unsafe_allow_html=True)
            st.markdown(f"### ✅ SITUATION OPTIMALE")
            st.markdown("**Votre stock actuel est déjà à son niveau économique optimal**")
            st.markdown("</div>", unsafe_allow_html=True)

    # SECTION 5: TABLEAU DE BORD
    st.markdown("<br><h3>📋 Vue d'Ensemble des Actifs</h3>", unsafe_allow_html=True)
    
    def style_risque(val):
        return 'background-color: #F8D7DA; color: #DC3545; font-weight: bold' if val == 1 else 'background-color: #D4EDDA; color: #28A745'
    
    styled_df = df.style.map(style_risque, subset=['risque_panne', 'risque_rupture_stock'])
    st.dataframe(styled_df, use_container_width=True, height=300)
    
    st.markdown("---")
    st.caption("PFE Sonatrach - Système Intelligent IA + Recherche Opérationnelle")

if __name__ == "__main__":
    main()
