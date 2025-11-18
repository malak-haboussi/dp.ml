import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# --- MODULE RO AVEC OPTIMISATION ÉCONOMIQUE ---
COUT_STOCK_UNITAIRE = 5.0
COUT_RUPTURE_UNITAIRE = 50.0
DEMANDE_MOYENNE_JOUR = 0.5
ECART_TYPE_DEMANDE = 0.2

def calculer_cout_total(stock_cible, probabilite_rupture):
    """Calcule le coût total pour un niveau de stock donné"""
    cout_possession = stock_cible * COUT_STOCK_UNITAIRE
    cout_rupture_attendu = probabilite_rupture * COUT_RUPTURE_UNITAIRE
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
    """OPTIMISATION RO CORRIGÉE - Prend en compte le risque de panne"""
    
    # 1. Ajustement PLUS FORT de la demande basé sur le risque de panne
    facteur_risque_panne = 1.0 + (probabilite_panne_ia * 1.5)  # Augmenté de 0.8 à 1.5
    demande_ajustee = DEMANDE_MOYENNE_JOUR * facteur_risque_panne
    
    # 2. Calcul du besoin pendant le délai de livraison
    besoin_delai = demande_ajustee * delai_fournisseur
    
    # 3. Stock de sécurité PLUS ÉLEVÉ en cas de risque de panne
    z_score = 1.96
    stock_securite = (z_score * ECART_TYPE_DEMANDE * np.sqrt(delai_fournisseur) + 
                     probabilite_rupture_ia * 10 +  # Augmenté de 8 à 10
                     probabilite_panne_ia * 5)      # NOUVEAU: ajout du risque panne
    
    # 4. Stock cible optimal
    stock_cible_optimal = max(besoin_delai + stock_securite, 8)  # Minimum augmenté à 8
    
    # 5. CALCUL ÉCONOMIQUE - Logique CORRIGÉE
    cout_commande = calculer_cout_total(stock_cible_optimal, probabilite_rupture_ia)
    cout_actuel = calculer_cout_total(stock_actuel, probabilite_rupture_ia)
    
    economie_potentielle = cout_actuel['cout_total'] - cout_commande['cout_total']
    
    # DÉCISION CORRIGÉE : Commander si nécessaire même avec faible risque rupture
    seuil_commande = stock_cible_optimal * 1.1  # 10% de marge
    
    if stock_actuel < seuil_commande or economie_potentielle > 0:
        quantite_commander = max(0, int(np.ceil(stock_cible_optimal - stock_actuel)))
        if quantite_commander > 0:
            if probabilite_panne_ia > 0.6:  # Si risque panne élevé
                recommandation = f"🚨 **Commander {quantite_commander} unités URGENT** (Risque panne élevé)"
            elif economie_potentielle > 0:
                recommandation = f"🚀 **Commander {quantite_commander} unités** (Économie: {economie_potentielle:.1f}€)"
            else:
                recommandation = f"📦 **Commander {quantite_commander} unités** (Stock sécurité)"
            decision_optimale = "COMMANDER"
        else:
            recommandation = f"✅ **Maintenir stock actuel** (Niveau optimal)"
            decision_optimale = "MAINTENIR"
    else:
        quantite_commander = 0
        recommandation = f"✅ **Maintenir stock actuel** (Suffisant)"
        decision_optimale = "MAINTENIR"
    
    return {
        "demande_ajustee": demande_ajustee,
        "besoin_delai": besoin_delai,
        "stock_securite": stock_securite,
        "stock_cible_optimal": stock_cible_optimal,
        "quantite_commander": quantite_commander,
        "recommandation": recommandation,
        "decision_optimale": decision_optimale,
        "analyse_economique": {
            "scenario_commande": cout_commande,
            "scenario_actuel": cout_actuel,
            "economie_potentielle": economie_potentielle
        }
    }

def estimer_delai_panne(vibration, temperature, heures_fonctionnement):
    """Estime le délai probable avant panne"""
    score_vibration = max(0, (vibration - 3) / 7)
    score_temperature = max(0, (temperature - 80) / 40)
    score_heures = min(1, heures_fonctionnement / 2000)
    
    score_risque = 0.4 * score_vibration + 0.4 * score_temperature + 0.2 * score_heures
    
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

# ==================== CONFIGURATION ====================
st.set_page_config(page_title="Système Intelligent Sonatrach", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #F8F9FA; }
    .metric-container { padding: 15px; border-radius: 10px; text-align: center; margin-top: 10px; box-shadow: 0 2px 5px rgba(0, 0, 0, 0.05); }
    .high-risk { background-color: #F8D7DA; border: 2px solid #DC3545; }
    .medium-risk { background-color: #FFF3CD; border: 2px solid #FFC107; }
    .low-risk { background-color: #D4EDDA; border: 2px solid #28A745; }
    .ro-card { background-color: #E3F2FD; border: 3px solid #1976D2; padding: 20px; border-radius: 10px; }
    .economie-card { background-color: #E8F5E8; border: 2px solid #28A745; padding: 15px; border-radius: 10px; }
    .urgence-card { background-color: #FFE6E6; border: 2px solid #DC3545; padding: 15px; border-radius: 10px; }
    .timeline-badge { background-color: #0B2E59; color: white; padding: 5px 10px; border-radius: 15px; font-size: 0.8em; margin: 5px 0; }
    </style>
    """, unsafe_allow_html=True)

# ==================== INITIALISATION ====================

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
    
    st.session_state.vibration_val = 7.0  # Augmenté pour simuler risque
    st.session_state.temperature_val = 95  # Augmenté pour simuler risque
    st.session_state.heures_val = 1800     # Augmenté pour simuler risque
    st.session_state.stock_val = 5         # Réduit pour tester commande
    st.session_state.delai_val = 10
    st.session_state.resultat_panne = 0.0
    st.session_state.resultat_stock = 0.0
    st.session_state.resultat_ro = {}
    st.session_state.delai_panne = "NON CALCULÉ"
    st.session_state.niveau_urgence = "INCONNU"

@st.cache_resource
def entrainer_modeles(df):
    """Entraîne les modèles IA"""
    X_panne = df[['vibration', 'temperature', 'heures_fonctionnement']]
    y_panne = df['risque_panne']
    model_panne = RandomForestClassifier(n_estimators=100, random_state=42)
    model_panne.fit(X_panne, y_panne)
    
    X_stock = df[['stock_pieces', 'delai_fournisseur']]
    y_stock = df['risque_rupture_stock']
    model_stock = RandomForestClassifier(n_estimators=100, random_state=42)
    model_stock.fit(X_stock, y_stock)
    
    return model_panne, model_stock

def executer_prediction(model_panne, model_stock, vibration, temperature, heures, stock, delai):
    """Effectue la prédiction IA et l'optimisation RO"""
    with st.spinner("Optimisation économique en cours..."):
        risque_panne = model_panne.predict_proba([[vibration, temperature, heures]])[0][1]
        risque_stock = model_stock.predict_proba([[stock, delai]])[0][1]
    
    delai_panne, niveau_urgence = estimer_delai_panne(vibration, temperature, heures)
    resultat_ro = optimiser_decision_ro(stock, delai, risque_stock, risque_panne)
    
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
    
    # SECTION 1: SAISIE DES PARAMÈTRES
    st.markdown("<h2>🎯 Paramètres d'Entrée</h2>", unsafe_allow_html=True)
    
    with st.container(border=True):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("<h4>🔧 Données Équipement</h4>", unsafe_allow_html=True)
            vibration = st.slider("Vibration", 0.0, 10.0, st.session_state.vibration_val, key="vib")
            temperature = st.slider("Température (°C)", 50, 120, st.session_state.temperature_val, key="temp")
            heures = st.slider("Heures Fonctionnement", 0, 2000, st.session_state.heures_val, key="heures")
        
        with col2:
            st.markdown("<h4>📦 Données Stock</h4>", unsafe_allow_html=True)
            stock = st.number_input("Stock Actuel (unités)", 0, 50, st.session_state.stock_val, key="stock")
            delai = st.slider("Délai Fournisseur (jours)", 0, 30, st.session_state.delai_val, key="delai")

        if st.button("🔍 ANALYSER ET OPTIMISER", type="primary", use_container_width=True):
            st.session_state.vibration_val = vibration
            st.session_state.temperature_val = temperature
            st.session_state.heures_val = heures
            st.session_state.stock_val = stock
            st.session_state.delai_val = delai
            executer_prediction(model_panne, model_stock, vibration, temperature, heures, stock, delai)
            st.rerun()
    
    # EXÉCUTION INITIALE
    if not st.session_state.resultat_ro:
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
        st.markdown(f"**Décision RO:** {resultat_ro.get('recommandation', 'En cours...')}")
        st.markdown("</div>", unsafe_allow_html=True)
    
    # SECTION 3: OPTIMISATION RO DÉTAILLÉE
    st.markdown("<br><h3>💰 Optimisation Économique (Recherche Opérationnelle)</h3>", unsafe_allow_html=True)
    
    with st.container(border=True):
        st.markdown("<div class='ro-card'>", unsafe_allow_html=True)
        
        col_ro1, col_ro2 = st.columns(2)
        
        with col_ro1:
            st.metric("Stock Cible Optimal", f"{resultat_ro.get('stock_cible_optimal', 0):.1f} unités")
            st.metric("Quantité à Commander", f"{resultat_ro.get('quantite_commander', 0)} unités")
            st.metric("Décision", resultat_ro.get('decision_optimale', 'EN ATTENTE'))
            
        with col_ro2:
            st.markdown("**Calculs RO :**")
            st.markdown(f"- Demande ajustée: {resultat_ro.get('demande_ajustee', 0):.2f} unités/jour")
            st.markdown(f"- Besoin délai: {resultat_ro.get('besoin_delai', 0):.1f} unités")
            st.markdown(f"- Stock sécurité: {resultat_ro.get('stock_securite', 0):.1f} unités")
            st.markdown(f"- Risque panne intégré: {risque_panne:.1%}")
        
        # ALERTE SI RISQUE PANNE ÉLEVÉ MAIS PAS DE COMMANDE
        if risque_panne > 0.6 and resultat_ro.get('quantite_commander', 0) == 0:
            st.markdown("<div class='urgence-card'>", unsafe_allow_html=True)
            st.markdown("### ⚠️ ATTENTION : Risque de panne élevé")
            st.markdown("**Recommandation manuelle :** Vérifier le stock pour anticiper les besoins de maintenance")
            st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
