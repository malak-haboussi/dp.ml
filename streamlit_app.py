import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# ... (Autres imports restent inchangés) ...

# --- [FONCTIONS D'ÉVALUATION] (Non modifiées) ---
# ... (La fonction calculate_metrics reste inchangée) ...

# --- CONFIGURATION STREAMLIT & LOGGING ---
# ... (Configuration de la page et du logging restent inchangés) ...

# --- TITRE PRINCIPAL ---
st.title("📈 Application d'Analyse et de Prévision de Séries Temporelles")
append_to_log("--- Démarrage de la session ---")


# --- BARRE LATÉRALE (CONFIGURATION) ---
with st.sidebar:
    st.header("⚙️ Configuration des Données et du Modèle")
    
    # NOUVEAU : Option de démo
    demo_option = st.checkbox("Utiliser les données démo (Consommation Électrique Sim.)", value=False)
    
    uploaded_file = st.file_uploader("Importer votre jeu de données (CSV/Excel)", type=["csv", "xlsx"])
    
    # FONCTION POUR CHARGER LES DONNÉES (DEMO OU UPLOAD)
    df = None
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Erreur de lecture du fichier : {e}")
            
    elif demo_option:
        # Création d'un jeu de données de démo avec Date_Heure
        date_range = pd.date_range(start='2023-01-01', periods=1000, freq='H')
        df = pd.DataFrame({
            'Date_Heure': date_range,
            'Consommation_Active_Globale': 100 + np.sin(np.arange(1000) / 20) * 10 + np.random.randn(1000) * 2 + (date_range.hour % 24) * 0.5
        })
        st.info("Jeu de données de démo chargé.")


    if df is not None:
        st.subheader("Sélection des Colonnes")
        
        # Laissez l'utilisateur choisir l'index et la cible
        date_col = st.selectbox("Colonne Date/Index", options=df.columns, 
                                index=df.columns.get_loc('Date_Heure') if 'Date_Heure' in df.columns else 0)
        
        target_col = st.selectbox("Colonne Valeur Cible", options=df.columns,
                                  index=df.columns.get_loc('Consommation_Active_Globale') if 'Consommation_Active_Globale' in df.columns else (1 if len(df.columns) > 1 else 0))

        st.subheader("Paramètres de Modélisation")
        
        # Période Saisonnière (P)
        period_saisonniere = st.number_input(
            "Période Saisonnière (P)", 
            value=24, # 24 heures pour une saisonnalité journalière si la fréquence est horaire
            min_value=1, 
            help="Période de répétition du cycle (Ex: 24 pour horaire journalier, 7 pour journalier hebdomadaire)."
        )
        
        # Proportion Train/Test (inchangée)
        train_ratio = st.slider("Proportion Entraînement (%)", min_value=50, max_value=90, value=80) / 100
        
        # Horizon de Prévision (H) (inchangé)
        horizon_prevision = st.number_input("Horizon de Prévision (H)", value=48, min_value=1)
        
        st.button("Lancer l'Analyse et la Prévision", key="run_analysis")


# --- [TRAITEMENT ET ANALYSE (CORPS PRINCIPAL)] ---
# ... (Reste du code (Étapes 2 à 5) reste inchangé, il utilisera df_ts) ...

# ... (Le code complet du bloc précédent reste valable, assurez-vous de le remplacer intégralement
#     avec les modifications ci-dessus dans la section st.sidebar) ...

# --- [INSTRUCTIONS D'EXÉCUTION] (Non modifiées) ---
if uploaded_file is None and not demo_option:
    st.info("⬆️ Veuillez téléverser votre fichier de données ou cocher l'option démo pour commencer.")
