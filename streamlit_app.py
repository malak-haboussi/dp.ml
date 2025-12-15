import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.api import SimpleExpSmoothing, Holt, ExponentialSmoothing
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy.stats import shapiro
from sklearn.metrics import mean_squared_error, mean_absolute_error
from numpy import sqrt
import logging
import io

# --- 1. FONCTIONS D'ÉVALUATION ---

def calculate_metrics(y_true, y_pred, model_resid=None):
    """Calcule les métriques de performance et réalise les tests de résidus."""
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = sqrt(mse)
    
    # Éviter la division par zéro dans le MAPE
    if np.any(y_true == 0):
        mape = np.nan
    else:
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    # Tests sur les résidus
    ljung_box_p = None
    shapiro_p = None
    if model_resid is not None and len(model_resid) > 1:
        # Test de Ljung-Box pour l'autocorrélation (Hypothèse nulle: pas d'autocorrélation)
        ljung_box_results = acorr_ljungbox(model_resid, lags=[10], return_df=True)
        ljung_box_p = ljung_box_results['lb_pvalue'].iloc[0]
        
        # Test de Shapiro-Wilk pour la normalité (Hypothèse nulle: distribution normale)
        if len(model_resid) >= 3 and len(model_resid) <= 5000: # Shapiro-Wilk a des limites de taille
            shapiro_test = shapiro(model_resid)
            shapiro_p = shapiro_test.pvalue

    return {
        'MSE': mse, 'MAE': mae, 'RMSE': rmse, 'MAPE': mape,
        'Ljung-Box P-value': ljung_box_p,
        'Shapiro-Wilk P-value': shapiro_p
    }

# --- 2. CONFIGURATION STREAMLIT ---

st.set_page_config(layout="wide", page_title="Application de Prévision de Séries Temporelles")

# --- 3. MISE EN PLACE DU LOGGING DANS LA SESSION ---

if 'log_content' not in st.session_state:
    st.session_state['log_content'] = ""

def append_to_log(message):
    """Ajoute un message au journal de la session Streamlit."""
    timestamp = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state['log_content'] += f"[{timestamp}] {message}\n"

# --- TITRE PRINCIPAL ---
st.title("📈 Application d'Analyse et de Prévision de Séries Temporelles")
append_to_log("--- Démarrage de la session ---")


# --- BARRE LATÉRALE (CONFIGURATION) ---
with st.sidebar:
    st.header("⚙️ Configuration des Données et du Modèle")
    
    # A. UPLOADER LE FICHIER
    uploaded_file = st.file_uploader("Importer votre jeu de données (CSV/Excel)", type=["csv", "xlsx"])
    
    if uploaded_file is not None:
        # Tenter de lire le fichier
        try:
            df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Erreur de lecture du fichier : {e}")
            df = None

        if df is not None:
            st.subheader("Sélection des Colonnes")
            
            # Sélection de la colonne de Date (si existe, sinon utiliser 'id' pour le démo)
            date_col = st.selectbox("Colonne Date/Index", options=df.columns)
            
            # Sélection de la colonne Cible
            target_col = st.selectbox("Colonne Valeur Cible", options=df.columns)

            st.subheader("Paramètres de Modélisation")
            
            # Période Saisonnière (P)
            period_saisonniere = st.number_input(
                "Période Saisonnière (P)", 
                value=12, 
                min_value=1, 
                help="Ex: 7 (Journalier/Hebdomadaire), 12 (Mensuel/Annuel)"
            )
            
            # Proportion Train/Test
            train_ratio = st.slider(
                "Proportion Entraînement (%)", 
                min_value=50, 
                max_value=90, 
                value=80
            ) / 100
            
            # Horizon de Prévision (H)
            horizon_prevision = st.number_input(
                "Horizon de Prévision (H)", 
                value=12, 
                min_value=1, 
                help="Nombre de périodes futures à prédire après le jeu de test."
            )
            
            st.button("Lancer l'Analyse et la Prévision", key="run_analysis")


# --- 4. TRAITEMENT ET ANALYSE (CORPS PRINCIPAL) ---

if uploaded_file is not None and df is not None:
    
    if st.session_state.get('run_analysis', False):
        
        # --- PHASE 1: PRÉTRAITEMENT ---
        
        try:
            # Conversion et Indexation
            append_to_log(f"Phase 1: Importation et Prétraitement de {target_col}")
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            df.dropna(subset=[date_col], inplace=True)
            df_ts = df.set_index(date_col)[target_col].astype(float)
            df_ts = df_ts.sort_index().asfreq(pd.infer_freq(df_ts.index))
            
            # Traitement des Manquantes
            initial_n = len(df_ts)
            missing_n = df_ts.isnull().sum()
            df_ts.interpolate(method='linear', inplace=True)
            append_to_log(f"  -> {initial_n} observations importées. {missing_n} valeurs manquantes interpolées.")
            st.success(f"Données chargées : {initial_n} observations de {df_ts.index.min()} à {df_ts.index.max()}")
            
        except Exception as e:
            st.error(f"Erreur lors du Prétraitement de la série : {e}. Vérifiez le format de vos colonnes.")
            append_to_log(f"  -> ERREUR DE PRÉTRAITEMENT: {e}")
            st.stop()
        
        # Découpage Train/Test
        train_size = int(len(df_ts) * train_ratio)
        train = df_ts.iloc[:train_size]
        test = df_ts.iloc[train_size:]
        append_to_log(f"  -> Découpage Train ({train_ratio*100:.0f}%) : {len(train)} obs. Test : {len(test)} obs.")


        # --- PHASE 2: ANALYSE EXPLORATOIRE ---
        
        st.header("1. 📊 Analyse Exploratoire")
        append_to_log("Phase 2: Analyse Exploratoire (EDA)")
        
        col1, col2 = st.columns(2)
        
        # A. Série Temporelle
        with col1:
            fig, ax = plt.subplots(figsize=(10, 4))
            df_ts.plot(ax=ax)
            ax.set_title(f"Série Temporelle : {target_col}")
            st.pyplot(fig)
            
        # B. Décomposition
        with col2:
            try:
                decomposition = seasonal_decompose(df_ts, model='additive', period=period_saisonniere, extrapolate_trend='freq')
                fig_decomp = decomposition.plot()
                fig_decomp.set_size_inches(10, 8)
                st.pyplot(fig_decomp)
            except Exception as e:
                st.warning(f"Impossible de décomposer (période saisonnière/données insuffisantes) : {e}")

        st.subheader("Tests de Stationnarité")
        try:
            adf_result = adfuller(df_ts.dropna())
            kpss_result = kpss(df_ts.dropna())
            st.write(f"Test ADF (p-value): **{adf_result[1]:.4f}** (Hypothèse nulle: non stationnaire)")
            st.write(f"Test KPSS (p-value): **{kpss_result[1]:.4f}** (Hypothèse nulle: stationnaire)")
            append_to_log(f"  -> ADF p-value: {adf_result[1]:.4f} | KPSS p-value: {kpss_result[1]:.4f}")
        except Exception as e:
            st.warning(f"Tests ADF/KPSS impossible : {e}")

        # --- PHASE 3: MODÉLISATION ET ÉVALUATION ---
        
        st.header("2. 🏗️ Modélisation et Évaluation")
        append_to_log("Phase 3: Modélisation et Optimisation")
        
        # Dictionnaire pour stocker les modèles ajustés et les résultats
        models = {}
        results_list = []
        
        # Définition des modèles à tester
        model_configs = {
            'SES': SimpleExpSmoothing(train, initialization_method="estimated"),
            'Holt': Holt(train, initialization_method="estimated"),
            'HW Additif': ExponentialSmoothing(train, seasonal_periods=period_saisonniere, trend='add', seasonal='add', initialization_method="estimated"),
            'HW Multiplicatif': ExponentialSmoothing(train, seasonal_periods=period_saisonniere, trend='add', seasonal='mul', initialization_method="estimated"),
        }
        
        progress_bar = st.progress(0)
        
        for i, (name, model_def) in enumerate(model_configs.items()):
            
            try:
                # Ajustement et Optimisation (Minimisation du MSE)
                fitted_model = model_def.fit(disp=False)
                models[name] = fitted_model
                
                # Récupération des paramètres optimaux
                params = fitted_model.params
                append_to_log(f"  -> Modèle {name} ajusté. Params: {fitted_model.params}")
                
                # Prévisions sur l'ensemble de TEST
                forecast_test = fitted_model.forecast(len(test))
                
                # Calcul des Métriques sur le TEST set
                metrics = calculate_metrics(test, forecast_test, fitted_model.resid)
                
                # Enregistrement des résultats
                result = {
                    'Modèle': name,
                    'AIC': fitted_model.aic,
                    'BIC': fitted_model.bic,
                    'Paramètres (α)': params.get('smoothing_level', np.nan),
                    'Paramètres (γ)': params.get('smoothing_seasonal', np.nan),
                    **metrics
                }
                results_list.append(result)
                
            except Exception as e:
                append_to_log(f"  -> ERREUR lors de l'ajustement du modèle {name}: {e}")
                st.warning(f"Modèle {name} non ajusté: {e}")
                
            progress_bar.progress((i + 1) / len(model_configs))

        progress_bar.empty()
        
        # Affichage du Tableau Comparatif
        if results_list:
            results_df = pd.DataFrame(results_list)
            results_df = results_df.sort_values(by='MSE')
            st.subheader("Tableau Comparatif et Sélection Automatique")
            st.dataframe(results_df.style.highlight_min(subset=['MSE', 'MAE', 'RMSE', 'AIC', 'BIC'], axis=0, color='lightgreen'))
            append_to_log("  -> Tableau comparatif généré et classé par MSE.")
            
            # Sélection du meilleur modèle
            best_model_name = results_df.iloc[0]['Modèle']
            best_model = models[best_model_name]
            st.success(f"🚀 **Meilleur Modèle Sélectionné (par MSE) : {best_model_name}**")
            append_to_log(f"  -> Modèle final sélectionné : {best_model_name}")

            # --- PHASE 4: PRÉVISIONS FINALES ---

            st.header("3. 📈 Prévisions Futures")
            append_to_log("Phase 4: Génération des Prévisions Futures")

            # Prédictions et Intervalles de Confiance
            forecast_result = best_model.predict(
                start=len(df_ts), 
                end=len(df_ts) + horizon_prevision - 1, 
                return_conf_int=True, 
                alpha=0.05 # Intervalle de confiance à 95%
            )
            forecast_values = pd.Series(forecast_result[0], name='Prévision')
            conf_int = pd.DataFrame(forecast_result[1], columns=['Borne_Inf', 'Borne_Sup'])
            
            # Création de l'index de prévision
            last_date = df_ts.index[-1]
            future_index = pd.date_range(start=last_date, periods=horizon_prevision + 1, freq=df_ts.index.freq)[1:]
            
            forecast_df = pd.concat([forecast_values, conf_int], axis=1).set_index(future_index)
            
            # A. Visualisation
            fig, ax = plt.subplots(figsize=(12, 6))
            df_ts.plot(ax=ax, label='Historique', color='blue')
            forecast_df['Prévision'].plot(ax=ax, label='Prévision Future', color='red', linestyle='--')
            
            # Afficher la zone de test (pour vérification)
            test.plot(ax=ax, label='Données de Test', color='orange')
            
            # Intervalle de confiance
            ax.fill_between(forecast_df.index, forecast_df['Borne_Inf'], forecast_df['Borne_Sup'], alpha=0.1, color='red', label='Intervalle de Confiance 95%')
            
            ax.legend()
            ax.set_title(f"Prévisions avec Intervalle de Confiance (Modèle: {best_model_name})")
            st.pyplot(fig)
            
            # B. Export
            st.subheader("Résultats Exportables")
            
            # Préparation du CSV/JSON
            csv_export = forecast_df.to_csv(index_label=date_col).encode('utf-8')
            json_export = forecast_df.to_json(orient='index').encode('utf-8')
            
            col_dl1, col_dl2, _ = st.columns([1, 1, 2])
            
            with col_dl1:
                st.download_button(
                    label="Télécharger les Prévisions (CSV)",
                    data=csv_export,
                    file_name='previsions_temporelles.csv',
                    mime='text/csv'
                )
            with col_dl2:
                 st.download_button(
                    label="Télécharger les Prévisions (JSON)",
                    data=json_export,
                    file_name='previsions_temporelles.json',
                    mime='application/json'
                )

        # --- PHASE 5: JOURNAL DE SORTIE (OUTPUT LOG) ---

        st.header("4. 📝 Journal de Sortie (Output Log)")
        st.code(st.session_state['log_content'], language='text')
        
        # Export du journal
        log_file = io.StringIO()
        log_file.write(st.session_state['log_content'])
        log_file_bytes = log_file.getvalue().encode('utf-8')
        
        st.download_button(
            label="Télécharger le Journal d'Audit (.txt)",
            data=log_file_bytes,
            file_name="journal_audit_prevision.txt",
            mime="text/plain"
        )


# --- INSTRUCTIONS D'EXÉCUTION ---
if uploaded_file is None:
    st.info("⬆️ Veuillez téléverser votre fichier de données via la barre latérale pour commencer.")
    st.caption("Assurez-vous qu'il contienne une colonne de date/heure et une colonne de valeur.")
