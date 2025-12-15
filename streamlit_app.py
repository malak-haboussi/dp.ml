import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.api import SimpleExpSmoothing, Holt, ExponentialSmoothing
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy.stats import shapiro
from sklearn.metrics import mean_squared_error, mean_absolute_error
from numpy import sqrt
import io

# --- 1. FONCTIONS D'ÉVALUATION ---

def calculate_metrics(y_true, y_pred, model_resid=None):
    """Calcule les métriques de performance et réalise les tests de résidus."""
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = sqrt(mse)
    
    # Calcul du MAPE, gère les zéros dans y_true
    mape_series = np.abs((y_true - y_pred) / y_true)
    mape = np.mean(mape_series[np.isfinite(mape_series)]) * 100
    
    # Tests sur les résidus
    ljung_box_p = None
    shapiro_p = None
    
    resid = model_resid.dropna() if model_resid is not None else pd.Series([])

    if len(resid) > 1:
        try:
            # Test de Ljung-Box pour l'autocorrélation
            ljung_box_results = acorr_ljungbox(resid, lags=[10], return_df=True)
            ljung_box_p = ljung_box_results['lb_pvalue'].iloc[0]
        except Exception:
             ljung_box_p = np.nan
        
        # Test de Shapiro-Wilk (limité en taille)
        if len(resid) >= 3 and len(resid) <= 5000: 
            try:
                shapiro_test = shapiro(resid)
                shapiro_p = shapiro_test.pvalue
            except Exception:
                shapiro_p = np.nan
        else:
            shapiro_p = f"N/A (Taille: {len(resid)})"

    return {
        'MSE': mse, 'MAE': mae, 'RMSE': rmse, 'MAPE': mape,
        'Ljung-Box P-value': ljung_box_p,
        'Shapiro-Wilk P-value': shapiro_p
    }

# --- 2. CONFIGURATION STREAMLIT & LOGGING ---

st.set_page_config(layout="wide", page_title="Application de Prévision de Séries Temporelles")

if 'log_content' not in st.session_state:
    st.session_state['log_content'] = ""

def append_to_log(message):
    """Ajoute un message au journal de la session Streamlit."""
    timestamp = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state['log_content'] += f"[{timestamp}] {message}\n"

# --- TITRE PRINCIPAL ---
st.title("📈 Application d'Analyse et de Prévision de Séries Temporelles")
append_to_log("--- Démarrage de la session ---")


# --- DÉCLARATIONS INITIALES ---
uploaded_file = None
df = None
date_col = None
target_col = None
period_saisonniere = 24
train_ratio = 0.8
horizon_prevision = 48
run_analysis = False


# --- BARRE LATÉRALE (CONFIGURATION) ---
with st.sidebar:
    st.header("⚙️ Configuration des Données et du Modèle")
    
    # Réinitialiser le log
    st.session_state['log_content'] = "" 
    
    uploaded_file = st.file_uploader("Importer votre jeu de données (CSV/Excel)", type=["csv", "xlsx"])
    
    if uploaded_file is not None:
        try:
            # Lire le fichier
            df = pd.read_csv(uploaded_file)
            append_to_log(f"Fichier chargé: {uploaded_file.name}")
            
            st.subheader("Sélection des Colonnes")
            
            # Aide à la sélection des colonnes
            default_date_idx = next((i for i, col in enumerate(df.columns) if 'date' in col.lower() or 'time' in col.lower()), 0)
            default_target_idx = next((i for i, col in enumerate(df.columns) if 'conso' in col.lower() or 'usage' in col.lower() or 'value' in col.lower()), 1 if len(df.columns) > 1 else 0)
            
            date_col = st.selectbox("Colonne Date/Index", options=df.columns, index=default_date_idx)
            target_col = st.selectbox("Colonne Valeur Cible (Consommation/Ventes)", options=df.columns, index=default_target_idx)
            
            if date_col == target_col:
                st.warning("La colonne date et la colonne cible ne peuvent être identiques.")
            else:
                st.subheader("Agrégation et Paramètres")
                
                # AGRÉGATION / RESAMPLING pour gérer les duplicata
                resample_freq = st.selectbox(
                    "Fréquence d'agrégation (Resample)",
                    options=['T', 'H', 'D', 'W', 'M'],
                    index=1, # 'H' pour horaire
                    help="T=Minute, H=Heure, D=Jour, W=Semaine, M=Mois. Choisir H si les données sont sub-horaires (votre cas)."
                )
                
                resample_method = st.selectbox(
                    "Méthode d'Agrégation",
                    options=['sum', 'mean'],
                    index=1, # Moyenne
                    help="Opération pour agréger les enregistrements multiples au même moment."
                )
                
                # Période Saisonnière (P)
                # La période doit correspondre à la fréquence agrégée
                if resample_freq == 'H':
                    default_period = 24 # Saisonnalité journalière
                elif resample_freq == 'D':
                    default_period = 7 # Saisonnalité hebdomadaire
                else:
                    default_period = 1
                
                period_saisonniere = st.number_input(
                    "Période Saisonnière (P)", 
                    value=default_period, 
                    min_value=1, 
                    help=f"Nombre de périodes dans un cycle. Ex: {default_period} si la fréquence est {resample_freq}."
                )
                
                # Proportion Train/Test
                train_ratio = st.slider("Proportion Entraînement (%)", min_value=50, max_value=90, value=80) / 100
                
                # Horizon de Prévision (H)
                horizon_prevision = st.number_input(
                    "Horizon de Prévision (H)", 
                    value=period_saisonniere * 2,
                    min_value=1
                )
                
                run_analysis = st.button("Lancer l'Analyse et la Prévision")

        except Exception as e:
            st.error(f"Erreur de lecture du fichier ou de sélection de colonnes : {e}")
            df = None


# --- 4. TRAITEMENT ET ANALYSE (CORPS PRINCIPAL) ---

if run_analysis and df is not None and date_col is not None and target_col is not None and date_col != target_col:
    
    with st.spinner("Analyse en cours... (Cela peut prendre du temps sur 90 000 enregistrements)"):
        
        # --- PHASE 1: PRÉTRAITEMENT ---
        
        try:
            # 1. Conversion et Indexation initiale
            append_to_log(f"Phase 1: Importation et Prétraitement de {target_col}")
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            df.dropna(subset=[date_col], inplace=True)
            
            df_ts_initial = df.set_index(date_col)[target_col].astype(float)
            df_ts_initial = df_ts_initial.sort_index()
            initial_n = len(df_ts_initial)
            
            # 2. AGRÉGATION / RESAMPLING pour gérer les duplicata
            append_to_log(f"  -> Agrégation à la fréquence {resample_freq} par {resample_method}...")
            
            if resample_method == 'sum':
                df_ts = df_ts_initial.resample(resample_freq).sum()
            else: # 'mean'
                df_ts = df_ts_initial.resample(resample_freq).mean()
            
            # 3. Traitement des Manquantes (interpolation linéaire)
            missing_n = df_ts.isnull().sum()
            df_ts.interpolate(method='linear', inplace=True)
            
            final_n = len(df_ts)
            
            append_to_log(f"  -> {initial_n} obs. initiales agrégées en {final_n} obs. {resample_freq}.")
            append_to_log(f"  -> {missing_n} périodes manquantes (vides) interpolées.")
            st.success(f"Données chargées et agrégées : {final_n} observations de {df_ts.index.min()} à {df_ts.index.max()} (Fréquence : {resample_freq})")
            
        except Exception as e:
            st.error(f"Erreur critique dans le Prétraitement : {e}. Le Resampling a échoué.")
            append_to_log(f"  -> ERREUR CRITIQUE DANS LE PRÉTRAITEMENT: {e}")
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
            df_ts.plot(ax=ax, label="Série Complète")
            ax.set_title(f"Série Temporelle Agrégée : {target_col}")
            st.pyplot(fig)
            
        # B. Décomposition Saisonière
        with col2:
            try:
                # Réduire les données pour la décomposition si elles sont trop nombreuses (> 2000)
                decomp_data = df_ts.tail(period_saisonniere * 2 if len(df_ts) > period_saisonniere * 2 else len(df_ts)).copy()
                decomposition = seasonal_decompose(decomp_data, model='additive', period=period_saisonniere, extrapolate_trend='freq')
                fig_decomp = decomposition.plot() 
                fig_decomp.set_size_inches(10, 8)
                st.pyplot(fig_decomp)
                st.caption(f"Décomposition affichée sur les {len(decomp_data)} dernières observations.")
            except Exception as e:
                st.warning(f"Impossible de décomposer (période/données insuffisantes) : {e}")
                append_to_log(f"  -> AVERTISSEMENT: Échec de la décomposition: {e}")

        # C. Tests de Stationnarité
        st.subheader("Tests de Stationnarité")
        try:
            adf_result = adfuller(df_ts.dropna())
            kpss_result = kpss(df_ts.dropna())
            st.write(f"Test ADF (p-value): **{adf_result[1]:.4f}** (H0: non stationnaire)")
            st.write(f"Test KPSS (p-value): **{kpss_result[1]:.4f}** (H0: stationnaire)")
            append_to_log(f"  -> ADF p-value: {adf_result[1]:.4f} | KPSS p-value: {kpss_result[1]:.4f}")
        except Exception as e:
            st.warning(f"Tests ADF/KPSS impossible : {e}")


        # --- PHASE 3: MODÉLISATION ET ÉVALUATION ---
        
        st.header("2. 🏗️ Modélisation et Évaluation")
        append_to_log("Phase 3: Modélisation et Optimisation")
        
        models = {}
        results_list = []
        
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
                
                params = fitted_model.params
                append_to_log(f"  -> Modèle {name} ajusté. Params: α={params.get('smoothing_level', np.nan):.4f}, γ={params.get('smoothing_seasonal', np.nan):.4f}")
                
                # Prévisions sur l'ensemble de TEST
                forecast_test = fitted_model.forecast(len(test))
                
                # Calcul des Métriques
                metrics = calculate_metrics(test, forecast_test, fitted_model.resid)
                
                # Enregistrement des résultats
                result = {
                    'Modèle': name,
                    'AIC': fitted_model.aic,
                    'BIC': fitted_model.bic,
                    'Paramètres (α)': f"{params.get('smoothing_level', np.nan):.4f}",
                    'Paramètres (γ)': f"{params.get('smoothing_seasonal', np.nan):.4f}",
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
            # Gestion des colonnes de p-value qui pourraient ne pas être numériques
            numeric_cols = ['MSE', 'MAE', 'RMSE', 'AIC', 'BIC']
            results_df = results_df.sort_values(by='MSE')
            st.subheader("Tableau Comparatif et Sélection Automatique")
            st.dataframe(results_df.style.highlight_min(subset=numeric_cols, axis=0, color='lightgreen'))
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
                alpha=0.05
            )
            forecast_values = pd.Series(forecast_result[0], name='Prévision')
            conf_int = pd.DataFrame(forecast_result[1], columns=['Borne_Inf', 'Borne_Sup'])
            
            # Création de l'index de prévision
            last_date = df_ts.index[-1]
            future_index = pd.date_range(start=last_date, periods=horizon_prevision + 1, freq=df_ts.index.freq)[1:]
            
            forecast_df = pd.concat([forecast_values, conf_int], axis=1).set_index(future_index)
            
            # A. Visualisation
            fig, ax = plt.subplots(figsize=(12, 6))
            df_ts.plot(ax=ax, label='Historique (Train)', color='blue')
            test.plot(ax=ax, label='Données de Test', color='orange')
            forecast_df['Prévision'].plot(ax=ax, label='Prévision Future', color='red', linestyle='--')
            
            ax.fill_between(forecast_df.index, forecast_df['Borne_Inf'], forecast_df['Borne_Sup'], alpha=0.1, color='red', label='Intervalle de Confiance 95%') 
            
            ax.legend()
            ax.set_title(f"Prévisions avec Intervalle de Confiance (Modèle: {best_model_name})")
            st.pyplot(fig)
            
            # B. Export
            st.subheader("Résultats Exportables")
            
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
    st.caption("Une fois le fichier chargé, sélectionnez la colonne de date/heure et la colonne cible, puis ajustez la Fréquence d'agrégation (H, D, etc.) pour gérer les doublons.")
