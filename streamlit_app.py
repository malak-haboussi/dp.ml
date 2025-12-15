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
from sklearn.linear_model import LinearRegression # Pour la Régression Linéaire
from numpy import sqrt
import io
import time # Pour mesurer le temps d'exécution
from datetime import datetime

# --- 1. FONCTIONS D'ÉVALUATION ---

def calculate_metrics(y_true, y_pred, model_resid=None):
    """Calcule les métriques de performance et réalise les tests de résidus."""
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = sqrt(mse)
    
    # Calcul du MAPE
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
session_id = str(hash(datetime.now()))[:8]
append_to_log(f"--- En-tête du processus ---")
append_to_log(f"Date et heure d'exécution: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
append_to_log(f"Identifiant unique de la session: {session_id}")


# --- DÉCLARATIONS INITIALES ---
uploaded_file = None
df = None
# ... (déclarations des variables restent inchangées) ...


# --- BARRE LATÉRALE (CONFIGURATION) ---
with st.sidebar:
    st.header("⚙️ Configuration des Données et du Modèle")
    
    st.session_state['log_content'] = "" # Réinitialiser le log à chaque rechargement
    
    uploaded_file = st.file_uploader("Importer votre jeu de données (CSV/Excel)", type=["csv", "xlsx"])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            append_to_log(f"Fichier chargé: {uploaded_file.name}")
            
            st.subheader("Sélection des Colonnes")
            
            # Aide à la sélection des colonnes (basée sur la recherche de mots-clés)
            default_date_idx = next((i for i, col in enumerate(df.columns) if 'date' in col.lower() or 'time' in col.lower()), 0)
            default_target_idx = next((i for i, col in enumerate(df.columns) if 'conso' in col.lower() or 'usage' in col.lower() or 'value' in col.lower()), 1 if len(df.columns) > 1 else 0)
            
            date_col = st.selectbox("Colonne Date/Index", options=df.columns, index=default_date_idx)
            target_col = st.selectbox("Colonne Valeur Cible", options=df.columns, index=default_target_idx)
            
            if date_col == target_col:
                st.warning("La colonne date et la colonne cible ne peuvent être identiques.")
            else:
                append_to_log(f"Description de la série analysée: Cible='{target_col}', Index='{date_col}'")
                
                st.subheader("Agrégation et Paramètres")
                
                # AGRÉGATION / RESAMPLING pour gérer les duplicata
                resample_freq = st.selectbox(
                    "Fréquence d'agrégation (Resample)",
                    options=['T', 'H', 'D', 'W', 'M'],
                    index=1, 
                    help="T=Minute, H=Heure, D=Jour, W=Semaine, M=Mois."
                )
                
                resample_method = st.selectbox(
                    "Méthode d'Agrégation",
                    options=['sum', 'mean'],
                    index=1,
                )
                
                # Détermination de la période saisonnière par défaut
                if resample_freq == 'H': default_period = 24
                elif resample_freq == 'D': default_period = 7
                else: default_period = 1
                
                period_saisonniere = st.number_input("Période Saisonnière (P)", value=default_period, min_value=1)
                train_ratio = st.slider("Proportion Entraînement (%)", min_value=50, max_value=90, value=80) / 100
                horizon_prevision = st.number_input("Horizon de Prévision (H)", value=period_saisonniere * 2, min_value=1)

                append_to_log(f"Paramètres de configuration: Fréquence={resample_freq}, Méthode agrégation={resample_method}, Période Saisonnière={period_saisonniere}, Train Ratio={train_ratio}")
                
                run_analysis = st.button("Lancer l'Analyse et la Prévision")

        except Exception as e:
            st.error(f"Erreur de lecture du fichier ou de sélection de colonnes : {e}")
            append_to_log(f"Erreur d'importation : {e}")
            df = None


# --- 4. TRAITEMENT ET ANALYSE (CORPS PRINCIPAL) ---

if run_analysis and df is not None and date_col is not None and target_col is not None and date_col != target_col:
    
    with st.spinner("Analyse en cours..."):
        
        # --- JOURNAL D'IMPORTATION ET PRÉTRAITEMENT ---
        
        append_to_log("--- Journal d'importation et prétraitement ---")
        try:
            # 1. Conversion et Indexation initiale
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            df.dropna(subset=[date_col], inplace=True)
            
            df_ts_initial = df.set_index(date_col)[target_col].astype(float)
            df_ts_initial = df_ts_initial.sort_index()
            initial_n = len(df_ts_initial)
            
            # 2. AGRÉGATION / RESAMPLING
            if resample_method == 'sum':
                df_ts = df_ts_initial.resample(resample_freq).sum()
            else:
                df_ts = df_ts_initial.resample(resample_freq).mean()
            
            # 3. Traitement des Manquantes
            missing_n = df_ts.isnull().sum()
            df_ts.interpolate(method='linear', inplace=True)
            final_n = len(df_ts)
            
            append_to_log(f"Nombre d'observations importées (brutes): {initial_n}")
            append_to_log(f"Variables disponibles et leur type: {target_col} (float)")
            append_to_log(f"Traitement des valeurs manquantes: {missing_n} périodes interpolées.")
            append_to_log(f"Transformations appliquées: Agrégation par {resample_method} à la fréquence {resample_freq}.")
            
            st.success(f"Données chargées et agrégées : {final_n} observations.")
            
        except Exception as e:
            st.error(f"Erreur lors du Prétraitement de la série : {e}. Le Resampling a échoué.")
            append_to_log(f"ERREUR CRITIQUE DANS LE PRÉTRAITEMENT: {e}")
            st.stop()
        
        # Découpage Train/Test
        train_size = int(len(df_ts) * train_ratio)
        train = df_ts.iloc[:train_size]
        test = df_ts.iloc[train_size:]


        # --- JOURNAL D'ANALYSE EXPLORATOIRE (EDA) ---
        
        st.header("1. 📊 Analyse Exploratoire")
        append_to_log("--- Journal d'analyse exploratoire ---")
        
        # Statistiques descriptives
        stats = df_ts.describe()
        skewness = df_ts.skew()
        kurtosis = df_ts.kurt()
        append_to_log(f"Statistiques descriptives: Moyenne={stats['mean']:.2f}, Variance={stats['std']**2:.2f}, Skewness={skewness:.2f}, Kurtosis={kurtosis:.2f}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots(figsize=(10, 4))
            df_ts.plot(ax=ax, label="Série Complète")
            ax.set_title(f"Série Temporelle Agrégée : {target_col}")
            st.pyplot(fig)
            
        # Détection de la saisonnalité et décomposition
        with col2:
            try:
                decomp_data = df_ts.tail(period_saisonniere * 2 if len(df_ts) > period_saisonniere * 2 else len(df_ts)).copy()
                decomposition = seasonal_decompose(decomp_data, model='additive', period=period_saisonniere, extrapolate_trend='freq')
                fig_decomp = decomposition.plot() 
                st.pyplot(fig_decomp)
                append_to_log(f"Détection de la saisonnalité: Période identifiée P={period_saisonniere}.")
                append_to_log(f"Analyse de la tendance: Tendance observée dans le graphique de décomposition.")
            except Exception as e:
                append_to_log(f"AVERTISSEMENT: Échec de la décomposition: {e}")

        # Tests de Stationnarité
        st.subheader("Tests de Stationnarité")
        try:
            adf_result = adfuller(df_ts.dropna())
            kpss_result = kpss(df_ts.dropna())
            st.write(f"Test ADF (p-value): **{adf_result[1]:.4f}** (H0: non stationnaire)")
            st.write(f"Test KPSS (p-value): **{kpss_result[1]:.4f}** (H0: stationnaire)")
            append_to_log(f"Résultats des tests de stationnarité: ADF p-value: {adf_result[1]:.4f} | KPSS p-value: {kpss_result[1]:.4f}")
        except Exception as e:
            append_to_log(f"ERREUR: Tests de stationnarité impossible: {e}")


        # --- JOURNAL DE MODÉLISATION ET ÉVALUATION ---
        
        st.header("2. 🏗️ Modélisation et Évaluation")
        append_to_log("--- Journal de modélisation ---")
        
        models = {}
        results_list = []
        
        # 1. Modèles de Lissage
        model_configs = {
            'SES': SimpleExpSmoothing(train, initialization_method="estimated"),
            'Holt': Holt(train, initialization_method="estimated"),
            'HW Additif': ExponentialSmoothing(train, seasonal_periods=period_saisonniere, trend='add', seasonal='add', initialization_method="estimated"),
            'HW Multiplicatif': ExponentialSmoothing(train, seasonal_periods=period_saisonniere, trend='add', seasonal='mul', initialization_method="estimated"),
        }
        
        # 2. Modèles Classiques (Moyenne Mobile et Régression Linéaire)
        
        # Moyenne Mobile (MM) - Naive Forecast using a simple rolling mean (k=period_saisonniere)
        k_mm = period_saisonniere
        forecast_mm = train.rolling(k_mm).mean().shift(1).iloc[-len(test):]
        # On ne peut pas calculer AIC/BIC pour la MM, mais on peut calculer les métriques de performance
        metrics_mm = calculate_metrics(test, forecast_mm, model_resid=test - forecast_mm)
        append_to_log(f"Modèles testés: Moyenne Mobile (k={k_mm}). Paramètres initiaux: k={k_mm}.")
        results_list.append({
            'Modèle': 'Moyenne Mobile', 'AIC': np.nan, 'BIC': np.nan, 'Paramètres (α)': np.nan, 'Paramètres (γ)': np.nan, **metrics_mm, 'Temps (s)': 0.0
        })

        # Régression Linéaire (RL) - Contre le temps
        start_time_lr = time.time()
        X_train = np.arange(len(train)).reshape(-1, 1)
        X_test = np.arange(len(train), len(df_ts)).reshape(-1, 1)
        model_lr = LinearRegression().fit(X_train, train.values)
        forecast_lr = pd.Series(model_lr.predict(X_test), index=test.index)
        time_lr = time.time() - start_time_lr
        
        # Calcul des résidus pour la RL
        resid_lr = test.values - forecast_lr.values
        metrics_lr = calculate_metrics(test, forecast_lr, model_resid=pd.Series(resid_lr))

        append_to_log(f"Modèles testés: Régression Linéaire (T). Temps d'exécution: {time_lr:.4f}s.")
        results_list.append({
            'Modèle': 'Régression Linéaire', 'AIC': np.nan, 'BIC': np.nan, 'Paramètres (α)': np.nan, 'Paramètres (γ)': np.nan, **metrics_lr, 'Temps (s)': time_lr
        })
        models['Régression Linéaire'] = model_lr # Stocker le modèle pour l'analyse des résidus
        
        # Modèles de Lissage (Optimisation par Minimisation du MSE)
        progress_bar = st.progress(0)
        
        for i, (name, model_def) in enumerate(model_configs.items()):
            start_time = time.time()
            try:
                fitted_model = model_def.fit(disp=False)
                models[name] = fitted_model
                
                time_fit = time.time() - start_time
                params = fitted_model.params
                append_to_log(f"Modèles testés: {name}. Temps d'exécution: {time_fit:.4f}s.")
                append_to_log(f"Paramètres optimaux retenus pour {name}: α={params.get('smoothing_level', np.nan):.4f}, γ={params.get('smoothing_seasonal', np.nan):.4f}")
                
                forecast_test = fitted_model.forecast(len(test))
                metrics = calculate_metrics(test, forecast_test, fitted_model.resid)
                
                results_list.append({
                    'Modèle': name, 'AIC': fitted_model.aic, 'BIC': fitted_model.bic,
                    'Paramètres (α)': f"{params.get('smoothing_level', np.nan):.4f}", 
                    'Paramètres (γ)': f"{params.get('smoothing_seasonal', np.nan):.4f}",
                    **metrics, 'Temps (s)': time_fit
                })
                
            except Exception as e:
                append_to_log(f"ERREUR/AVERTISSEMENT pour {name}: {e}")
                st.warning(f"Modèle {name} non ajusté: {e}")
                
            progress_bar.progress((i + 1) / len(model_configs))

        progress_bar.empty()
        
        # --- JOURNAL D'ÉVALUATION ---
        append_to_log("--- Journal d'évaluation ---")

        if results_list:
            results_df = pd.DataFrame(results_list)
            numeric_cols_eval = ['MSE', 'MAE', 'RMSE', 'AIC', 'BIC']
            results_df = results_df.sort_values(by='MSE')
            
            st.subheader("Tableau Comparatif des Performances")
            st.dataframe(results_df.style.highlight_min(subset=numeric_cols_eval, axis=0, color='lightgreen'))
            append_to_log("Tableau comparatif des performances (AIC, BIC, MSE, MAE, MAPE) généré.")
            append_to_log(f"Classement des modèles par MSE: 1er: {results_df.iloc[0]['Modèle']}, 2ème: {results_df.iloc[1]['Modèle']}")
            
            best_model_name = results_df.iloc[0]['Modèle']
            st.success(f"🚀 **Meilleur Modèle Sélectionné (par MSE) : {best_model_name}**")
            
            # --- Analyse des Résidus du Meilleur Modèle ---
            if best_model_name in models:
                best_model = models[best_model_name]
                
                if hasattr(best_model, 'resid'):
                     resid_best = best_model.resid
                elif best_model_name == 'Régression Linéaire':
                     resid_best = pd.Series(resid_lr, index=test.index)
                else: # Moyenne Mobile (MM) n'est pas stocké en objet, on utilise les résidus calculés précédemment
                     resid_best = test - forecast_mm
                
                st.subheader("Analyse des Résidus du Meilleur Modèle")
                
                fig_resid, ax_resid = plt.subplots(figsize=(10, 4))
                resid_best.plot(ax=ax_resid, title=f"Résidus de {best_model_name}")
                st.pyplot(fig_resid)
                append_to_log("Analyse graphique des résidus générée.")
                
                st.write(f"Tests Statistiques sur les Résidus (Ljung-Box p-value): {results_df[results_df['Modèle'] == best_model_name]['Ljung-Box P-value'].iloc[0]}")
                st.write(f"Tests Statistiques sur les Résidus (Shapiro-Wilk p-value): {results_df[results_df['Modèle'] == best_model_name]['Shapiro-Wilk P-value'].iloc[0]}")


            # --- PHASE 4: PRÉVISIONS FINALES ---

            st.header("3. 📈 Prévisions Futures et Export")
            append_to_log("--- Journal de prévision ---")

            # Prédictions et Intervalles de Confiance du Meilleur Modèle
            if best_model_name in models:
                if best_model_name == 'Régression Linéaire':
                    # Logique de prévision pour la Régression Linéaire
                    X_future = np.arange(len(df_ts), len(df_ts) + horizon_prevision).reshape(-1, 1)
                    forecast_values = pd.Series(best_model.predict(X_future))
                    conf_int = pd.DataFrame(index=forecast_values.index, data={'Borne_Inf': np.nan, 'Borne_Sup': np.nan}) # Pas d'IC facile
                
                elif best_model_name == 'Moyenne Mobile':
                     # Logique de prévision pour la Moyenne Mobile (prédiction plate)
                    last_k_mean = df_ts.tail(k_mm).mean()
                    forecast_values = pd.Series([last_k_mean] * horizon_prevision)
                    conf_int = pd.DataFrame(index=forecast_values.index, data={'Borne_Inf': np.nan, 'Borne_Sup': np.nan})

                else:
                    # Logique de prévision pour les modèles de Lissage (avec IC)
                    forecast_result = best_model.predict(start=len(df_ts), end=len(df_ts) + horizon_prevision - 1, return_conf_int=True, alpha=0.05)
                    forecast_values = pd.Series(forecast_result[0])
                    conf_int = pd.DataFrame(forecast_result[1], columns=['Borne_Inf', 'Borne_Sup'])
            
                # Création du DataFrame de prévision
                last_date = df_ts.index[-1]
                future_index = pd.date_range(start=last_date, periods=horizon_prevision + 1, freq=df_ts.index.freq)[1:]
                forecast_df = pd.concat([forecast_values, conf_int], axis=1).set_index(future_index)
                
                append_to_log("Prévisions ponctuelles et intervalles de confiance générés.")

                # A. Visualisation
                fig, ax = plt.subplots(figsize=(12, 6))
                df_ts.plot(ax=ax, label='Historique (Train)', color='blue')
                test.plot(ax=ax, label='Données de Test', color='orange')
                forecast_df['Prévision'].plot(ax=ax, label='Prévision Future', color='red', linestyle='--')
                
                if not forecast_df['Borne_Inf'].isnull().all():
                     ax.fill_between(forecast_df.index, forecast_df['Borne_Inf'], forecast_df['Borne_Sup'], alpha=0.1, color='red', label='Intervalle de Confiance 95%') 
                
                ax.legend()
                ax.set_title(f"Prévisions futures (Modèle: {best_model_name})")
                st.pyplot(fig)
                append_to_log("Visualisations générées.")

                # B. Export
                st.subheader("Fichiers Exportés")
                
                csv_export = forecast_df.to_csv(index_label=date_col).encode('utf-8')
                json_export = forecast_df.to_json(orient='index').encode('utf-8')
                
                col_dl1, col_dl2, col_dl3 = st.columns([1, 1, 1])
                
                with col_dl1:
                    st.download_button(
                        label="Télécharger les Prévisions (CSV)",
                        data=csv_export,
                        file_name='previsions_temporelles.csv',
                        mime='text/csv'
                    )
                    append_to_log("Fichiers exportés et leur emplacement: previsions_temporelles.csv")
                with col_dl2:
                     st.download_button(
                        label="Télécharger les Prévisions (JSON)",
                        data=json_export,
                        file_name='previsions_temporelles.json',
                        mime='application/json'
                    )
                     append_to_log("Fichiers exportés et leur emplacement: previsions_temporelles.json")
                
                # Le rapport PDF et l'archive nécessitent des libs externes (reportlab, zipfile) - Affichage d'un placeholder
                with col_dl3:
                    st.download_button(
                        label="Télécharger le Rapport d'Audit (.txt)",
                        data=io.StringIO(st.session_state['log_content']).getvalue().encode('utf-8'),
                        file_name="rapport_audit.txt",
                        mime="text/plain"
                    )
                    st.caption("Le Rapport PDF et l'Archive nécessitent des librairies non incluses.")
                

        # --- JOURNAL DE SORTIE FINAL ---

        st.header("4. 📝 Journal de Sortie (Output Log)")
        st.code(st.session_state['log_content'], language='text')

# --- INSTRUCTIONS D'EXÉCUTION ---
if uploaded_file is None:
    st.info("⬆️ Veuillez téléverser votre fichier de données via la barre latérale pour commencer. Assurez-vous d'ajuster les paramètres d'agrégation si l'erreur de 'duplicate labels' réapparaît.")
