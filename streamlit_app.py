import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ─── Configuration de la page ─────────────────────────────────────────────────
st.set_page_config(
    page_title="Sonatrach DAT — Prédiction de Rupture",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🛢️"
)

# ─── CSS GLOBAL ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Imports ── */
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500;600&display=swap');

/* ── Variables ── */
:root {
    --bg-primary:    #0D1117;
    --bg-card:       #161B22;
    --bg-card2:      #1C2330;
    --amber:         #F0A500;
    --amber-light:   #FFD166;
    --amber-muted:   #A06B00;
    --red:           #E63946;
    --red-muted:     #7D1E24;
    --orange:        #F4722B;
    --green:         #2DC653;
    --green-muted:   #155724;
    --text-primary:  #E8EAF0;
    --text-muted:    #7A8599;
    --border:        #2A3340;
}

/* ── Base ── */
html, body, [class*="css"]  {
    font-family: 'DM Sans', sans-serif;
    color: var(--text-primary);
}

/* ── Fond global ── */
.stApp {
    background: linear-gradient(135deg, #0D1117 0%, #111820 50%, #0D1117 100%);
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #0A0E14 !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] .block-container {
    padding-top: 2rem;
}

/* ── Supprime le padding par défaut du bloc principal ── */
.block-container {
    padding-top: 1.5rem !important;
    padding-bottom: 3rem !important;
    max-width: 1400px !important;
}

/* ── Bannière titre ── */
.hero-banner {
    background: linear-gradient(90deg, #0D1117 0%, #1A2030 40%, #1C2A1A 100%);
    border: 1px solid var(--border);
    border-left: 4px solid var(--amber);
    border-radius: 12px;
    padding: 1.5rem 2rem;
    margin-bottom: 1.5rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.6rem;
    font-weight: 800;
    color: var(--amber);
    letter-spacing: -0.02em;
    margin: 0;
}
.hero-sub {
    font-size: 0.85rem;
    color: var(--text-muted);
    margin: 0.2rem 0 0;
    font-weight: 400;
}
.hero-badge {
    background: rgba(240,165,0,0.12);
    border: 1px solid rgba(240,165,0,0.3);
    border-radius: 20px;
    padding: 0.4rem 1rem;
    font-size: 0.75rem;
    color: var(--amber);
    font-weight: 600;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}

/* ── KPI Cards ── */
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1rem;
    margin-bottom: 1.5rem;
}
.kpi-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s;
}
.kpi-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: var(--card-accent, var(--amber));
    border-radius: 12px 12px 0 0;
}
.kpi-card.danger { --card-accent: var(--red); }
.kpi-card.warning { --card-accent: var(--orange); }
.kpi-card.info { --card-accent: var(--amber); }

.kpi-icon { font-size: 1.6rem; margin-bottom: 0.5rem; display: block; }
.kpi-value {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 800;
    line-height: 1;
    color: var(--text-primary);
}
.kpi-label {
    font-size: 0.8rem;
    color: var(--text-muted);
    margin-top: 0.3rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}
.kpi-delta {
    font-size: 0.75rem;
    margin-top: 0.4rem;
    font-weight: 500;
}
.kpi-delta.bad { color: var(--red); }
.kpi-delta.warn { color: var(--orange); }
.kpi-delta.good { color: var(--green); }

/* ── Section headers ── */
.section-header {
    font-family: 'Syne', sans-serif;
    font-size: 1rem;
    font-weight: 700;
    color: var(--text-primary);
    letter-spacing: 0.02em;
    text-transform: uppercase;
    margin: 1.5rem 0 0.75rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.section-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    background: var(--amber);
    display: inline-block;
    flex-shrink: 0;
}

/* ── Tableau ── */
.dataframe-container {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    overflow: hidden;
}

/* ── Statut pills ── */
.pill {
    display: inline-block;
    border-radius: 20px;
    padding: 0.2rem 0.65rem;
    font-size: 0.75rem;
    font-weight: 600;
}
.pill-red    { background: rgba(230,57,70,0.15); color: #FF6B75; border: 1px solid rgba(230,57,70,0.3); }
.pill-orange { background: rgba(244,114,43,0.15); color: #F4722B; border: 1px solid rgba(244,114,43,0.3); }
.pill-amber  { background: rgba(240,165,0,0.15);  color: #F0A500; border: 1px solid rgba(240,165,0,0.3); }
.pill-green  { background: rgba(45,198,83,0.15);  color: #2DC653; border: 1px solid rgba(45,198,83,0.3); }

/* ── Sidebar widgets ── */
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stMultiSelect label {
    color: var(--text-muted) !important;
    font-size: 0.8rem !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
[data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
    color: var(--amber) !important;
    font-family: 'Syne', sans-serif !important;
}

/* ── Streamlit metric overrides ── */
[data-testid="metric-container"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    padding: 1rem 1.25rem !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-family: 'Syne', sans-serif !important;
    font-size: 2rem !important;
    font-weight: 800 !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg-primary); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--amber-muted); }

/* ── Dataframe ── */
.stDataFrame {
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    overflow: hidden !important;
}

/* ── Barre de progression ── */
.progress-bar-outer {
    background: var(--border);
    border-radius: 4px;
    height: 6px;
    width: 100%;
    overflow: hidden;
}
.progress-bar-inner {
    height: 100%;
    border-radius: 4px;
}
</style>
""", unsafe_allow_html=True)


# ─── Données LSTM réelles ─────────────────────────────────────────────────────
@st.cache_data
def load_data():
    # Données issues du modèle LSTM — valeurs exactes
    raw = [
        ('538Y042219', 11707,  1043.93, 'RUPTURE J+12'),
        ('586L015592',  1007,   964.89, 'OK'),
        ('584C110991',   376,  1031.48, 'OK'),
        ('538Y041201',  1359,  1091.11, 'OK'),
        ('584W011710',     0,  1078.42, 'RUPTURE STOCK'),
        ('584C110457',  1339,   909.38, 'RUPTURE J+10'),
        ('584C113270',     0,   960.46, 'RUPTURE STOCK'),
        ('584W010711',     0,   992.09, 'RUPTURE STOCK'),
        ('538Y042606',  1154,  1016.21, 'RUPTURE J+11'),
        ('538Y030905',   526,  1078.73, 'RUPTURE J+4'),
        ('584C030965',     0,  1022.14, 'RUPTURE STOCK'),
        ('538Y042632',   628,  1062.98, 'RUPTURE J+11'),
        ('536Y200600',     0,   910.04, 'RUPTURE STOCK'),
        ('588W662595', 23405,  1018.07, 'RUPTURE J+6'),
        ('584C030232',   108,   943.49, 'OK'),
        ('538Y042626',    43,  1083.86, 'RUPTURE J+16'),
        ('584J250350',  4834,  1062.52, 'OK'),
        ('586M038493',   225,  1031.73, 'OK'),
        ('584C110023',   457,   906.18, 'OK'),
        ('586L015590',     0,   991.98, 'RUPTURE STOCK'),
        ('584J250270',  8802,   908.66, 'OK'),
        ('588W662525',     0,  1004.16, 'RUPTURE STOCK'),
        ('584W010713',   430,  1021.67, 'OK'),
    ]

    lignes = []
    for code, stock, besoin, alerte in raw:
        # Extraire les jours restants depuis l'alerte LSTM
        if alerte == 'RUPTURE STOCK':
            jours_restants = 0
        elif alerte.startswith('RUPTURE J+'):
            jours_restants = int(alerte.replace('RUPTURE J+', ''))
        else:  # OK
            # Estimation : stock / (besoin/30)
            conso_jour = besoin / 30
            jours_restants = int(stock / conso_jour) if conso_jour > 0 else 999
            jours_restants = min(jours_restants, 999)

        couverture_pct = min(100, round(stock / besoin * 100, 1)) if besoin > 0 else 0

        # Statut unifié basé sur l'alerte LSTM
        if alerte == 'RUPTURE STOCK':
            statut  = 'RUPTURE'
            priorite = 0
        elif alerte.startswith('RUPTURE J+'):
            j = jours_restants
            if j <= 7:
                statut   = 'CRITIQUE'
                priorite = 1
            elif j <= 15:
                statut   = 'ATTENTION'
                priorite = 2
            else:
                statut   = 'ATTENTION'
                priorite = 2
        else:
            statut   = 'OK'
            priorite = 3

        lignes.append({
            'Code Article':      code,
            'Stock Actuel':      stock,
            'Besoin Prévu (30j)': besoin,
            'Jours Restants':    jours_restants,
            'Couverture (%)':    couverture_pct,
            'Alerte LSTM':       alerte,
            'Statut':            statut,
            'Priorité':          priorite
        })

    return pd.DataFrame(lignes).sort_values(['Priorité', 'Jours Restants']).reset_index(drop=True)

df = load_data()

# ─── Session state — articles ajoutés dynamiquement ──────────────────────────
if 'articles_extra' not in st.session_state:
    st.session_state.articles_extra = []

# Fusion des articles ajoutés avec les données de base
if st.session_state.articles_extra:
    df_extra = pd.DataFrame(st.session_state.articles_extra)
    df = pd.concat([df, df_extra], ignore_index=True)

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 0.5rem 0 1.5rem;">
        <div style="font-size:2.5rem;">🛢️</div>
        <div style="font-family:'Syne',sans-serif; font-size:1rem; font-weight:800;
                    color:#F0A500; letter-spacing:0.05em;">SONATRACH</div>
        <div style="font-size:0.7rem; color:#7A8599; text-transform:uppercase;
                    letter-spacing:0.08em; margin-top:0.15rem;">Système Prédictif</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 🔎 Filtres")

    filtre_statut = st.multiselect(
        "Statut",
        options=["RUPTURE", "CRITIQUE", "ATTENTION", "OK"],
        default=["RUPTURE", "CRITIQUE", "ATTENTION", "OK"]
    )

    st.markdown("---")
    st.markdown("### 📅 Horizon d'analyse")

    seuil_jours = st.slider(
        "Horizon (jours)",
        min_value=7, max_value=90, value=30, step=1,
        help="Modifie la fenêtre temporelle du graphique et recalcule les statuts"
    )

    # Seuils dérivés automatiquement de l'horizon
    seuil_critique  = max(3,  round(seuil_jours * 0.20))   # 20% de l'horizon
    seuil_attention = seuil_jours                            # = l'horizon complet

    st.markdown(f"""
    <div style="background:#1C2330; border:1px solid #2A3340; border-radius:8px;
                padding:0.75rem 1rem; margin-top:0.5rem; font-size:0.78rem; line-height:2;">
        <div>🔴 <strong style="color:#E63946;">Rupture stock</strong> — stock = 0</div>
        <div>🟠 <strong style="color:#F4722B;">Critique</strong> — ≤ {seuil_critique} j</div>
        <div>🟡 <strong style="color:#F0A500;">Attention</strong> — ≤ {seuil_attention} j</div>
        <div>🟢 <strong style="color:#2DC653;">OK</strong> — > {seuil_attention} j</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### ➕ Nouvel Article")

    with st.expander("📂 Importer un historique", expanded=False):
        st.markdown("""
        <div style="font-size:0.78rem; color:#7A8599; line-height:1.7; margin-bottom:0.8rem;">
            Téléchargez l'historique de consommation (5 ans) d'un article.<br>
            Format accepté : <strong style="color:#E8EAF0;">CSV ou Excel (.xlsx)</strong><br>
            Colonnes requises : <code style="color:#F0A500;">date</code> · <code style="color:#F0A500;">consommation</code>
        </div>
        """, unsafe_allow_html=True)

        new_code = st.text_input(
            "Code Article",
            placeholder="ex: 584C999888",
            max_chars=20,
            key="upload_code"
        ).strip().upper()

        new_stock_upload = st.number_input(
            "Stock Actuel (unités)",
            min_value=0, step=1, value=0,
            key="upload_stock"
        )

        uploaded_file = st.file_uploader(
            "Historique consommation",
            type=["csv", "xlsx"],
            help="Fichier avec colonnes 'date' et 'consommation'"
        )

        if uploaded_file is not None:
            try:
                # Lecture du fichier
                if uploaded_file.name.endswith('.xlsx'):
                    df_hist = pd.read_excel(uploaded_file)
                else:
                    df_hist = pd.read_csv(uploaded_file, sep=None, engine='python')

                # Normaliser les noms de colonnes
                df_hist.columns = [c.strip().lower() for c in df_hist.columns]

                # Chercher colonnes date et consommation
                col_date = next((c for c in df_hist.columns if 'date' in c), None)
                col_conso = next((c for c in df_hist.columns
                                  if any(k in c for k in ['conso', 'quantit', 'qty', 'mouvement', 'sortie'])), None)

                if col_date is None or col_conso is None:
                    st.error(f"❌ Colonnes introuvables. Colonnes détectées : {list(df_hist.columns)}")
                else:
                    df_hist[col_date]  = pd.to_datetime(df_hist[col_date], errors='coerce')
                    df_hist[col_conso] = pd.to_numeric(df_hist[col_conso], errors='coerce')
                    df_hist = df_hist.dropna(subset=[col_date, col_conso])
                    df_hist = df_hist.sort_values(col_date)

                    n_lignes = len(df_hist)
                    date_min = df_hist[col_date].min().strftime('%d/%m/%Y')
                    date_max = df_hist[col_date].max().strftime('%d/%m/%Y')
                    conso_totale = df_hist[col_conso].sum()

                    # ── Aperçu ──
                    st.markdown(f"""
                    <div style="background:#0F1A0F; border:1px solid #1A3A1A; border-radius:8px;
                                padding:0.75rem 1rem; font-size:0.78rem; line-height:2; margin:0.5rem 0;">
                        <div>📅 <strong style="color:#E8EAF0;">Période :</strong>
                             <span style="color:#2DC653;">{date_min} → {date_max}</span></div>
                        <div>📋 <strong style="color:#E8EAF0;">Lignes :</strong>
                             <span style="color:#2DC653;">{n_lignes:,}</span></div>
                        <div>📦 <strong style="color:#E8EAF0;">Conso totale :</strong>
                             <span style="color:#2DC653;">{conso_totale:,.0f} unités</span></div>
                    </div>
                    """, unsafe_allow_html=True)

                    # ── Simulation prédiction LSTM ──────────────────────────────
                    # Calcul de la consommation mensuelle sur les 6 derniers mois
                    derniers_6m = df_hist[
                        df_hist[col_date] >= df_hist[col_date].max() - pd.DateOffset(months=6)
                    ]
                    conso_jour_recent = (
                        derniers_6m[col_conso].sum() / max(1, (derniers_6m[col_date].max()
                        - derniers_6m[col_date].min()).days)
                    ) if len(derniers_6m) > 1 else conso_totale / max(1, n_lignes)

                    # Saisonnalité : ratio dernier mois vs moyenne
                    dernier_mois = df_hist[
                        df_hist[col_date] >= df_hist[col_date].max() - pd.DateOffset(months=1)
                    ]
                    conso_dernier_mois = dernier_mois[col_conso].sum()
                    conso_moy_mensuelle = conso_totale / max(1, n_lignes / 30)
                    facteur_saison = min(2.0, max(0.5, conso_dernier_mois / conso_moy_mensuelle)) \
                                     if conso_moy_mensuelle > 0 else 1.0

                    # Besoin prévu sur 30j (avec saisonnalité)
                    besoin_30j_pred = round(conso_jour_recent * 30 * facteur_saison, 2)
                    besoin_30j_pred = max(besoin_30j_pred, 1.0)

                    # Jours restants estimés
                    jr_pred = int(new_stock_upload / conso_jour_recent) \
                              if conso_jour_recent > 0 else 999

                    # Alerte
                    if new_stock_upload == 0:
                        alerte_pred = 'RUPTURE STOCK'
                    elif jr_pred <= 30:
                        alerte_pred = f'RUPTURE J+{jr_pred}'
                    else:
                        alerte_pred = 'OK'

                    couv_pred = min(100, round(new_stock_upload / besoin_30j_pred * 100, 1)) \
                                if besoin_30j_pred > 0 else 0

                    # Couleur alerte
                    if alerte_pred == 'RUPTURE STOCK':
                        clr_a = '#E63946'; icon_a = '🔴'
                    elif alerte_pred.startswith('RUPTURE'):
                        clr_a = '#F4722B' if jr_pred <= seuil_critique else '#F0A500'
                        icon_a = '🟠'
                    else:
                        clr_a = '#2DC653'; icon_a = '🟢'

                    st.markdown(f"""
                    <div style="background:#1A1108; border:1px solid #4A3010; border-radius:8px;
                                padding:0.85rem 1rem; font-size:0.78rem; line-height:2.1; margin:0.5rem 0;">
                        <div style="font-family:'Syne',sans-serif; font-size:0.85rem;
                                    font-weight:700; color:#F0A500; margin-bottom:0.3rem;">
                            ⚡ Prédiction LSTM
                        </div>
                        <div>📦 <strong style="color:#E8EAF0;">Besoin prévu (30j) :</strong>
                             <span style="color:#FFD166;">{besoin_30j_pred:,.2f}</span></div>
                        <div>📅 <strong style="color:#E8EAF0;">Jours restants :</strong>
                             <span style="color:#FFD166;">J+{jr_pred}</span></div>
                        <div>📊 <strong style="color:#E8EAF0;">Couverture :</strong>
                             <span style="color:#FFD166;">{couv_pred}%</span></div>
                        <div>🔔 <strong style="color:#E8EAF0;">Alerte système :</strong>
                             <span style="color:{clr_a}; font-weight:700;">{icon_a} {alerte_pred}</span></div>
                    </div>
                    """, unsafe_allow_html=True)

                    # ── Bouton Ajouter ───────────────────────────────────────
                    if st.button("✅ Ajouter au Dashboard", use_container_width=True,
                                 key="btn_add_upload"):
                        if not new_code:
                            st.error("❌ Entrez un code article.")
                        else:
                            codes_existants = [r['Code Article']
                                               for r in st.session_state.articles_extra]
                            if new_code in df['Code Article'].values \
                                    or new_code in codes_existants:
                                st.warning(f"⚠️ L'article **{new_code}** existe déjà.")
                            else:
                                new_row = {
                                    'Code Article':       new_code,
                                    'Stock Actuel':       int(new_stock_upload),
                                    'Besoin Prévu (30j)': float(besoin_30j_pred),
                                    'Jours Restants':     jr_pred,
                                    'Couverture (%)':     couv_pred,
                                    'Alerte LSTM':        alerte_pred,
                                    'Statut':             'OK',
                                    'Priorité':           3
                                }
                                st.session_state.articles_extra.append(new_row)
                                st.success(f"✅ Article **{new_code}** ajouté au dashboard !")
                                st.rerun()

            except Exception as e:
                st.error(f"❌ Erreur de lecture : {e}")

        else:
            # Template téléchargeable
            template_csv = "date,consommation\n2020-01-01,45\n2020-01-02,38\n2020-01-03,52\n"
            st.download_button(
                label="📥 Télécharger un modèle CSV",
                data=template_csv,
                file_name="historique_template.csv",
                mime="text/csv",
                use_container_width=True
            )

    # ── Articles ajoutés ──────────────────────────────────────────────────────
    if st.session_state.articles_extra:
        st.markdown(f"""
        <div style="font-size:0.78rem; color:#F0A500; font-weight:600; margin:0.6rem 0 0.3rem;">
            📦 {len(st.session_state.articles_extra)} article(s) importé(s)
        </div>""", unsafe_allow_html=True)

        for i, art in enumerate(st.session_state.articles_extra):
            c_a, c_b = st.columns([4, 1])
            c_a.markdown(
                f"<span style='font-size:0.75rem; font-family:monospace; "
                f"color:#E8EAF0;'>{art['Code Article']}</span>",
                unsafe_allow_html=True
            )
            if c_b.button("🗑️", key=f"del_{i}", help="Supprimer"):
                st.session_state.articles_extra.pop(i)
                st.rerun()

        if st.button("🗑️ Tout supprimer", use_container_width=True):
            st.session_state.articles_extra = []
            st.rerun()

    st.markdown("---")
    st.markdown("""
    <div style="font-size:0.72rem; color:#7A8599; line-height:1.6;">
        <strong style="color:#F0A500;">Direction Approvisionnement<br>et Transport (DAT)</strong><br><br>
        Modèle LSTM · Données réelles
    </div>
    """, unsafe_allow_html=True)

# ─── Recalcul dynamique des statuts selon l'horizon choisi ──────────────────
def recalc_statut(row, seuil_crit, seuil_att):
    """Recalcule le statut en fonction des seuils dynamiques."""
    j = row['Jours Restants']
    if row['Stock Actuel'] == 0:
        return 'RUPTURE'
    elif j <= seuil_crit:
        return 'CRITIQUE'
    elif j <= seuil_att:
        return 'ATTENTION'
    else:
        return 'OK'

df_dyn = df.copy()
df_dyn['Statut'] = df_dyn.apply(
    lambda r: recalc_statut(r, seuil_critique, seuil_attention), axis=1
)
df_dyn['Priorité'] = df_dyn['Statut'].map(
    {'RUPTURE': 0, 'CRITIQUE': 1, 'ATTENTION': 2, 'OK': 3}
)
df_dyn = df_dyn.sort_values(['Priorité', 'Jours Restants']).reset_index(drop=True)

df_f = df_dyn[df_dyn['Statut'].isin(filtre_statut)]

# ─── Bannière ────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="hero-banner">
    <div>
        <p class="hero-title">🛢️ Tableau de Bord Prédictif — Gestion de Stock</p>
        <p class="hero-sub">Direction Approvisionnement et Transport · Modèle LSTM · Horizon <strong style="color:#F0A500;">{seuil_jours} jours</strong></p>
    </div>
    <div class="hero-badge">⚡ Horizon J+{seuil_jours}</div>
</div>
""", unsafe_allow_html=True)

# ─── KPIs ─────────────────────────────────────────────────────────────────────
n_rupture  = len(df_dyn[df_dyn['Statut'] == 'RUPTURE'])
n_critique = len(df_dyn[df_dyn['Statut'] == 'CRITIQUE'])
n_attention= len(df_dyn[df_dyn['Statut'] == 'ATTENTION'])
n_ok       = len(df_dyn[df_dyn['Statut'] == 'OK'])
stock_total= df_dyn['Stock Actuel'].sum()
taux_dispo = round(n_ok / len(df_dyn) * 100, 1)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="kpi-card danger">
        <span class="kpi-icon">💥</span>
        <div class="kpi-value">{n_rupture}</div>
        <div class="kpi-label">Articles en Rupture</div>
        <div class="kpi-delta bad">▲ Réapprovisionnement urgent</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="kpi-card warning">
        <span class="kpi-icon">⚠️</span>
        <div class="kpi-value">{n_critique}</div>
        <div class="kpi-label">Alertes Critiques ≤ {seuil_critique}j</div>
        <div class="kpi-delta warn">▲ Action immédiate requise</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="kpi-card info">
        <span class="kpi-icon">🔶</span>
        <div class="kpi-value">{n_attention}</div>
        <div class="kpi-label">Surveillance ≤ {seuil_attention}j</div>
        <div class="kpi-delta warn">Horizon sélectionné</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="kpi-card info">
        <span class="kpi-icon">📦</span>
        <div class="kpi-value">{stock_total:,.0f}</div>
        <div class="kpi-label">Stock Total (Classe A)</div>
        <div class="kpi-delta good">✓ Taux disponibilité : {taux_dispo}%</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='margin-top:1rem'></div>", unsafe_allow_html=True)

# ─── Graphiques ──────────────────────────────────────────────────────────────
col_left, col_right = st.columns([3, 2], gap="medium")

# ── Heatmap ──────────────────────────────────────────────────────────────────
with col_left:
    st.markdown("""
    <div class="section-header">
        <span class="section-dot"></span> Cartographie de Disponibilité — Horizon 30 Jours
    </div>
    """, unsafe_allow_html=True)

    # ── Graphique Gantt de disponibilité ──────────────────────────────────────
    # Trier par jours restants (critiques en haut)
    df_gantt = df_f.sort_values('Jours Restants', ascending=True).reset_index(drop=True)
    n_items = len(df_gantt)
    HORIZON = seuil_jours   # ← horizon dynamique depuis le slider
    BAR_H   = 0.62
    PAD     = 0.19

    fig, ax = plt.subplots(figsize=(11, max(5, n_items * 0.52 + 1.2)))
    fig.patch.set_facecolor('#161B22')
    ax.set_facecolor('#161B22')

    # Couleurs par statut
    color_map = {
        'RUPTURE':  ('#E63946', '#FF6B75', '#3D0B0E'),   # vif, label, fond rupture
        'CRITIQUE': ('#F4722B', '#FF9A6C', '#3D1B0B'),
        'ATTENTION':('#F0A500', '#FFD166', '#3D2C00'),
        'OK':       ('#2DC653', '#5EE87F', '#0A2E14'),
    }

    for i, row in df_gantt.iterrows():
        j     = min(int(row['Jours Restants']), HORIZON)
        stat  = row['Statut']
        code  = row['Code Article']
        clr_vif, clr_lbl, clr_bg = color_map[stat]

        y = i * (BAR_H + PAD)

        # Fond grisé total (zone 30j)
        ax.barh(y, HORIZON, height=BAR_H, left=0,
                color='#1E2530', edgecolor='none', zorder=1)

        # Barre "stock disponible"
        if j > 0:
            ax.barh(y, j, height=BAR_H, left=0,
                    color=clr_vif, edgecolor='none',
                    alpha=0.88, zorder=2)

        # Barre "rupture prévue" (zone après j)
        if j < HORIZON:
            ax.barh(y, HORIZON - j, height=BAR_H, left=j,
                    color='#E63946', edgecolor='none',
                    alpha=0.18, zorder=2)

        # Ligne de rupture verticale
        if 0 < j < HORIZON:
            ax.axvline(x=j, ymin=(y - BAR_H/2) / (n_items * (BAR_H + PAD)),
                       ymax=(y + BAR_H/2) / (n_items * (BAR_H + PAD)),
                       color=clr_vif, linewidth=1.2, alpha=0.7, zorder=3)

        # Label jours restants dans la barre
        label_x = j / 2 if j >= 3 else j + 0.5
        label_txt = f"J+{j}" if j > 0 else "Rupture"
        label_col = '#0D1117' if j >= 4 else clr_vif
        ax.text(label_x, y, label_txt,
                ha='center', va='center',
                fontsize=7.5, fontweight='bold',
                color=label_col, zorder=4,
                fontfamily='monospace')

        # Label code article (à gauche)
        ax.text(-0.4, y, code,
                ha='right', va='center',
                fontsize=8.2, color=clr_lbl,
                fontfamily='monospace', zorder=4)

    # Ligne verticale "Aujourd'hui"
    ax.axvline(x=0, color='#F0A500', linewidth=1.5, linestyle='--', alpha=0.5, zorder=5)

    # Grille verticale aux jalons (dynamiques selon horizon)
    jalons = sorted(set([
        seuil_critique,
        seuil_attention // 2 if seuil_attention > 14 else seuil_attention,
        seuil_attention
    ]))
    for x_mark in jalons:
        if x_mark <= HORIZON:
            ax.axvline(x=x_mark, color='#2A3340', linewidth=0.8, linestyle='-', zorder=0)
            ax.text(x_mark, -0.5, f"J{x_mark}",
                    ha='center', va='top', fontsize=7.5, color='#7A8599')

    # Axes
    ax.set_xlim(-0.5, HORIZON + 0.2)
    ax.set_ylim(-0.8, n_items * (BAR_H + PAD))
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_xlabel("")

    for spine in ax.spines.values():
        spine.set_visible(False)

    # Légende
    leg_items = [
        mpatches.Patch(color='#2DC653', alpha=0.88, label=f'OK  — > J+{seuil_attention}'),
        mpatches.Patch(color='#F0A500', alpha=0.88, label=f'Attention  — ≤ J+{seuil_attention}'),
        mpatches.Patch(color='#F4722B', alpha=0.88, label=f'Critique  — ≤ J+{seuil_critique}'),
        mpatches.Patch(color='#E63946', alpha=0.88, label='Rupture  — stock nul'),
    ]
    ax.legend(handles=leg_items, loc='lower right', fontsize=8,
              facecolor='#1C2330', edgecolor='#2A3340',
              labelcolor='#E8EAF0', framealpha=0.95,
              handlelength=1.2, handleheight=0.9)

    plt.tight_layout(pad=0.4)
    st.pyplot(fig, use_container_width=True)
    plt.close()

# ── Graphiques droite ────────────────────────────────────────────────────────
with col_right:
    # Répartition des statuts (donut)
    st.markdown("""
    <div class="section-header">
        <span class="section-dot"></span> Répartition des Statuts
    </div>
    """, unsafe_allow_html=True)

    statut_counts = df['Statut'].value_counts()
    labels_map = {'RUPTURE': 'Rupture', 'CRITIQUE': 'Critique', 'ATTENTION': 'Attention', 'OK': 'OK'}
    colors_map  = {'RUPTURE': '#E63946', 'CRITIQUE': '#F4722B', 'ATTENTION': '#F0A500', 'OK': '#2DC653'}

    statuts_order = ['RUPTURE', 'CRITIQUE', 'ATTENTION', 'OK']
    vals   = [statut_counts.get(s, 0) for s in statuts_order]
    clrs   = [colors_map[s] for s in statuts_order]
    lbls   = [labels_map[s] for s in statuts_order]

    fig2, ax2 = plt.subplots(figsize=(5, 4))
    fig2.patch.set_facecolor('#161B22')
    ax2.set_facecolor('#161B22')

    wedges, texts, autotexts = ax2.pie(
        vals, labels=None, colors=clrs,
        autopct='%1.0f%%', startangle=90,
        wedgeprops=dict(width=0.55, edgecolor='#161B22', linewidth=2),
        pctdistance=0.78
    )
    for at in autotexts:
        at.set_color('#E8EAF0'); at.set_fontsize(9); at.set_fontweight('bold')

    ax2.legend(
        wedges, [f"{l} ({v})" for l, v in zip(lbls, vals)],
        loc="lower center", fontsize=8, ncol=2,
        facecolor='#1C2330', edgecolor='#2A3340',
        labelcolor='#E8EAF0', framealpha=0.9,
        bbox_to_anchor=(0.5, -0.08)
    )
    ax2.text(0, 0, f"{len(df)}\narticles", ha='center', va='center',
             fontsize=11, color='#E8EAF0', fontweight='bold',
             fontfamily='DejaVu Sans')

    plt.tight_layout()
    st.pyplot(fig2, use_container_width=True)
    plt.close()


# ─── Tableau détaillé ─────────────────────────────────────────────────────────
st.markdown("""
<div class="section-header">
    <span class="section-dot"></span> Détail des Articles — Vue Complète
</div>
""", unsafe_allow_html=True)

# Préparation tableau
def format_alerte(a):
    if a == 'RUPTURE STOCK':
        return '🔴 RUPTURE STOCK'
    elif a.startswith('RUPTURE J+'):
        j = a.replace('RUPTURE J+', '')
        return f'🟠 RUPTURE J+{j}'
    else:
        return '✅ OK'

df_display = df_f.copy()
df_display['Alerte Système'] = df_display['Alerte LSTM'].map(format_alerte)
df_display['Couverture'] = df_display['Couverture (%)'].apply(lambda x: f"{x}%")

COLS = ['Code Article', 'Stock Actuel', 'Besoin Prévu (30j)', 'Jours Restants', 'Couverture', 'Alerte Système']
df_show = df_display[COLS]

def color_row(row):
    stat = df_display.loc[row.name, 'Statut']
    alerte_col = 'Alerte Système'
    styles = []
    for c in COLS:
        if stat == 'RUPTURE':
            if c in ['Stock Actuel', alerte_col]:
                styles.append('color: #FF6B75; font-weight: 700')
            else:
                styles.append('')
        elif stat == 'CRITIQUE':
            if c in ['Stock Actuel', alerte_col]:
                styles.append('color: #F4722B; font-weight: 700')
            else:
                styles.append('')
        elif stat == 'ATTENTION':
            if c in ['Stock Actuel', alerte_col]:
                styles.append('color: #F0A500; font-weight: 700')
            else:
                styles.append('')
        else:
            if c == alerte_col:
                styles.append('color: #2DC653; font-weight: 700')
            else:
                styles.append('')
    return styles

st.dataframe(
    df_show.style.apply(color_row, axis=1).format({'Besoin Prévu (30j)': '{:.2f}'}),
    use_container_width=True,
    height=460
)

# ─── Actions recommandées ────────────────────────────────────────────────────
st.markdown("""
<div class="section-header">
    <span class="section-dot"></span> Actions Recommandées
</div>
""", unsafe_allow_html=True)

ruptures  = df_dyn[df_dyn['Statut'] == 'RUPTURE']['Code Article'].tolist()
critiques = df_dyn[df_dyn['Statut'] == 'CRITIQUE']['Code Article'].tolist()

c1, c2, c3 = st.columns(3)
with c1:
    st.markdown(f"""
    <div style="background:#1A0D0F; border:1px solid #4A1018; border-radius:10px; padding:1rem 1.2rem;">
        <div style="color:#E63946; font-weight:700; font-size:0.85rem; text-transform:uppercase;
                    letter-spacing:0.05em; margin-bottom:0.6rem;">💥 Commande Urgente</div>
        <div style="font-size:0.8rem; color:#C0A0A5; line-height:1.7; font-family:monospace;">
            {'<br>'.join(ruptures[:6]) if ruptures else '— Aucun —'}
        </div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div style="background:#1A1108; border:1px solid #4A3010; border-radius:10px; padding:1rem 1.2rem;">
        <div style="color:#F4722B; font-weight:700; font-size:0.85rem; text-transform:uppercase;
                    letter-spacing:0.05em; margin-bottom:0.6rem;">⚠️ Planifier sous 7j</div>
        <div style="font-size:0.8rem; color:#C0A898; line-height:1.7; font-family:monospace;">
            {'<br>'.join(critiques[:6]) if critiques else '— Aucun —'}
        </div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <div style="background:#0F1A0F; border:1px solid #1A3A1A; border-radius:10px; padding:1rem 1.2rem;">
        <div style="color:#2DC653; font-weight:700; font-size:0.85rem; text-transform:uppercase;
                    letter-spacing:0.05em; margin-bottom:0.6rem;">✅ En Stock Suffisant</div>
        <div style="font-size:0.8rem; color:#90B098; line-height:1.7; font-family:monospace;">
            {n_ok} article(s) couverts au-delà de 30 jours.<br><br>
            <span style="color:#2DC653;">Stock total :</span> {stock_total:,.0f} unités<br>
            <span style="color:#2DC653;">Taux de dispo :</span> {taux_dispo}%
        </div>
    </div>
    """, unsafe_allow_html=True)

# ─── Footer ──────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="margin-top: 3rem; padding-top: 1rem; border-top: 1px solid #2A3340;
            text-align: center; color: #7A8599; font-size: 0.72rem;">
    Sonatrach · DAT · Système de Prédiction des Ruptures de Stock — Modèle LSTM
    &nbsp;|&nbsp; Horizon actif : <strong style="color:#F0A500;">J+{seuil_jours}</strong>
    &nbsp;|&nbsp; Critique ≤ {seuil_critique}j · Attention ≤ {seuil_attention}j
    &nbsp;|&nbsp; <strong style="color:#F0A500;">v2.1</strong>
</div>
""", unsafe_allow_html=True)
