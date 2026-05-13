import streamlit as st
import pandas as pd
import seaborn as sns
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


# ─── Données ──────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    stocks_reels = {
        '538Y042219': 11707, '586L015592': 1007, '584C110991': 376, '538Y041201': 1359,
        '584W011710': 0,    '584C110457': 1339, '584C113270': 0,    '584W010711': 0,
        '538Y042606': 1154, '538Y030905': 526,  '584C030965': 0,    '536Y200600': 0,
        '584C030232': 108,  '538Y042632': 628,  '588W662595': 23405,'538Y042626': 43,
        '584J250350': 4834, '584C110023': 457,  '586M038493': 225,  '586L015590': 0,
        '584J250270': 8802, '588W662525': 0,    '584W010713': 0
    }

    categories = {
        '538Y042219': 'Vanne',    '586L015592': 'Joint',    '584C110991': 'Filtre',
        '538Y041201': 'Pompe',    '584W011710': 'Vanne',    '584C110457': 'Joint',
        '584C113270': 'Filtre',   '584W010711': 'Vanne',    '538Y042606': 'Pompe',
        '538Y030905': 'Compresseur','584C030965':'Filtre',  '536Y200600': 'Joint',
        '584C030232': 'Compresseur','538Y042632':'Vanne',   '588W662595': 'Vanne',
        '538Y042626': 'Joint',    '584J250350': 'Pompe',    '584C110023': 'Filtre',
        '586M038493': 'Joint',    '586L015590': 'Joint',    '584J250270': 'Pompe',
        '588W662525': 'Compresseur','584W010713':'Vanne'
    }

    conso_moy = 40
    lignes = []
    for item, stock in stocks_reels.items():
        jours_restants = 0 if stock == 0 else int(stock / conso_moy)
        besoin_30j = round(conso_moy * 30, 2)
        couverture_pct = min(100, round(stock / besoin_30j * 100)) if besoin_30j > 0 else 0

        if stock == 0:
            statut = "RUPTURE"
            priorite = 0
        elif jours_restants <= 7:
            statut = "CRITIQUE"
            priorite = 1
        elif jours_restants <= 30:
            statut = "ATTENTION"
            priorite = 2
        else:
            statut = "OK"
            priorite = 3

        lignes.append({
            'Code Article':       item,
            'Catégorie':          categories.get(item, 'Autre'),
            'Stock Actuel':       stock,
            'Besoin 30j':         besoin_30j,
            'Jours Restants':     jours_restants,
            'Couverture (%)':     couverture_pct,
            'Statut':             statut,
            'Priorité':           priorite
        })
    return pd.DataFrame(lignes).sort_values('Priorité')

df = load_data()

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

    filtre_cat = st.multiselect(
        "Catégorie",
        options=sorted(df['Catégorie'].unique()),
        default=sorted(df['Catégorie'].unique())
    )

    seuil_jours = st.slider("Horizon d'alerte (jours)", 1, 90, 30)

    st.markdown("---")
    st.markdown("""
    <div style="font-size:0.72rem; color:#7A8599; line-height:1.6;">
        <strong style="color:#F0A500;">Direction Approvisionnement<br>et Transport (DAT)</strong><br><br>
        Modèle LSTM · Horizon 30j<br>
        Seuil critique : <strong style="color:#E63946;">≤ 7 jours</strong><br>
        Seuil attention : <strong style="color:#F4722B;">≤ 30 jours</strong>
    </div>
    """, unsafe_allow_html=True)

# ─── Filtrage ────────────────────────────────────────────────────────────────
df_f = df[
    df['Statut'].isin(filtre_statut) &
    df['Catégorie'].isin(filtre_cat)
]

# ─── Bannière ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
    <div>
        <p class="hero-title">🛢️ Tableau de Bord Prédictif — Gestion de Stock</p>
        <p class="hero-sub">Direction Approvisionnement et Transport · Modèle LSTM · Horizon 30 jours</p>
    </div>
    <div class="hero-badge">⚡ Temps Réel</div>
</div>
""", unsafe_allow_html=True)

# ─── KPIs ─────────────────────────────────────────────────────────────────────
n_rupture  = len(df[df['Statut'] == 'RUPTURE'])
n_critique = len(df[df['Statut'] == 'CRITIQUE'])
n_attention= len(df[df['Statut'] == 'ATTENTION'])
n_ok       = len(df[df['Statut'] == 'OK'])
stock_total= df['Stock Actuel'].sum()
taux_dispo = round(n_ok / len(df) * 100, 1)

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
        <div class="kpi-label">Alertes Critiques ≤ 7j</div>
        <div class="kpi-delta warn">▲ Action immédiate requise</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="kpi-card info">
        <span class="kpi-icon">🔶</span>
        <div class="kpi-value">{n_attention}</div>
        <div class="kpi-label">Articles en Surveillance</div>
        <div class="kpi-delta warn">Horizon &lt; 30 jours</div>
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

    # Couleur de ligne selon statut
    row_colors = []
    for _, r in df_f.iterrows():
        if r['Statut'] == 'RUPTURE':  row_colors.append('#E63946')
        elif r['Statut'] == 'CRITIQUE': row_colors.append('#F4722B')
        elif r['Statut'] == 'ATTENTION': row_colors.append('#F0A500')
        else: row_colors.append('#2DC653')

    matrice = []
    for _, row in df_f.iterrows():
        j = int(row['Jours Restants'])
        if j == 0 and row['Stock Actuel'] == 0:
            matrice.append([0] * 30)
        elif j > 30:
            matrice.append([1] * 30)
        else:
            matrice.append([1] * j + [0] * (30 - j))

    fig, ax = plt.subplots(figsize=(11, max(4, len(df_f) * 0.38)))
    fig.patch.set_facecolor('#161B22')
    ax.set_facecolor('#161B22')

    # Heatmap avec palette personnalisée
    from matplotlib.colors import LinearSegmentedColormap
    cmap_custom = LinearSegmentedColormap.from_list(
        'sonatrach', ['#3D0B0E', '#E63946', '#F0A500', '#2DC653'], N=256
    )
    # Heatmap binaire rouge/vert
    cmap_bin = LinearSegmentedColormap.from_list('bin', ['#3D0B0E', '#1A3A1F'], N=2)

    sns.heatmap(
        matrice,
        cmap=['#4A1018', '#1E3A24'],
        cbar=False,
        yticklabels=df_f['Code Article'].values,
        xticklabels=[str(i) if i % 5 == 0 else '' for i in range(1, 31)],
        ax=ax,
        linewidths=0.3,
        linecolor='#0D1117'
    )

    # Colorier les labels Y selon statut
    ytick_labels = ax.get_yticklabels()
    for label, color in zip(ytick_labels, row_colors):
        label.set_color(color)
        label.set_fontsize(8.5)
        label.set_fontfamily('monospace')

    ax.set_xticklabels(ax.get_xticklabels(), color='#7A8599', fontsize=8)
    ax.tick_params(axis='both', which='both', length=0, pad=6)
    ax.set_xlabel("Jours futurs", color='#7A8599', fontsize=9, labelpad=8)
    ax.set_ylabel("")

    # Légende
    leg_patches = [
        mpatches.Patch(color='#1E3A24', label='Stock disponible'),
        mpatches.Patch(color='#4A1018', label='Rupture prévue'),
    ]
    ax.legend(handles=leg_patches, loc='lower right', fontsize=8,
              facecolor='#1C2330', edgecolor='#2A3340',
              labelcolor='#E8EAF0', framealpha=0.9)

    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout(pad=0.5)
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

    # Barres par catégorie
    st.markdown("""
    <div class="section-header" style="margin-top:1rem">
        <span class="section-dot"></span> Stock par Catégorie
    </div>
    """, unsafe_allow_html=True)

    cat_stock = df.groupby('Catégorie')['Stock Actuel'].sum().sort_values(ascending=True)
    fig3, ax3 = plt.subplots(figsize=(5, 3))
    fig3.patch.set_facecolor('#161B22')
    ax3.set_facecolor('#161B22')

    bars = ax3.barh(
        cat_stock.index, cat_stock.values,
        color=['#F0A500' if v == cat_stock.max() else '#2A3340' for v in cat_stock.values],
        edgecolor='none', height=0.6
    )

    for bar, val in zip(bars, cat_stock.values):
        ax3.text(val + cat_stock.max() * 0.01, bar.get_y() + bar.get_height() / 2,
                 f'{val:,.0f}', va='center', ha='left', fontsize=8,
                 color='#7A8599', fontfamily='monospace')

    ax3.set_facecolor('#161B22')
    ax3.tick_params(colors='#7A8599', labelsize=8)
    for sp in ax3.spines.values(): sp.set_visible(False)
    ax3.set_xlabel("Unités en stock", color='#7A8599', fontsize=8)
    ax3.xaxis.label.set_color('#7A8599')
    ax3.tick_params(axis='x', colors='#7A8599')
    ax3.tick_params(axis='y', colors='#E8EAF0')

    plt.tight_layout(pad=0.5)
    st.pyplot(fig3, use_container_width=True)
    plt.close()

# ─── Tableau détaillé ─────────────────────────────────────────────────────────
st.markdown("""
<div class="section-header">
    <span class="section-dot"></span> Détail des Articles — Vue Complète
</div>
""", unsafe_allow_html=True)

# Préparation tableau
def format_statut(s):
    icons = {'RUPTURE': '💥 RUPTURE', 'CRITIQUE': '⚠️ CRITIQUE', 'ATTENTION': '🔶 ATTENTION', 'OK': '✅ OK'}
    return icons.get(s, s)

df_display = df_f.copy()
df_display['Statut Affiché'] = df_display['Statut'].map(format_statut)
df_display['Couverture'] = df_display['Couverture (%)'].apply(lambda x: f"{x}%")

# Style conditionnel
def style_table(df_in):
    styles = pd.DataFrame('', index=df_in.index, columns=df_in.columns)
    for i, row in df_in.iterrows():
        if row['Statut'] == 'RUPTURE':
            styles.loc[i, 'Stock Actuel'] = 'color: #FF6B75; font-weight: 600'
            styles.loc[i, 'Statut Affiché'] = 'color: #FF6B75; font-weight: 600'
        elif row['Statut'] == 'CRITIQUE':
            styles.loc[i, 'Stock Actuel'] = 'color: #F4722B; font-weight: 600'
            styles.loc[i, 'Statut Affiché'] = 'color: #F4722B; font-weight: 600'
        elif row['Statut'] == 'ATTENTION':
            styles.loc[i, 'Stock Actuel'] = 'color: #F0A500; font-weight: 600'
            styles.loc[i, 'Statut Affiché'] = 'color: #F0A500; font-weight: 600'
        else:
            styles.loc[i, 'Statut Affiché'] = 'color: #2DC653; font-weight: 600'
    return styles

cols_show = ['Code Article', 'Catégorie', 'Stock Actuel', 'Besoin 30j', 'Jours Restants', 'Couverture', 'Statut Affiché']
df_show = df_display[cols_show].rename(columns={'Statut Affiché': 'Statut'})

st.dataframe(
    df_show.style.apply(lambda _: style_table(df_display)[['Code Article','Catégorie','Stock Actuel','Besoin 30j','Jours Restants','Couverture (%)','Statut']].rename(columns={'Statut':'Statut Affiché'}).values, axis=None),
    use_container_width=True,
    height=420
)

# ─── Actions recommandées ────────────────────────────────────────────────────
st.markdown("""
<div class="section-header">
    <span class="section-dot"></span> Actions Recommandées
</div>
""", unsafe_allow_html=True)

ruptures  = df[df['Statut'] == 'RUPTURE']['Code Article'].tolist()
critiques = df[df['Statut'] == 'CRITIQUE']['Code Article'].tolist()

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
st.markdown("""
<div style="margin-top: 3rem; padding-top: 1rem; border-top: 1px solid #2A3340;
            text-align: center; color: #7A8599; font-size: 0.72rem;">
    Sonatrach · DAT · Système de Prédiction des Ruptures de Stock — Modèle LSTM
    &nbsp;|&nbsp; Données actualisées en temps réel &nbsp;|&nbsp; <strong style="color:#F0A500;">v2.0</strong>
</div>
""", unsafe_allow_html=True)
