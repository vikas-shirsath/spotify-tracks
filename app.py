"""
===============================================================================
 SPOTIFY BIG DATA ANALYTICS — STREAMLIT DASHBOARD
 -----------------------------------------------------------------------------
 Interactive visual dashboard analyzing 114,000+ Spotify tracks.
 Focus: Audio features (Danceability, Energy, Tempo, Loudness, etc.) vs
        Track Popularity & Multi-Genre Comparisons.
===============================================================================
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ─────────────────────────────────────────────────────────────
# 1. PAGE CONFIGURATION & CUSTOM SPOTIFY THEME CSS
# ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Spotify Big Data Analytics | Audio Features & Genres",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

SPOTIFY_GREEN = "#1DB954"
SPOTIFY_LIGHT_GREEN = "#1ed760"
SPOTIFY_DARK = "#121212"
SPOTIFY_CARD_BG = "#181818"
SPOTIFY_HOVER = "#282828"
SPOTIFY_TEXT = "#FFFFFF"
SPOTIFY_MUTED = "#B3B3B3"

CUSTOM_CSS = f"""
<style>
    /* Global background and font styling */
    .stApp {{
        background: linear-gradient(180deg, #121212 0%, #0a0a0a 100%);
        color: {SPOTIFY_TEXT};
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }}

    /* Custom Header Banner */
    .spotify-hero {{
        background: linear-gradient(135deg, rgba(29, 185, 84, 0.22) 0%, rgba(18, 18, 18, 0.95) 100%);
        border: 1px solid rgba(29, 185, 84, 0.35);
        border-radius: 16px;
        padding: 24px 32px;
        margin-bottom: 24px;
        backdrop-filter: blur(10px);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
    }}
    .spotify-hero h1 {{
        color: #ffffff;
        font-size: 2.2rem;
        font-weight: 800;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 12px;
    }}
    .spotify-hero p {{
        color: {SPOTIFY_MUTED};
        font-size: 1.05rem;
        margin: 8px 0 0 0;
    }}
    .hero-badge {{
        display: inline-block;
        background-color: {SPOTIFY_GREEN};
        color: #000000;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 700;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }}

    /* Metric Card Styling */
    .metric-card {{
        background: {SPOTIFY_CARD_BG};
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 16px 20px;
        transition: transform 0.2s ease, border-color 0.2s ease;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.3);
    }}
    .metric-card:hover {{
        transform: translateY(-2px);
        border-color: {SPOTIFY_GREEN};
    }}
    .metric-title {{
        color: {SPOTIFY_MUTED};
        font-size: 0.82rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 4px;
    }}
    .metric-value {{
        color: #ffffff;
        font-size: 1.8rem;
        font-weight: 800;
        margin: 0;
    }}
    .metric-subtitle {{
        color: {SPOTIFY_GREEN};
        font-size: 0.82rem;
        font-weight: 600;
        margin-top: 4px;
    }}

    /* Styled Tabs */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 8px;
        background-color: transparent;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        padding-bottom: 4px;
    }}
    .stTabs [data-baseweb="tab"] {{
        background-color: rgba(255, 255, 255, 0.04);
        border-radius: 8px 8px 0 0;
        color: {SPOTIFY_MUTED};
        padding: 10px 20px;
        font-weight: 600;
        border: none;
    }}
    .stTabs [aria-selected="true"] {{
        background-color: rgba(29, 185, 84, 0.15) !important;
        color: {SPOTIFY_GREEN} !important;
        border-bottom: 3px solid {SPOTIFY_GREEN} !important;
    }}

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {{
        background-color: #0d0d0d;
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }}

    /* Badges */
    .spark-badge {{
        background: linear-gradient(135deg, #E25A1C 0%, #FF8C00 100%);
        color: white;
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 700;
    }}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# 2. DATA LOADING & PREPROCESSING (CACHED)
# ─────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="⚡ Ingesting & indexing Spotify tracks dataset...")
def load_and_preprocess_data():
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "spotify-tracks-dataset.csv")
    if not os.path.exists(csv_path):
        st.error(f"Dataset not found at {csv_path}!")
        return pd.DataFrame()

    df = pd.read_csv(csv_path, low_memory=False)

    # Drop index columns
    drop_cols = [c for c in df.columns if "Unnamed" in c or c == "" or c == "_c0"]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    # Cast numeric columns safely
    numeric_cols = [
        "popularity", "duration_ms", "danceability", "energy",
        "loudness", "speechiness", "acousticness", "instrumentalness",
        "liveness", "valence", "tempo"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows missing crucial values
    df = df.dropna(subset=["popularity", "danceability", "energy", "tempo", "track_genre"])
    df = df[(df["popularity"] >= 0) & (df["popularity"] <= 100)]

    # Add Popularity Tier
    tier_conditions = [
        (df["popularity"] >= 80),
        (df["popularity"] >= 60) & (df["popularity"] < 80),
        (df["popularity"] >= 40) & (df["popularity"] < 60),
        (df["popularity"] >= 20) & (df["popularity"] < 40),
        (df["popularity"] < 20)
    ]
    tier_labels = [
        "⭐ Very High (80-100)",
        "🟢 High (60-79)",
        "🟡 Medium (40-59)",
        "🟠 Low (20-39)",
        "⚪ Very Low (0-19)"
    ]
    df["popularity_tier"] = np.select(tier_conditions, tier_labels, default="Unknown")

    return df


df_raw = load_and_preprocess_data()
if df_raw.empty:
    st.stop()


# ─────────────────────────────────────────────────────────────
# 3. SIDEBAR FILTERS & CONTROLS
# ─────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### 🎛️ Analytics Controls")
    st.caption("Filter 114,000+ tracks in real time")

    # Text Search Filter
    search_query = st.text_input("🔍 Search Track or Artist", placeholder="e.g. Taylor Swift, Queen...")

    # Genre Multi-Select
    all_genres = sorted(df_raw["track_genre"].dropna().unique().tolist())
    top_genres_preset = ["pop", "rock", "k-pop", "edm", "hip-hop", "dance", "latin", "indie", "chill", "classical"]

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("Top Genres", use_container_width=True):
            st.session_state["selected_genres"] = [g for g in top_genres_preset if g in all_genres]
    with col_btn2:
        if st.button("Clear Genres", use_container_width=True):
            st.session_state["selected_genres"] = []

    selected_genres = st.multiselect(
        "Select Music Genres",
        options=all_genres,
        default=st.session_state.get("selected_genres", [g for g in top_genres_preset if g in all_genres]),
        key="selected_genres_widget"
    )

    st.markdown("---")
    st.markdown("#### 🎯 Audio Feature Filters")

    # Popularity Slider
    pop_range = st.slider("Popularity Score Range", 0, 100, (0, 100))

    # Danceability Slider
    dance_range = st.slider("Danceability Range", 0.0, 1.0, (0.0, 1.0), step=0.05)

    # Energy Slider
    energy_range = st.slider("Energy Range", 0.0, 1.0, (0.0, 1.0), step=0.05)

    # Tempo Slider
    min_tempo = int(df_raw["tempo"].min())
    max_tempo = int(df_raw["tempo"].max())
    tempo_range = st.slider("Tempo Range (BPM)", min_tempo, max_tempo, (min_tempo, max_tempo))

    # Explicit Filter
    explicit_choice = st.radio("Explicit Content", ["All Tracks", "Clean Only", "Explicit Only"], horizontal=True)

    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #888; font-size: 0.8rem;'>
            Powered by <b>Apache Spark</b> & <b>Streamlit</b><br>
            Big Data Analytics Project
        </div>
        """,
        unsafe_allow_html=True
    )


# ─────────────────────────────────────────────────────────────
# 4. FILTERING APPLICATION
# ─────────────────────────────────────────────────────────────

df = df_raw.copy()

# Filter by genres if any selected
if selected_genres:
    df = df[df["track_genre"].isin(selected_genres)]

# Filter by search query
if search_query:
    q = search_query.lower()
    df = df[
        df["track_name"].astype(str).str.lower().str.contains(q) |
        df["artists"].astype(str).str.lower().str.contains(q)
    ]

# Filter ranges
df = df[
    (df["popularity"] >= pop_range[0]) & (df["popularity"] <= pop_range[1]) &
    (df["danceability"] >= dance_range[0]) & (df["danceability"] <= dance_range[1]) &
    (df["energy"] >= energy_range[0]) & (df["energy"] <= energy_range[1]) &
    (df["tempo"] >= tempo_range[0]) & (df["tempo"] <= tempo_range[1])
]

# Explicit filter
if explicit_choice == "Clean Only" and "explicit" in df.columns:
    df = df[df["explicit"] == False]
elif explicit_choice == "Explicit Only" and "explicit" in df.columns:
    df = df[df["explicit"] == True]


# ─────────────────────────────────────────────────────────────
# 5. HERO HEADER & KPI METRICS
# ─────────────────────────────────────────────────────────────

st.markdown(
    f"""
    <div class="spotify-hero">
        <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap;">
            <div>
                <span class="hero-badge">Big Data Analytics Mini Project</span>
                <h1>🎵 Spotify Tracks Audio Feature & Popularity Analytics</h1>
                <p>Exploring how <b>Danceability</b>, <b>Energy</b>, and <b>Tempo</b> drive track popularity across 114+ music genres.</p>
            </div>
            <div style="margin-top: 10px;">
                <span class="spark-badge">⚡ PySpark Distributed Engine</span>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# KPI Cards row
kpi_col1, kpi_col2, kpi_col3, kpi_col4, kpi_col5 = st.columns(5)

total_filtered = len(df)
avg_pop = df["popularity"].mean() if not df.empty else 0
avg_dance = df["danceability"].mean() if not df.empty else 0
avg_energy = df["energy"].mean() if not df.empty else 0
avg_tempo = df["tempo"].mean() if not df.empty else 0

top_genre = df.groupby("track_genre")["popularity"].mean().idxmax() if not df.empty else "N/A"
top_genre_score = df.groupby("track_genre")["popularity"].mean().max() if not df.empty else 0

with kpi_col1:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">Tracks Filtered</div>
            <div class="metric-value">{total_filtered:,}</div>
            <div class="metric-subtitle">of {len(df_raw):,} total tracks</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with kpi_col2:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">Avg Popularity</div>
            <div class="metric-value">{avg_pop:.1f}</div>
            <div class="metric-subtitle">Score (0-100 scale)</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with kpi_col3:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">Avg Danceability</div>
            <div class="metric-value">{avg_dance:.3f}</div>
            <div class="metric-subtitle">Rhythm stability index</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with kpi_col4:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">Avg Energy & Tempo</div>
            <div class="metric-value">{avg_energy:.2f} <span style="font-size: 1rem; color: #888;">/ {avg_tempo:.0f}</span></div>
            <div class="metric-subtitle">Energy / BPM</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with kpi_col5:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">Top Filtered Genre</div>
            <div class="metric-value" style="font-size: 1.4rem; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">{top_genre}</div>
            <div class="metric-subtitle">Avg Pop: {top_genre_score:.1f}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("<br>", unsafe_allow_html=True)

if df.empty:
    st.warning("⚠️ No tracks match your filter criteria. Please broaden your filters in the sidebar.")
    st.stop()


# ─────────────────────────────────────────────────────────────
# 6. DASHBOARD TABS
# ─────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Audio Features vs Popularity",
    "🎸 Genre Comparison & Deep Dive",
    "🎵 Track Explorer & Viral Songs",
    "⚡ Big Data & PySpark Engine"
])


# ═════════════════════════════════════════════════════════════
# TAB 1: AUDIO FEATURES vs POPULARITY
# ═════════════════════════════════════════════════════════════
with tab1:
    st.markdown("### 🔍 How Audio Features Relate to Track Popularity")
    st.caption("Pearson correlation analysis and multi-feature regressions on 114K tracks")

    # Feature List
    feat_cols = [
        "danceability", "energy", "tempo", "loudness",
        "valence", "speechiness", "acousticness",
        "instrumentalness", "liveness", "popularity"
    ]
    feat_cols_avail = [c for c in feat_cols if c in df.columns]

    corr_df = df[feat_cols_avail].corr()

    col_t1a, col_t1b = st.columns([1.2, 1])

    with col_t1a:
        # Plotly Heatmap
        fig_heat = px.imshow(
            corr_df.round(3),
            text_auto=True,
            aspect="auto",
            color_continuous_scale="Viridis",
            title="Correlation Matrix (Audio Features & Popularity)"
        )
        fig_heat.update_layout(
            paper_bgcolor=SPOTIFY_CARD_BG,
            plot_bgcolor=SPOTIFY_CARD_BG,
            font_color=SPOTIFY_TEXT,
            margin=dict(l=40, r=40, t=50, b=40),
            coloraxis_colorbar=dict(title="Pearson r")
        )
        st.plotly_chart(fig_heat, use_container_width=True)

    with col_t1b:
        # Correlation Bar Chart with Popularity
        if "popularity" in corr_df.columns:
            pop_corr = corr_df["popularity"].drop("popularity").sort_values(ascending=True)
            colors = [SPOTIFY_GREEN if val > 0 else "#E74C3C" for val in pop_corr.values]

            fig_bar = go.Figure(go.Bar(
                x=pop_corr.values,
                y=pop_corr.index,
                orientation="h",
                marker=dict(color=colors),
                text=[f"{v:+.3f}" for v in pop_corr.values],
                textposition="outside"
            ))
            fig_bar.update_layout(
                title="Correlation with Popularity Score",
                xaxis_title="Pearson Correlation (r)",
                yaxis_title="Feature",
                paper_bgcolor=SPOTIFY_CARD_BG,
                plot_bgcolor=SPOTIFY_CARD_BG,
                font_color=SPOTIFY_TEXT,
                xaxis=dict(range=[-0.18, 0.18], gridcolor="rgba(255,255,255,0.08)"),
                margin=dict(l=40, r=40, t=50, b=40)
            )
            st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("---")

    # Interactive Scatter & Regression
    col_scat_ctrl1, col_scat_ctrl2 = st.columns([1, 3])
    with col_scat_ctrl1:
        feature_choice = st.selectbox(
            "Select Audio Feature to Compare with Popularity",
            options=["danceability", "energy", "tempo", "loudness", "valence", "acousticness", "instrumentalness"],
            index=0
        )
        color_by = st.selectbox("Color By", ["track_genre", "popularity_tier"], index=1)
        sample_size = st.slider("Sample Size for Scatter Plot", 1000, min(10000, len(df)), 3000, step=500)

    with col_scat_ctrl2:
        sampled_df = df.sample(n=min(sample_size, len(df)), random_state=42)
        fig_scatter = px.scatter(
            sampled_df,
            x=feature_choice,
            y="popularity",
            color=color_by,
            hover_data=["track_name", "artists", "track_genre"],
            trendline="ols",
            title=f"Scatter Trend: {feature_choice.capitalize()} vs Popularity (Sampled {len(sampled_df):,} tracks)",
            color_discrete_sequence=px.colors.qualitative.Prism
        )
        fig_scatter.update_layout(
            paper_bgcolor=SPOTIFY_CARD_BG,
            plot_bgcolor=SPOTIFY_CARD_BG,
            font_color=SPOTIFY_TEXT,
            xaxis=dict(gridcolor="rgba(255,255,255,0.08)"),
            yaxis=dict(gridcolor="rgba(255,255,255,0.08)"),
            margin=dict(l=40, r=40, t=50, b=40)
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    # Popularity Tier Breakdown
    st.markdown("#### 🏆 Feature Comparison by Popularity Tiers")
    tier_summary = df.groupby("popularity_tier")[["danceability", "energy", "valence", "acousticness"]].mean().reset_index()

    tier_order = [
        "⭐ Very High (80-100)",
        "🟢 High (60-79)",
        "🟡 Medium (40-59)",
        "🟠 Low (20-39)",
        "⚪ Very Low (0-19)"
    ]
    tier_summary["tier_order"] = tier_summary["popularity_tier"].map(lambda x: tier_order.index(x) if x in tier_order else 99)
    tier_summary = tier_summary.sort_values("tier_order").drop(columns=["tier_order"])

    fig_tier = px.bar(
        tier_summary,
        x="popularity_tier",
        y=["danceability", "energy", "valence", "acousticness"],
        barmode="group",
        title="Average Audio Features Across Popularity Tiers",
        color_discrete_sequence=["#1DB954", "#3498DB", "#E67E22", "#9B59B6"]
    )
    fig_tier.update_layout(
        paper_bgcolor=SPOTIFY_CARD_BG,
        plot_bgcolor=SPOTIFY_CARD_BG,
        font_color=SPOTIFY_TEXT,
        xaxis_title="",
        yaxis_title="Average Metric Value (0 - 1)",
        legend_title="Feature",
        margin=dict(l=40, r=40, t=50, b=40),
        yaxis=dict(gridcolor="rgba(255,255,255,0.08)")
    )
    st.plotly_chart(fig_tier, use_container_width=True)


# ═════════════════════════════════════════════════════════════
# TAB 2: GENRE COMPARISON & DEEP DIVE
# ═════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### 🎸 Genre Comparison & Acoustic Profiles")
    st.caption("Compare how audio characteristics differ between genres and sub-genres")

    # Genre aggregation
    genre_agg = df_raw.groupby("track_genre").agg(
        track_count=("track_name", "count"),
        avg_popularity=("popularity", "mean"),
        avg_danceability=("danceability", "mean"),
        avg_energy=("energy", "mean"),
        avg_tempo=("tempo", "mean"),
        avg_valence=("valence", "mean"),
        avg_acousticness=("acousticness", "mean")
    ).reset_index()

    col_g1, col_g2 = st.columns(2)

    with col_g1:
        top_n = st.slider("Select Top N Genres", 5, 30, 15)
        top_genres = genre_agg.sort_values("avg_popularity", ascending=False).head(top_n)

        fig_top_g = px.bar(
            top_genres,
            x="avg_popularity",
            y="track_genre",
            orientation="h",
            color="avg_popularity",
            color_continuous_scale="Greens",
            title=f"Top {top_n} Genres by Average Popularity",
            text="avg_popularity"
        )
        fig_top_g.update_traces(texttemplate="%{text:.1f}", textposition="inside")
        fig_top_g.update_layout(
            yaxis=dict(autorange="reversed"),
            paper_bgcolor=SPOTIFY_CARD_BG,
            plot_bgcolor=SPOTIFY_CARD_BG,
            font_color=SPOTIFY_TEXT,
            margin=dict(l=40, r=40, t=50, b=40)
        )
        st.plotly_chart(fig_top_g, use_container_width=True)

    with col_g2:
        # Bottom genres
        bottom_genres = genre_agg.sort_values("avg_popularity", ascending=True).head(top_n)
        fig_bot_g = px.bar(
            bottom_genres,
            x="avg_popularity",
            y="track_genre",
            orientation="h",
            color="avg_popularity",
            color_continuous_scale="Reds_r",
            title=f"Bottom {top_n} Genres by Average Popularity",
            text="avg_popularity"
        )
        fig_bot_g.update_traces(texttemplate="%{text:.1f}", textposition="outside")
        fig_bot_g.update_layout(
            paper_bgcolor=SPOTIFY_CARD_BG,
            plot_bgcolor=SPOTIFY_CARD_BG,
            font_color=SPOTIFY_TEXT,
            margin=dict(l=40, r=40, t=50, b=40)
        )
        st.plotly_chart(fig_bot_g, use_container_width=True)

    st.markdown("---")

    # Radar Comparison Between Selected Genres
    st.markdown("#### 🎯 Head-to-Head Genre Radar Comparison")
    default_compare = ["pop", "rock", "k-pop", "edm"] if all(x in all_genres for x in ["pop", "rock", "k-pop", "edm"]) else all_genres[:4]
    compare_genres = st.multiselect(
        "Choose 2 to 6 Genres to Compare",
        options=all_genres,
        default=default_compare
    )

    if compare_genres:
        radar_categories = ["Danceability", "Energy", "Valence", "Acousticness", "Tempo (Norm)"]
        fig_radar = go.Figure()

        # Compute max tempo for normalization
        max_t = genre_agg["avg_tempo"].max()

        for g in compare_genres:
            g_data = genre_agg[genre_agg["track_genre"] == g]
            if not g_data.empty:
                r_values = [
                    g_data["avg_danceability"].values[0],
                    g_data["avg_energy"].values[0],
                    g_data["avg_valence"].values[0],
                    g_data["avg_acousticness"].values[0],
                    g_data["avg_tempo"].values[0] / max_t
                ]
                # close loop
                r_values.append(r_values[0])
                categories_closed = radar_categories + [radar_categories[0]]

                fig_radar.add_trace(go.Scatterpolar(
                    r=r_values,
                    theta=categories_closed,
                    fill="toself",
                    name=g
                ))

        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 1], gridcolor="rgba(255,255,255,0.1)"),
                bgcolor=SPOTIFY_CARD_BG
            ),
            paper_bgcolor=SPOTIFY_CARD_BG,
            font_color=SPOTIFY_TEXT,
            title="Acoustic DNA Radar: Audio Features Across Selected Genres",
            margin=dict(l=60, r=60, t=50, b=40)
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    # Box Plot Distribution for Selected Genres
    st.markdown("#### 📦 Feature Distribution Spread Across Genres")
    sub_df = df_raw[df_raw["track_genre"].isin(compare_genres if compare_genres else top_genres_preset)]

    box_metric = st.selectbox("Select Audio Metric for Box Plot", ["danceability", "energy", "tempo", "valence", "popularity"])
    fig_box = px.box(
        sub_df,
        x="track_genre",
        y=box_metric,
        color="track_genre",
        points=False,
        title=f"Distribution of {box_metric.capitalize()} by Genre",
        color_discrete_sequence=px.colors.qualitative.Bold
    )
    fig_box.update_layout(
        paper_bgcolor=SPOTIFY_CARD_BG,
        plot_bgcolor=SPOTIFY_CARD_BG,
        font_color=SPOTIFY_TEXT,
        xaxis_title="Genre",
        yaxis_title=box_metric.capitalize(),
        showlegend=False,
        margin=dict(l=40, r=40, t=50, b=40)
    )
    st.plotly_chart(fig_box, use_container_width=True)


# ═════════════════════════════════════════════════════════════
# TAB 3: TRACK EXPLORER & VIRAL SONGS
# ═════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### 🎧 Interactive Track Explorer")
    st.caption("Inspect individual tracks, filter viral hits, and export datasets")

    col_track1, col_track2 = st.columns([2, 1])

    with col_track1:
        st.markdown("#### 🌟 Top 10 Most Popular Tracks (Current Selection)")
        top10_tracks = df.sort_values("popularity", ascending=False).head(10)[
            ["track_name", "artists", "track_genre", "popularity", "danceability", "energy", "tempo"]
        ]
        st.dataframe(
            top10_tracks.style.format({
                "popularity": "{:.0f}",
                "danceability": "{:.3f}",
                "energy": "{:.3f}",
                "tempo": "{:.1f} BPM"
            }),
            use_container_width=True
        )

    with col_track2:
        st.markdown("#### ⚡ High Energy & Danceable Outliers")
        viral_outliers = df[(df["danceability"] > 0.75) & (df["energy"] > 0.85) & (df["popularity"] > 70)]
        st.write(f"Found **{len(viral_outliers)}** tracks matching club hit criteria (Danceability > 0.75, Energy > 0.85, Popularity > 70).")
        if not viral_outliers.empty:
            st.dataframe(
                viral_outliers[["track_name", "artists", "popularity"]].head(6),
                use_container_width=True
            )

    st.markdown("---")
    st.markdown("#### 📋 Filtered Tracks Dataset")

    cols_to_display = [
        "track_name", "artists", "album_name", "track_genre",
        "popularity", "danceability", "energy", "tempo", "loudness", "valence"
    ]
    cols_to_display = [c for c in cols_to_display if c in df.columns]

    st.dataframe(df[cols_to_display].head(100), use_container_width=True)

    # Download Button
    csv_data = df.head(1000).to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Current Filtered Sample (CSV)",
        data=csv_data,
        file_name="spotify_filtered_analysis.csv",
        mime="text/csv"
    )


# ═════════════════════════════════════════════════════════════
# TAB 4: BIG DATA & PYSPARK ENGINE
# ═════════════════════════════════════════════════════════════
with tab4:
    st.markdown("### ⚡ Big Data Architecture & PySpark Integration")
    st.caption("How distributed computing scales analytics over hundreds of thousands of audio records")

    col_spark1, col_spark2 = st.columns(2)

    with col_spark1:
        st.markdown(
            f"""
            <div style="background: {SPOTIFY_CARD_BG}; padding: 20px; border-radius: 12px; border: 1px solid rgba(255,255,255,0.08);">
                <h4 style="color: {SPOTIFY_GREEN}; margin-top: 0;">Distributed Analytics Highlights</h4>
                <ul style="color: {SPOTIFY_MUTED}; line-height: 1.8;">
                    <li><b>DataFrame In-Memory Caching:</b> <code>df.cache()</code> stores 114K records across JVM memory for sub-second iterative queries.</li>
                    <li><b>PySpark ML VectorAssembler:</b> Assembles multi-dimensional audio metrics into dense feature vectors for distributed Pearson matrix calculation.</li>
                    <li><b>Catalyst Query Optimizer:</b> Translates high-level SQL aggregations into optimized physical execution plans (DAG).</li>
                    <li><b>Window Functions:</b> Executes <code>DENSE_RANK() OVER (ORDER BY avg_pop DESC)</code> across partitioned genre blocks.</li>
                    <li><b>Fault Tolerant Type Casting:</b> Handled corrupted text values safely using Spark's <code>try_cast</code> without executor aborts.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col_spark2:
        st.markdown(
            f"""
            <div style="background: {SPOTIFY_CARD_BG}; padding: 20px; border-radius: 12px; border: 1px solid rgba(255,255,255,0.08);">
                <h4 style="color: #3498DB; margin-top: 0;">Spark SQL Query Executed</h4>
                <pre style="background: #0f0f0f; color: #1ed760; padding: 12px; border-radius: 8px; font-size: 0.82rem; overflow-x: auto;">
SELECT track_genre,
       COUNT(*) AS track_count,
       ROUND(AVG(popularity), 2) AS avg_popularity,
       ROUND(AVG(danceability), 4) AS avg_danceability,
       ROUND(AVG(energy), 4) AS avg_energy
FROM spotify_tracks
GROUP BY track_genre
HAVING AVG(popularity) > 50 AND AVG(danceability) > 0.6
ORDER BY avg_popularity DESC
LIMIT 10;
                </pre>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("---")
    st.markdown("#### 📁 Pre-computed PySpark Artifacts & Charts")
    st.write("Charts rendered by the PySpark pipeline batch job:")

    chart_col1, chart_col2, chart_col3 = st.columns(3)
    charts_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output_charts")

    with chart_col1:
        c1 = os.path.join(charts_dir, "01_correlation_heatmap.png")
        if os.path.exists(c1):
            st.image(c1, caption="Batch PySpark Correlation Heatmap", use_container_width=True)

    with chart_col2:
        c2 = os.path.join(charts_dir, "03_top20_genres_popularity.png")
        if os.path.exists(c2):
            st.image(c2, caption="Top 20 Genres (Spark GroupBy)", use_container_width=True)

    with chart_col3:
        c3 = os.path.join(charts_dir, "07_popularity_tier_features.png")
        if os.path.exists(c3):
            st.image(c3, caption="Popularity Tier Distributions", use_container_width=True)
