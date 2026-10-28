"""
===============================================================================
 BIG DATA ANALYTICS MINI PROJECT
 -----------------------------------------------
 Title   : Audio Feature Analysis of Spotify Tracks Using PySpark
 Dataset : Spotify Tracks Dataset (~114K tracks, 20+ features)
 Objective:
   1. Analyze how audio features (danceability, energy, tempo) relate
      to track popularity.
   2. Compare audio feature distributions across music genres.
   3. Build a correlation matrix of audio features vs popularity.
   4. Identify the most and least popular genres.
   5. Perform genre-wise statistical aggregation at scale using PySpark.

 Tools   : Apache Spark (PySpark), Matplotlib, Seaborn
===============================================================================
"""

# ─────────────────────────────────────────────────────────────
# 1. IMPORTS & SPARK SESSION INITIALIZATION
# ─────────────────────────────────────────────────────────────

import os
import sys
import warnings
warnings.filterwarnings("ignore")

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType,
    FloatType, DoubleType, BooleanType
)
from pyspark.sql.window import Window
from pyspark.ml.stat import Correlation
from pyspark.ml.feature import VectorAssembler

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving figures
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# ─── Output directory for charts ─────────────────────────────
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output_charts")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── Styling for all plots ───────────────────────────────────
plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("viridis")
COLORS = {
    "primary":   "#1DB954",   # Spotify Green
    "secondary": "#191414",   # Spotify Black
    "accent":    "#1ED760",
    "bg":        "#121212",
    "text":      "#FFFFFF",
}


def create_spark_session():
    """Initialize a local Spark session for big data processing."""
    spark = SparkSession.builder \
        .appName("Spotify-BDA-AudioFeature-Analysis") \
        .master("local[*]") \
        .config("spark.driver.memory", "4g") \
        .config("spark.sql.shuffle.partitions", "8") \
        .config("spark.ui.showConsoleProgress", "false") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("ERROR")
    return spark


# ─────────────────────────────────────────────────────────────
# 2. DATA LOADING & SCHEMA DEFINITION
# ─────────────────────────────────────────────────────────────

def load_data(spark, filepath):
    """
    Load the Spotify tracks CSV into a Spark DataFrame.
    Returns the cleaned DataFrame.
    """
    print("\n" + "=" * 70)
    print("  STEP 1: LOADING DATA INTO SPARK")
    print("=" * 70)

    df = spark.read.csv(filepath, header=True, inferSchema=True)

    print(f"  ✔ Loaded {df.count():,} records with {len(df.columns)} columns")
    print(f"  ✔ Partitions: {df.rdd.getNumPartitions()}")
    print(f"\n  Schema:")
    df.printSchema()

    return df


def clean_data(df):
    """
    Clean and preprocess the Spark DataFrame:
      - Drop unnecessary index columns
      - Safely cast string columns to numeric (handles malformed values)
      - Remove rows where key numeric columns failed to cast
      - Filter out invalid popularity values
    """
    print("\n" + "=" * 70)
    print("  STEP 2: DATA CLEANING & PREPROCESSING")
    print("=" * 70)

    initial_count = df.count()

    # Drop unnamed index columns
    cols_to_drop = [c for c in df.columns if "Unnamed" in c or c == "_c0"]
    if cols_to_drop:
        df = df.drop(*cols_to_drop)

    # Safely cast numeric columns using try_cast to avoid CAST_INVALID_INPUT
    # errors on malformed rows (e.g., track names leaking into numeric cols)
    numeric_cols = [
        "popularity", "duration_ms", "danceability", "energy",
        "loudness", "speechiness", "acousticness", "instrumentalness",
        "liveness", "valence", "tempo"
    ]
    for col_name in numeric_cols:
        if col_name in df.columns:
            dtype = df.schema[col_name].dataType
            if isinstance(dtype, (DoubleType, FloatType)):
                # Already numeric, no cast needed
                continue
            # Use try_cast via SQL expr: returns NULL for non-castable values
            df = df.withColumn(col_name, F.expr(f"try_cast(`{col_name}` as double)"))

    # Key columns for analysis — drop rows where any key column is NULL
    key_cols = ["popularity", "danceability", "energy", "tempo", "track_genre"]
    df = df.dropna(subset=key_cols)

    # Filter out invalid popularity (keep 0-100 range)
    df = df.filter((F.col("popularity") >= 0) & (F.col("popularity") <= 100))

    final_count = df.count()
    print(f"  Records before cleaning: {initial_count:,}")
    print(f"  Records after cleaning:  {final_count:,}")
    print(f"  Records removed:         {initial_count - final_count:,}")
    print(f"  Columns retained:        {len(df.columns)}")

    return df


# ─────────────────────────────────────────────────────────────
# 3. EXPLORATORY DATA ANALYSIS (EDA) WITH SPARK
# ─────────────────────────────────────────────────────────────

def basic_statistics(df):
    """Compute and display basic statistics using Spark's describe()."""
    print("\n" + "=" * 70)
    print("  STEP 3: BASIC STATISTICAL SUMMARY (Spark Distributed Compute)")
    print("=" * 70)

    feature_cols = ["popularity", "danceability", "energy", "tempo",
                    "loudness", "valence", "speechiness", "acousticness"]

    stats_df = df.select(feature_cols).describe()
    stats_df.show(truncate=False)

    # Genre count
    genre_count = df.select("track_genre").distinct().count()
    print(f"  ✔ Total unique genres: {genre_count}")

    # Popularity distribution summary
    pop_quantiles = df.approxQuantile("popularity", [0.25, 0.50, 0.75], 0.01)
    print(f"  ✔ Popularity Quartiles — Q1: {pop_quantiles[0]}, "
          f"Median: {pop_quantiles[1]}, Q3: {pop_quantiles[2]}")

    return stats_df


# ─────────────────────────────────────────────────────────────
# 4. ANALYSIS 1 — AUDIO FEATURES vs POPULARITY (Correlation)
# ─────────────────────────────────────────────────────────────

def correlation_analysis(df):
    """
    Compute the Pearson correlation matrix between audio features and
    popularity using PySpark ML's Correlation module.
    """
    print("\n" + "=" * 70)
    print("  STEP 4: CORRELATION ANALYSIS — Audio Features vs Popularity")
    print("=" * 70)

    feature_cols = [
        "danceability", "energy", "tempo", "loudness",
        "valence", "speechiness", "acousticness",
        "instrumentalness", "liveness", "popularity"
    ]

    # Assemble features into a vector
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    vector_df = assembler.transform(df.select(feature_cols).dropna()).select("features")

    # Compute Pearson correlation matrix
    corr_matrix = Correlation.corr(vector_df, "features", method="pearson")
    corr_array = corr_matrix.collect()[0][0].toArray()

    corr_pdf = pd.DataFrame(corr_array, columns=feature_cols, index=feature_cols)

    print("\n  Correlation Matrix (Pearson):")
    print(corr_pdf.round(3).to_string())

    # ─── Correlations with popularity ────────────────────────
    pop_corr = corr_pdf["popularity"].drop("popularity").sort_values(ascending=False)
    print("\n  Correlations with Popularity (sorted):")
    for feat, val in pop_corr.items():
        bar = "█" * int(abs(val) * 40)
        sign = "+" if val > 0 else "-"
        print(f"    {feat:25s} {sign}{abs(val):.4f}  {bar}")

    # ─── Plot: Correlation Heatmap ───────────────────────────
    fig, ax = plt.subplots(figsize=(12, 9))
    mask = np.triu(np.ones_like(corr_pdf, dtype=bool), k=1)
    cmap = sns.diverging_palette(250, 130, l=65, s=90, as_cmap=True)

    sns.heatmap(
        corr_pdf, mask=mask, annot=True, fmt=".2f",
        cmap=cmap, center=0, linewidths=0.8,
        square=True, cbar_kws={"shrink": 0.8, "label": "Pearson r"},
        ax=ax
    )
    ax.set_title("Correlation Heatmap — Audio Features & Popularity",
                 fontsize=16, fontweight="bold", pad=20)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "01_correlation_heatmap.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\n  ✔ Saved: {path}")

    # ─── Plot: Bar chart of correlations with popularity ─────
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = [COLORS["primary"] if v > 0 else "#E74C3C" for v in pop_corr.values]
    bars = ax.barh(pop_corr.index, pop_corr.values, color=colors, edgecolor="white", height=0.6)

    for bar, val in zip(bars, pop_corr.values):
        offset = 0.005 if val > 0 else -0.005
        ax.text(val + offset, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=10, fontweight="bold")

    ax.set_xlabel("Pearson Correlation with Popularity", fontsize=12)
    ax.set_title("Audio Feature Correlations with Track Popularity",
                 fontsize=14, fontweight="bold", pad=15)
    ax.axvline(x=0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xlim(-0.35, 0.35)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "02_popularity_correlations_bar.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  ✔ Saved: {path}")

    return corr_pdf


# ─────────────────────────────────────────────────────────────
# 5. ANALYSIS 2 — GENRE-WISE FEATURE COMPARISON
# ─────────────────────────────────────────────────────────────

def genre_comparison(df):
    """
    Aggregate audio features by genre using Spark groupBy and compare
    genres across danceability, energy, tempo, and popularity.
    """
    print("\n" + "=" * 70)
    print("  STEP 5: GENRE-WISE COMPARISON (Spark GroupBy Aggregation)")
    print("=" * 70)

    genre_stats = df.groupBy("track_genre").agg(
        F.count("*").alias("track_count"),
        F.round(F.avg("popularity"), 2).alias("avg_popularity"),
        F.round(F.avg("danceability"), 4).alias("avg_danceability"),
        F.round(F.avg("energy"), 4).alias("avg_energy"),
        F.round(F.avg("tempo"), 2).alias("avg_tempo"),
        F.round(F.avg("valence"), 4).alias("avg_valence"),
        F.round(F.avg("loudness"), 2).alias("avg_loudness"),
        F.round(F.avg("acousticness"), 4).alias("avg_acousticness"),
        F.round(F.stddev("popularity"), 2).alias("std_popularity"),
    ).orderBy(F.desc("avg_popularity"))

    print("\n  Genre Statistics (Top 15 by Avg Popularity):")
    genre_stats.show(15, truncate=False)

    genre_pdf = genre_stats.toPandas()

    # ─── Plot 1: Top 20 Genres by Average Popularity ─────────
    top20 = genre_pdf.head(20)

    fig, ax = plt.subplots(figsize=(14, 8))
    bars = ax.barh(
        top20["track_genre"][::-1],
        top20["avg_popularity"][::-1],
        color=sns.color_palette("viridis", 20),
        edgecolor="white", height=0.7
    )
    for bar, val in zip(bars, top20["avg_popularity"][::-1]):
        ax.text(val + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}", va="center", fontsize=9, fontweight="bold")

    ax.set_xlabel("Average Popularity Score", fontsize=12)
    ax.set_title("Top 20 Genres by Average Popularity",
                 fontsize=15, fontweight="bold", pad=15)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "03_top20_genres_popularity.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  ✔ Saved: {path}")

    # ─── Plot 2: Bottom 20 Genres by Average Popularity ──────
    bottom20 = genre_pdf.tail(20)

    fig, ax = plt.subplots(figsize=(14, 8))
    bars = ax.barh(
        bottom20["track_genre"],
        bottom20["avg_popularity"],
        color=sns.color_palette("magma", 20),
        edgecolor="white", height=0.7
    )
    for bar, val in zip(bars, bottom20["avg_popularity"]):
        ax.text(val + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}", va="center", fontsize=9, fontweight="bold")

    ax.set_xlabel("Average Popularity Score", fontsize=12)
    ax.set_title("Bottom 20 Genres by Average Popularity",
                 fontsize=15, fontweight="bold", pad=15)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "04_bottom20_genres_popularity.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  ✔ Saved: {path}")

    # ─── Plot 3: Genre Feature Radar / Grouped Bar ───────────
    top10 = genre_pdf.head(10)
    features = ["avg_danceability", "avg_energy", "avg_valence", "avg_acousticness"]
    labels = ["Danceability", "Energy", "Valence", "Acousticness"]

    fig, ax = plt.subplots(figsize=(16, 8))
    x = np.arange(len(top10))
    width = 0.2

    for i, (feat, lbl) in enumerate(zip(features, labels)):
        ax.bar(x + i * width, top10[feat], width, label=lbl, edgecolor="white")

    ax.set_xlabel("Genre", fontsize=12)
    ax.set_ylabel("Average Feature Value (0-1 scale)", fontsize=12)
    ax.set_title("Audio Feature Comparison — Top 10 Popular Genres",
                 fontsize=15, fontweight="bold", pad=15)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(top10["track_genre"], rotation=40, ha="right", fontsize=10)
    ax.legend(fontsize=11, loc="upper right")
    ax.set_ylim(0, 1.0)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "05_genre_feature_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  ✔ Saved: {path}")

    # ─── Plot 4: Avg Tempo across Top 15 Genres ──────────────
    top15 = genre_pdf.head(15)

    fig, ax = plt.subplots(figsize=(12, 7))
    colors_tempo = sns.color_palette("coolwarm", 15)
    bars = ax.bar(top15["track_genre"], top15["avg_tempo"], color=colors_tempo, edgecolor="white")

    for bar, val in zip(bars, top15["avg_tempo"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.0f}", ha="center", fontsize=8, fontweight="bold")

    ax.set_xlabel("Genre", fontsize=12)
    ax.set_ylabel("Average Tempo (BPM)", fontsize=12)
    ax.set_title("Average Tempo Across Top 15 Popular Genres",
                 fontsize=14, fontweight="bold", pad=15)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "06_genre_tempo_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  ✔ Saved: {path}")

    return genre_pdf


# ─────────────────────────────────────────────────────────────
# 6. ANALYSIS 3 — POPULARITY BUCKETS & FEATURE DISTRIBUTIONS
# ─────────────────────────────────────────────────────────────

def popularity_bucket_analysis(df):
    """
    Bucket tracks into popularity tiers and analyze how audio features
    differ across popularity levels.
    """
    print("\n" + "=" * 70)
    print("  STEP 6: POPULARITY BUCKET ANALYSIS")
    print("=" * 70)

    # Create popularity buckets
    df = df.withColumn(
        "popularity_tier",
        F.when(F.col("popularity") >= 80, "★ Very High (80-100)")
         .when(F.col("popularity") >= 60, "● High (60-79)")
         .when(F.col("popularity") >= 40, "◆ Medium (40-59)")
         .when(F.col("popularity") >= 20, "▲ Low (20-39)")
         .otherwise("○ Very Low (0-19)")
    )

    tier_stats = df.groupBy("popularity_tier").agg(
        F.count("*").alias("count"),
        F.round(F.avg("danceability"), 4).alias("avg_danceability"),
        F.round(F.avg("energy"), 4).alias("avg_energy"),
        F.round(F.avg("tempo"), 2).alias("avg_tempo"),
        F.round(F.avg("valence"), 4).alias("avg_valence"),
        F.round(F.avg("loudness"), 2).alias("avg_loudness"),
        F.round(F.avg("acousticness"), 4).alias("avg_acousticness"),
    ).orderBy("popularity_tier")

    print("\n  Feature Averages by Popularity Tier:")
    tier_stats.show(truncate=False)

    tier_pdf = tier_stats.toPandas()

    # ─── Plot: Popularity Tier Feature Comparison ────────────
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle("Audio Features Distribution by Popularity Tier",
                 fontsize=17, fontweight="bold", y=1.02)

    features_to_plot = [
        ("avg_danceability", "Danceability", "viridis"),
        ("avg_energy", "Energy", "magma"),
        ("avg_tempo", "Avg Tempo (BPM)", "coolwarm"),
        ("avg_valence", "Valence (Positivity)", "cividis"),
        ("avg_loudness", "Loudness (dB)", "inferno"),
        ("avg_acousticness", "Acousticness", "plasma"),
    ]

    for ax, (col, title, cmap) in zip(axes.flat, features_to_plot):
        colors = sns.color_palette(cmap, len(tier_pdf))
        bars = ax.bar(tier_pdf["popularity_tier"], tier_pdf[col],
                      color=colors, edgecolor="white")
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.tick_params(axis="x", rotation=30, labelsize=8)

        for bar, val in zip(bars, tier_pdf[col]):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height(), f"{val:.3f}",
                    ha="center", va="bottom", fontsize=8, fontweight="bold")

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "07_popularity_tier_features.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\n  ✔ Saved: {path}")

    return tier_pdf


# ─────────────────────────────────────────────────────────────
# 7. ANALYSIS 4 — SCATTER PLOTS: FEATURES vs POPULARITY
# ─────────────────────────────────────────────────────────────

def feature_vs_popularity_scatter(df):
    """
    Create scatter plots of key audio features vs popularity.
    Uses a sampled Pandas DataFrame for visualization (sampled from Spark).
    """
    print("\n" + "=" * 70)
    print("  STEP 7: SCATTER PLOTS — Audio Features vs Popularity")
    print("=" * 70)

    # Sample 5000 rows for scatter plots (to keep them readable)
    sample_pdf = df.select(
        "danceability", "energy", "tempo", "valence",
        "loudness", "acousticness", "popularity"
    ).sample(fraction=0.05, seed=42).toPandas()

    print(f"  ✔ Sampled {len(sample_pdf):,} points for scatter visualization")

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle("Audio Features vs Track Popularity (Sampled)",
                 fontsize=17, fontweight="bold", y=1.02)

    scatter_features = [
        ("danceability", "Danceability"),
        ("energy", "Energy"),
        ("tempo", "Tempo (BPM)"),
        ("valence", "Valence"),
        ("loudness", "Loudness (dB)"),
        ("acousticness", "Acousticness"),
    ]

    for ax, (feat, label) in zip(axes.flat, scatter_features):
        ax.scatter(
            sample_pdf[feat], sample_pdf["popularity"],
            alpha=0.15, s=8, c=sample_pdf["popularity"],
            cmap="viridis", edgecolors="none"
        )
        # Add trend line
        z = np.polyfit(sample_pdf[feat].dropna(), sample_pdf.loc[sample_pdf[feat].notna(), "popularity"], 1)
        p = np.poly1d(z)
        x_range = np.linspace(sample_pdf[feat].min(), sample_pdf[feat].max(), 100)
        ax.plot(x_range, p(x_range), color="#E74C3C", linewidth=2, linestyle="--", label="Trend")

        ax.set_xlabel(label, fontsize=11)
        ax.set_ylabel("Popularity", fontsize=11)
        ax.set_title(f"{label} vs Popularity", fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "08_feature_vs_popularity_scatter.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  ✔ Saved: {path}")


# ─────────────────────────────────────────────────────────────
# 8. ANALYSIS 5 — GENRE HEATMAP (Features × Genres)
# ─────────────────────────────────────────────────────────────

def genre_feature_heatmap(df):
    """
    Create a heatmap showing average audio features for the top 25 genres.
    """
    print("\n" + "=" * 70)
    print("  STEP 8: GENRE-FEATURE HEATMAP")
    print("=" * 70)

    features = ["danceability", "energy", "tempo", "valence",
                "loudness", "acousticness", "speechiness", "instrumentalness"]

    agg_exprs = [F.round(F.avg(f), 4).alias(f) for f in features]
    agg_exprs.append(F.round(F.avg("popularity"), 2).alias("popularity"))

    genre_feat = df.groupBy("track_genre").agg(*agg_exprs)

    # Get top 25 genres by popularity
    top25 = genre_feat.orderBy(F.desc("popularity")).limit(25).toPandas()
    top25 = top25.set_index("track_genre")

    # Normalize tempo to 0-1 scale for heatmap
    top25_norm = top25.copy()
    top25_norm["tempo"] = (top25_norm["tempo"] - top25_norm["tempo"].min()) / \
                          (top25_norm["tempo"].max() - top25_norm["tempo"].min())
    top25_norm["loudness"] = (top25_norm["loudness"] - top25_norm["loudness"].min()) / \
                              (top25_norm["loudness"].max() - top25_norm["loudness"].min())
    top25_norm["popularity"] = top25_norm["popularity"] / 100.0

    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(
        top25_norm, annot=True, fmt=".2f", cmap="YlGnBu",
        linewidths=0.5, ax=ax, cbar_kws={"label": "Normalized Value"}
    )
    ax.set_title("Audio Feature Heatmap — Top 25 Genres (Normalized)",
                 fontsize=15, fontweight="bold", pad=15)
    ax.set_ylabel("Genre", fontsize=12)
    ax.set_xlabel("Audio Feature", fontsize=12)
    plt.xticks(rotation=30, ha="right")
    plt.yticks(fontsize=9)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "09_genre_feature_heatmap.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  ✔ Saved: {path}")


# ─────────────────────────────────────────────────────────────
# 9. ANALYSIS 6 — BOX PLOTS: FEATURE DISTRIBUTIONS BY GENRE
# ─────────────────────────────────────────────────────────────

def genre_boxplots(df):
    """
    Create box plots showing the distribution of key features across
    the top 10 most popular genres.
    """
    print("\n" + "=" * 70)
    print("  STEP 9: BOX PLOTS — Feature Distributions by Genre")
    print("=" * 70)

    # Get top 10 genres
    top10_genres = [row["track_genre"] for row in
                    df.groupBy("track_genre").agg(F.avg("popularity").alias("avg_pop"))
                      .orderBy(F.desc("avg_pop")).limit(10).collect()]

    subset = df.filter(F.col("track_genre").isin(top10_genres))
    subset_pdf = subset.select("track_genre", "danceability", "energy", "tempo", "valence").toPandas()

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Feature Distributions — Top 10 Popular Genres",
                 fontsize=16, fontweight="bold", y=1.02)

    box_features = [
        ("danceability", "Danceability"),
        ("energy", "Energy"),
        ("tempo", "Tempo (BPM)"),
        ("valence", "Valence"),
    ]

    for ax, (feat, label) in zip(axes.flat, box_features):
        sns.boxplot(
            data=subset_pdf, x="track_genre", y=feat,
            ax=ax, palette="Set2", fliersize=2
        )
        ax.set_title(f"{label} Distribution", fontsize=13, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel(label, fontsize=11)
        ax.tick_params(axis="x", rotation=45, labelsize=9)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "10_genre_boxplots.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  ✔ Saved: {path}")


# ─────────────────────────────────────────────────────────────
# 10. ANALYSIS 7 — SPARK SQL QUERIES
# ─────────────────────────────────────────────────────────────

def spark_sql_analysis(df, spark):
    """
    Register the DataFrame as a SQL temp view and run analytical queries
    using Spark SQL — demonstrating big data SQL capabilities.
    """
    print("\n" + "=" * 70)
    print("  STEP 10: SPARK SQL ANALYTICAL QUERIES")
    print("=" * 70)

    df.createOrReplaceTempView("spotify_tracks")

    # Query 1: Top genres with high danceability AND high popularity
    print("\n  -- Query 1: Genres with high avg danceability AND popularity --")
    q1 = spark.sql("""
        SELECT track_genre,
               COUNT(*) AS track_count,
               ROUND(AVG(popularity), 2) AS avg_popularity,
               ROUND(AVG(danceability), 4) AS avg_danceability,
               ROUND(AVG(energy), 4) AS avg_energy
        FROM spotify_tracks
        GROUP BY track_genre
        HAVING AVG(popularity) > 50 AND AVG(danceability) > 0.6
        ORDER BY avg_popularity DESC
        LIMIT 15
    """)
    q1.show(truncate=False)

    # Query 2: Tracks with extreme features (high energy + low acousticness)
    print("  -- Query 2: High-energy, low-acoustic tracks with popularity > 70 --")
    q2 = spark.sql("""
        SELECT track_name, artists, track_genre, popularity,
               ROUND(energy, 3) AS energy,
               ROUND(acousticness, 3) AS acousticness,
               ROUND(danceability, 3) AS danceability
        FROM spotify_tracks
        WHERE energy > 0.9
          AND acousticness < 0.05
          AND popularity > 70
        ORDER BY popularity DESC
        LIMIT 10
    """)
    q2.show(truncate=False)

    # Query 3: Average features for top 5 most popular genres
    print("  -- Query 3: Complete feature profile of Top 5 genres --")
    q3 = spark.sql("""
        SELECT track_genre,
               COUNT(*) AS tracks,
               ROUND(AVG(popularity), 1) AS avg_popularity,
               ROUND(AVG(danceability), 3) AS dance,
               ROUND(AVG(energy), 3) AS energy,
               ROUND(AVG(tempo), 1) AS tempo,
               ROUND(AVG(valence), 3) AS valence,
               ROUND(AVG(loudness), 1) AS loudness,
               ROUND(AVG(acousticness), 3) AS acoustic
        FROM spotify_tracks
        GROUP BY track_genre
        ORDER BY avg_popularity DESC
        LIMIT 5
    """)
    q3.show(truncate=False)

    # Query 4: Genre ranking by popularity using subquery with DENSE_RANK
    print("  -- Query 4: Genre ranking by popularity with window functions --")
    q4 = spark.sql("""
        SELECT track_genre, avg_pop, dance, energy,
               DENSE_RANK() OVER (ORDER BY avg_pop DESC) AS rank
        FROM (
            SELECT track_genre,
                   ROUND(AVG(popularity), 2) AS avg_pop,
                   ROUND(AVG(danceability), 3) AS dance,
                   ROUND(AVG(energy), 3) AS energy
            FROM spotify_tracks
            GROUP BY track_genre
        )
        ORDER BY rank
        LIMIT 10
    """)
    q4.show(truncate=False)


# ─────────────────────────────────────────────────────────────
# 11. SUMMARY & KEY FINDINGS
# ─────────────────────────────────────────────────────────────

def print_summary(corr_pdf, genre_pdf, tier_pdf):
    """Print a summary of key findings from the analysis."""
    print("\n" + "=" * 70)
    print("  SUMMARY OF KEY FINDINGS")
    print("=" * 70)

    # Top correlating features
    pop_corr = corr_pdf["popularity"].drop("popularity").sort_values(ascending=False)
    top_pos = pop_corr.head(3)
    top_neg = pop_corr.tail(3)

    print("\n  📊 Top Features POSITIVELY Correlated with Popularity:")
    for feat, val in top_pos.items():
        print(f"      • {feat}: r = {val:+.4f}")

    print("\n  📊 Top Features NEGATIVELY Correlated with Popularity:")
    for feat, val in top_neg.items():
        print(f"      • {feat}: r = {val:+.4f}")

    print(f"\n  🎵 Most Popular Genre:  {genre_pdf.iloc[0]['track_genre']} "
          f"(avg popularity: {genre_pdf.iloc[0]['avg_popularity']})")
    print(f"  🎵 Least Popular Genre: {genre_pdf.iloc[-1]['track_genre']} "
          f"(avg popularity: {genre_pdf.iloc[-1]['avg_popularity']})")

    # Danceability insight
    most_danceable = genre_pdf.sort_values("avg_danceability", ascending=False).iloc[0]
    print(f"\n  💃 Most Danceable Genre: {most_danceable['track_genre']} "
          f"(avg danceability: {most_danceable['avg_danceability']})")

    most_energetic = genre_pdf.sort_values("avg_energy", ascending=False).iloc[0]
    print(f"  ⚡ Most Energetic Genre: {most_energetic['track_genre']} "
          f"(avg energy: {most_energetic['avg_energy']})")

    fastest = genre_pdf.sort_values("avg_tempo", ascending=False).iloc[0]
    print(f"  🏎️  Fastest Tempo Genre:  {fastest['track_genre']} "
          f"(avg tempo: {fastest['avg_tempo']} BPM)")

    print("\n  📈 Popularity Tier Insights:")
    for _, row in tier_pdf.iterrows():
        print(f"      {row['popularity_tier']}: "
              f"dance={row['avg_danceability']:.3f}, "
              f"energy={row['avg_energy']:.3f}, "
              f"tempo={row['avg_tempo']:.1f}")

    print("\n" + "=" * 70)
    print(f"  ✅ All charts saved to: {OUTPUT_DIR}")
    print("=" * 70)


# ─────────────────────────────────────────────────────────────
# 12. MAIN EXECUTION
# ─────────────────────────────────────────────────────────────

def main():
    """Main entry point for the Spotify Big Data Analysis project."""
    print("\n" + "█" * 70)
    print("  🎵 SPOTIFY TRACKS — BIG DATA ANALYSIS WITH PYSPARK 🎵")
    print("█" * 70)

    # Path to dataset
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "spotify-tracks-dataset.csv")

    if not os.path.exists(csv_path):
        print(f"\n  ❌ Dataset not found at: {csv_path}")
        sys.exit(1)

    # Initialize Spark
    spark = create_spark_session()
    print(f"\n  ✔ Spark Version: {spark.version}")
    print(f"  ✔ App Name: {spark.sparkContext.appName}")
    print(f"  ✔ Master: {spark.sparkContext.master}")

    try:
        # Load and clean data
        df = load_data(spark, csv_path)
        df = clean_data(df)

        # Cache the DataFrame for repeated use
        df.cache()
        print(f"\n  ✔ DataFrame cached in memory for fast repeated access")

        # Run all analyses
        stats = basic_statistics(df)
        corr_pdf = correlation_analysis(df)
        genre_pdf = genre_comparison(df)
        tier_pdf = popularity_bucket_analysis(df)
        feature_vs_popularity_scatter(df)
        genre_feature_heatmap(df)
        genre_boxplots(df)
        spark_sql_analysis(df, spark)

        # Print summary
        print_summary(corr_pdf, genre_pdf, tier_pdf)

    except Exception as e:
        print(f"\n  ❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Stop Spark session
        spark.stop()
        print("\n  ✔ Spark session stopped. Analysis complete.\n")


if __name__ == "__main__":
    main()
