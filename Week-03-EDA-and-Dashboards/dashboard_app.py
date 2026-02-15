import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="MovieLens Dashboard", layout="wide")

# ---------- Data loaders ----------
@st.cache_data
def load_data():
    df = pd.read_csv("Week-03-EDA-and-Dashboards/data/movie_ratings.csv")
    # tidy types
    if "year" in df.columns:
        df["year"] = pd.to_numeric(df["year"], errors="coerce")
    if "rating_year" in df.columns:
        df["rating_year"] = pd.to_numeric(df["rating_year"], errors="coerce")
    # explode genres for per-genre analysis
    df_genres = df.assign(genre=df["genres"].str.split("|")).explode("genre")
    return df, df_genres

df, df_genres = load_data()

st.title("🎬 MovieLens Dashboard")
st.caption("Interactive analysis of MovieLens 200k ratings • Notes: Genres are exploded for preference profiling; use sample thresholds to avoid small-sample noise.")

# ---------- Sidebar filters ----------
with st.sidebar:
    st.header("Filters")
    # Genre selector
    all_genres = sorted([g for g in df_genres["genre"].dropna().unique()])
    picked_genres = st.multiselect("Genres", all_genres, default=all_genres)

    # Age range
    age_min, age_max = int(df["age"].min()), int(df["age"].max())
    age_range = st.slider("Age range", min_value=age_min, max_value=age_max, value=(age_min, age_max))

    # Occupation
    all_occ = sorted(df["occupation"].dropna().unique())
    picked_occ = st.multiselect("Occupations", all_occ, default=all_occ)

    # Minimum #ratings threshold (used in Q2/Q4)
    min_n = st.slider("Minimum ratings (threshold)", 1, 500, 50, step=10)

# apply filters
f = (
    (df_genres["genre"].isin(picked_genres)) &
    (df_genres["age"].between(age_range[0], age_range[1])) &
    (df_genres["occupation"].isin(picked_occ))
)
dfg = df_genres.loc[f].copy()

# Small helper to show counts next to charts
def show_count_note(n_rows, label="ratings"):
    st.caption(f"Records after filters: **{n_rows:,} {label}**")

# =====================================================================
# Q1 — Breakdown of genres for the movies that were rated
# =====================================================================
st.subheader("Q1. Breakdown of Ratings by Genre")
q1_counts = dfg["genre"].value_counts().sort_values(ascending=False)
col1, col2 = st.columns([2,1])
with col1:
    st.bar_chart(q1_counts)
with col2:
    st.dataframe(q1_counts.rename("count"))
show_count_note(len(dfg))

st.caption("Note: Genres are exploded (a movie with multiple genres contributes a rating to each listed genre).")

# =====================================================================
# Q2 — Which genres have the highest viewer satisfaction?
# =====================================================================
st.subheader("Q2. Highest Rated Genres (with sample threshold)")
q2_stats = (
    dfg.groupby("genre")
       .agg(mean_rating=("rating", "mean"), num_ratings=("rating", "count"))
       .query("num_ratings >= @min_n")
       .sort_values(["mean_rating","num_ratings"], ascending=[False, False])
)

c1, c2 = st.columns([2,1])
with c1:
    st.bar_chart(q2_stats["mean_rating"])
with c2:
    st.dataframe(q2_stats)

show_count_note(int(q2_stats["num_ratings"].sum()), "ratings (genres passing threshold)")
st.caption(f"Minimum ratings per genre = **{min_n}** to avoid small-sample noise.")

# =====================================================================
# Q3 — How does mean rating change across movie release years?
# =====================================================================
st.subheader("Q3. Mean Rating Across Movie Release Years")
q3 = (
    dfg.dropna(subset=["year"])
       .groupby("year")
       .agg(mean_rating=("rating","mean"), n=("rating","count"))
       .sort_index()
)

c3, c4 = st.columns([2,1])
with c3:
    st.line_chart(q3["mean_rating"])
with c4:
    st.dataframe(q3)

show_count_note(int(q3["n"].sum()), "ratings (yearly series)")
st.caption("Include counts to show uneven distribution across years; small years can look noisy.")

# =====================================================================
# Q4 — Top 5 best-rated movies with ≥50 and ≥150 ratings
# =====================================================================
st.subheader("Q4. Top-Rated Movies by Minimum Rating Count")
# Work from non-exploded df for movie-level stats
f_base = (
    (df["age"].between(age_range[0], age_range[1])) &
    (df["occupation"].isin(picked_occ))
)
df_base = df.loc[f_base].copy()

movie_stats = (
    df_base.groupby("title")
           .agg(mean_rating=("rating","mean"), num_ratings=("rating","count"))
           .sort_values(["mean_rating","num_ratings"], ascending=[False, False])
)

def top_with_threshold(threshold):
    return movie_stats.query("num_ratings >= @threshold").head(5)

tabs = st.tabs(["≥ 50 ratings", "≥ 150 ratings", f"≥ {min_n} (sidebar)"])
with tabs[0]:
    st.dataframe(top_with_threshold(50))
with tabs[1]:
    st.dataframe(top_with_threshold(150))
with tabs[2]:
    st.dataframe(top_with_threshold(min_n))

show_count_note(len(df_base), "ratings (movie-level)")

# =====================================================================
# Insights / notes box
# =====================================================================
st.markdown("### Quick Insights")
st.write(
    """
- **Genre mix**: The bar sizes in Q1 reflect rating *volume* per genre after filters.
- **Quality vs. volume**: Q2 applies your minimum-sample threshold to prevent tiny genres from topping the chart.
- **By release year**: Q3 shows mean ratings across release years; consult the counts column to gauge stability.
- **Top movies**: Q4 uses movie-level stats (not exploded). Use the thresholds (50/150/sidebar) to avoid tiny-sample bias.
"""
)
