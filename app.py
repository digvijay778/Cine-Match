import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CineMatch — Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=Inter:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Dark cinema-inspired header */
.cinema-header {
    background: #0d0d0d;
    padding: 2.5rem 2rem 2rem;
    margin: -1rem -1rem 2rem -1rem;
    border-bottom: 3px solid #e63946;
}
.cinema-header h1 {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 2.8rem;
    color: #ffffff;
    margin: 0;
    letter-spacing: -0.02em;
}
.cinema-header .tagline {
    color: #999;
    font-size: 0.95rem;
    margin-top: 0.4rem;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}
.cinema-header .red { color: #e63946; }

/* Movie card grid */
.movie-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
    gap: 1rem;
    margin-top: 1rem;
}
.movie-card {
    background: #fff;
    border: 1px solid #e8e8e8;
    border-radius: 10px;
    padding: 1.2rem 1.4rem;
    position: relative;
    transition: box-shadow 0.15s;
}
.movie-card:hover { box-shadow: 0 4px 20px rgba(0,0,0,0.1); }
.movie-rank {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 800;
    color: #f0f0f0;
    position: absolute;
    top: 0.8rem;
    right: 1rem;
    line-height: 1;
}
.movie-title {
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 1rem;
    color: #0d0d0d;
    margin-bottom: 0.5rem;
    padding-right: 2rem;
    line-height: 1.3;
}
.movie-genres {
    display: flex;
    flex-wrap: wrap;
    gap: 4px;
    margin-bottom: 0.8rem;
}
.genre-tag {
    background: #f3f3f3;
    color: #555;
    font-size: 0.7rem;
    padding: 2px 8px;
    border-radius: 999px;
    letter-spacing: 0.03em;
}
.pred-score {
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.score-bar-bg {
    flex: 1;
    background: #f0f0f0;
    border-radius: 999px;
    height: 6px;
    overflow: hidden;
}
.score-bar-fill {
    height: 100%;
    border-radius: 999px;
    background: linear-gradient(90deg, #e63946, #ff6b6b);
}
.score-val {
    font-family: 'Syne', sans-serif;
    font-size: 0.85rem;
    font-weight: 700;
    color: #e63946;
    min-width: 2.5rem;
    text-align: right;
}

/* History ratings */
.hist-card {
    background: #fafafa;
    border: 1px solid #ebebeb;
    border-radius: 8px;
    padding: 0.8rem 1rem;
    margin-bottom: 0.5rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.hist-title { font-size: 0.88rem; color: #333; font-weight: 500; }
.hist-rating { font-family: 'Syne', sans-serif; font-weight: 700; color: #e63946; }

/* Stat pills */
.stat-row { display: flex; gap: 0.75rem; flex-wrap: wrap; margin: 1rem 0; }
.stat-pill {
    background: #0d0d0d;
    color: #fff;
    border-radius: 8px;
    padding: 0.6rem 1rem;
    font-size: 0.8rem;
    text-align: center;
}
.stat-pill .num {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 1.3rem;
    display: block;
    color: #e63946;
}

.divider { border: none; border-top: 1px solid #ececec; margin: 1.5rem 0; }
</style>
""", unsafe_allow_html=True)

# ── Load artifacts ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    base = "model_artifacts"
    paths = {
        "svd":        os.path.join(base, "svd_model.pkl"),
        "matrix":     os.path.join(base, "user_item_matrix.pkl"),
        "similarity": os.path.join(base, "user_similarity_df.pkl"),
        "ratings":    os.path.join(base, "ratings.csv"),
        "movies":     os.path.join(base, "movies.csv"),
    }
    missing = [k for k, v in paths.items() if not os.path.exists(v)]
    if missing:
        return None, None, None, None, None, missing

    with open(paths["svd"], "rb") as f:
        svd_model = pickle.load(f)
    with open(paths["matrix"], "rb") as f:
        user_item_matrix = pickle.load(f)
    with open(paths["similarity"], "rb") as f:
        user_similarity_df = pickle.load(f)
    ratings = pd.read_csv(paths["ratings"])
    movies  = pd.read_csv(paths["movies"])
    return svd_model, user_item_matrix, user_similarity_df, ratings, movies, []

svd_model, user_item_matrix, user_similarity_df, ratings, movies, missing_files = load_artifacts()

# ── Recommendation function ───────────────────────────────────────────────────
def recommend_movies_svd(user_id: int, n_recommendations: int = 10) -> pd.DataFrame:
    if svd_model is None:
        return pd.DataFrame()
    # movies already rated by user
    rated_movie_ids = ratings[ratings["userId"] == user_id]["movieId"].values
    all_movie_ids   = movies["movieId"].values
    unrated         = [m for m in all_movie_ids if m not in rated_movie_ids]

    preds = [(mid, svd_model.predict(user_id, mid).est) for mid in unrated]
    top   = sorted(preds, key=lambda x: x[1], reverse=True)[:n_recommendations]

    df = pd.DataFrame(top, columns=["movieId", "predicted_rating"])
    df = df.merge(movies[["movieId", "title", "genres"]], on="movieId", how="left")
    return df[["title", "genres", "predicted_rating"]]

def get_user_history(user_id: int, n: int = 8) -> pd.DataFrame:
    user_ratings = ratings[ratings["userId"] == user_id].copy()
    user_ratings = user_ratings.merge(movies[["movieId", "title", "genres"]], on="movieId", how="left")
    return user_ratings.sort_values("rating", ascending=False).head(n)[["title", "genres", "rating"]]

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="cinema-header">
    <h1>Cine<span class="red">Match</span></h1>
    <div class="tagline">SVD Collaborative Filtering · MovieLens 100K</div>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")

    if missing_files:
        st.error(f"Missing files: `{'`, `'.join(missing_files)}`\n\nAdd them to the `model_artifacts/` folder.")
        st.stop()

    valid_users = sorted(ratings["userId"].unique().tolist())
    user_id = st.selectbox(
        "Select User ID",
        valid_users,
        index=0,
        help="610 users from the MovieLens dataset"
    )
    n_recs = st.slider("Number of recommendations", 5, 20, 10)

    st.divider()
    st.markdown("### 📊 Model Performance")
    st.markdown("""
    | Model | RMSE | MAE |
    |---|---|---|
    | **SVD ✅** | **0.8807** | **0.6766** |
    | KNN | 0.9847 | 0.7557 |
    """)
    st.caption("5-fold CV RMSE: 0.8745 ± 0.0078")

    st.divider()
    st.markdown("### 🔗 Resources")
    st.markdown("[📦 MovieLens Dataset](https://www.kaggle.com/datasets/sriharshabsprasad/movielens-dataset-100k-ratings)")

# ── Stats ──────────────────────────────────────────────────────────────────────
total_ratings = len(ratings)
total_users   = ratings["userId"].nunique()
total_movies  = movies["movieId"].nunique()
user_count    = len(ratings[ratings["userId"] == user_id])

st.markdown(f"""
<div class="stat-row">
    <div class="stat-pill"><span class="num">{total_ratings:,}</span>ratings</div>
    <div class="stat-pill"><span class="num">{total_users}</span>users</div>
    <div class="stat-pill"><span class="num">{total_movies:,}</span>movies</div>
    <div class="stat-pill"><span class="num">{user_count}</span>ratings by User {user_id}</div>
</div>
""", unsafe_allow_html=True)

# ── Main content ───────────────────────────────────────────────────────────────
col_recs, col_hist = st.columns([3, 2], gap="large")

with col_recs:
    st.markdown(f"### 🎬 Top {n_recs} picks for User {user_id}")

    with st.spinner("Generating recommendations…"):
        recs = recommend_movies_svd(user_id, n_recs)

    if recs.empty:
        st.warning("No recommendations found for this user.")
    else:
        cards_html = '<div class="movie-grid">'
        for i, row in recs.iterrows():
            rank      = i + 1
            score     = row["predicted_rating"]
            bar_pct   = min(100, (score / 5.0) * 100)
            genres    = row["genres"].split("|") if pd.notna(row["genres"]) else []
            genre_tags = "".join(f'<span class="genre-tag">{g}</span>' for g in genres[:4])
            title_safe = str(row["title"]).replace("<", "&lt;").replace(">", "&gt;")

            cards_html += f"""
            <div class="movie-card">
                <div class="movie-rank">{rank:02d}</div>
                <div class="movie-title">{title_safe}</div>
                <div class="movie-genres">{genre_tags}</div>
                <div class="pred-score">
                    <div class="score-bar-bg">
                        <div class="score-bar-fill" style="width:{bar_pct:.1f}%"></div>
                    </div>
                    <span class="score-val">{score:.2f}</span>
                </div>
            </div>"""
        cards_html += "</div>"
        st.markdown(cards_html, unsafe_allow_html=True)

with col_hist:
    st.markdown(f"### ⭐ User {user_id}'s top-rated movies")
    history = get_user_history(user_id)

    if history.empty:
        st.info("No rating history found.")
    else:
        for _, row in history.iterrows():
            stars = "★" * int(row["rating"]) + "☆" * (5 - int(row["rating"]))
            title_safe = str(row["title"]).replace("<", "&lt;").replace(">", "&gt;")
            genres = row["genres"].split("|")[:2] if pd.notna(row["genres"]) else []
            genre_str = " · ".join(genres)
            st.markdown(f"""
            <div class="hist-card">
                <div>
                    <div class="hist-title">{title_safe}</div>
                    <div style="font-size:0.75rem;color:#aaa;margin-top:2px">{genre_str}</div>
                </div>
                <div>
                    <div class="hist-rating">{row['rating']:.1f}</div>
                    <div style="font-size:0.65rem;color:#ccc;text-align:center">{stars}</div>
                </div>
            </div>""", unsafe_allow_html=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # Genre breakdown for this user
    st.markdown("**Genre breakdown**")
    user_movies = ratings[ratings["userId"] == user_id].merge(movies, on="movieId")
    all_genres  = user_movies["genres"].dropna().str.split("|").explode()
    genre_counts = all_genres.value_counts().head(8)

    genre_df = pd.DataFrame({"Genre": genre_counts.index, "Count": genre_counts.values})
    st.bar_chart(genre_df.set_index("Genre"), color="#e63946", height=220)
