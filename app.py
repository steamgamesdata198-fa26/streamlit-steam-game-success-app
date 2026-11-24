
"""
Steam Game Suggester - Streamlit app

Enter the name, tags, description, and price of a game
you have played to discover 10 similar Steam games.
"""

import ast
import re
import json
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer


# =====================================================
# 1. CONFIGURATION
# =====================================================

from pathlib import Path

COL_ID = "AppID"
COL_NAME = "Name_x"
COL_TAGS = "tags"
COL_DESC = "short_description"
COL_PRICE = "price_usd"
COL_EMB = "description embeddings"
COL_OWNERS = "Estimated owners"

BASE_DIR = Path(__file__).resolve().parent

DEFAULT_DATA_PATH = BASE_DIR / "data" / "merged.csv"
MODEL_PARAMS_PATH = BASE_DIR / "data" / "model_params.json"

st.set_page_config(
    page_title="Steam Game Suggester",
    page_icon="🎮",
    layout="wide"
)


# =====================================================
# 2. DATA LOADING AND CLEANING
# =====================================================

def parse_tags(value) -> list[str]:
    """
    Handles:
    'Action, Adventure, RPG'
    ['Action', 'Adventure', 'RPG']
    {'Action': 12, 'Adventure': 5}
    """

    if isinstance(value, dict):
        return [
            str(t).strip()
            for t in value.keys()
            if str(t).strip()
        ]

    if isinstance(value, (list, tuple, set)):
        return [
            str(t).strip()
            for t in value
            if str(t).strip()
        ]

    if pd.isna(value):
        return []

    s = str(value).strip()

    if s.startswith(("{", "[")):
        try:
            parsed = ast.literal_eval(s)

            items = (
                parsed.keys()
                if isinstance(parsed, dict)
                else parsed
            )

            return [
                str(t).strip()
                for t in items
                if str(t).strip()
            ]

        except (ValueError, SyntaxError):
            pass

    return [
        t.strip()
        for t in re.split(r"[,;|]", s)
        if t.strip()
    ]


@st.cache_data(show_spinner="Cleaning data...")
def clean(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    # Accommodate the column name used in Leila's model
    if COL_NAME not in df.columns and "name" in df.columns:
        df[COL_NAME] = df["name"]

    missing = [
        c for c in [
            COL_NAME, COL_TAGS, COL_DESC, COL_PRICE
        ]
        if c not in df.columns
    ]

    if missing:
        raise KeyError(f"Missing columns: {missing}")

    if COL_ID not in df.columns:
        df[COL_ID] = np.arange(len(df))

    df[COL_NAME] = df[COL_NAME].fillna("").astype(str)

    df[COL_DESC] = df[COL_DESC].fillna("").astype(str)

    df[COL_PRICE] = pd.to_numeric(
        df[COL_PRICE],
        errors="coerce"
    ).fillna(0.0).clip(lower=0)

    df["tag_list"] = df[COL_TAGS].apply(parse_tags)

    df["tag_text"] = df["tag_list"].apply(
        lambda tags: "|".join(t.lower() for t in tags)
    )

    df = df[df[COL_NAME].str.strip() != ""]

    df = df.drop_duplicates(
        subset=COL_NAME
    ).reset_index(drop=True)

    keep = [
        COL_ID,
        COL_NAME,
        COL_TAGS,
        COL_DESC,
        COL_PRICE,
        "tag_list",
        "tag_text"
    ]

    keep += [
        c for c in (COL_EMB, COL_OWNERS)
        if c in df.columns
    ]

    return df[keep]


def parse_embeddings(series: pd.Series):
    """
    Converts saved embedding strings into a normalized
    matrix of shape (n_games, embedding_dimension).
    """

    try:
        E = np.vstack([
            np.array(
                str(v).strip().strip("[]")
                .replace(",", " ").split(),
                dtype=float
            )
            for v in series
        ])

    except (ValueError, TypeError):
        return None

    if not np.isfinite(E).all():
        return None

    norms = np.linalg.norm(E, axis=1, keepdims=True)

    return E / np.where(norms == 0, 1, norms)


@st.cache_data(show_spinner="Loading Steam games...")
def load_data():
    df = pd.read_csv(DEFAULT_DATA_PATH)
    return clean(df)


@st.cache_data
def load_model_params():
    with open(MODEL_PARAMS_PATH, "r") as f:
        return json.load(f)


# =====================================================
# 3. PREPARE THE RECOMMENDATION MODEL
# =====================================================

@st.cache_resource(show_spinner="Loading MiniLM model...")
def load_sentence_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_resource(show_spinner="Preparing game features...")
def prepare_features(df):

    # ---------- Tags ----------

    tag_lists = [
        [t.lower() for t in tags]
        for tags in df["tag_list"]
    ]

    tag_encoder = MultiLabelBinarizer(
        sparse_output=True
    )

    tag_matrix = tag_encoder.fit_transform(tag_lists)


    # ---------- Name ----------

    name_vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 4)
    )

    name_matrix = name_vectorizer.fit_transform(
        df[COL_NAME]
    )


    # ---------- Description ----------

    sentence_model = load_sentence_model()

    desc_matrix = None

    if COL_EMB in df.columns:
        desc_matrix = parse_embeddings(df[COL_EMB])

    # Generate embeddings if saved embeddings are missing
    # or could not be parsed.

    if (
        desc_matrix is None
        or desc_matrix.shape[1]
        != sentence_model.get_sentence_embedding_dimension()
    ):
        desc_matrix = sentence_model.encode(
            df[COL_DESC].tolist(),
            normalize_embeddings=True,
            show_progress_bar=False
        )


    # ---------- Price ----------

    raw_price = df[COL_PRICE].to_numpy()

    log_price = np.log1p(raw_price)


    return {
        "tag_encoder": tag_encoder,
        "tag_matrix": tag_matrix,
        "name_vectorizer": name_vectorizer,
        "name_matrix": name_matrix,
        "desc_matrix": desc_matrix,
        "log_price": log_price
    }


# =====================================================
# 4. RECOMMENDATION FUNCTION
# =====================================================

def recommend_games(
    game_name,
    game_price,
    game_description,
    game_tags,
    df,
    features,
    params,
    k=10
):

    # ---------- Tag similarity ----------

    user_tags = [
        t.lower()
        for t in parse_tags(game_tags)
    ]

    # Only use tags that exist in the dataset's vocabulary
    known_tags = set(features["tag_encoder"].classes_)

    user_tags = [
        t for t in user_tags
        if t in known_tags
    ]

    user_tag_vector = features["tag_encoder"].transform(
        [user_tags]
    )

    tag_scores = cosine_similarity(
        user_tag_vector,
        features["tag_matrix"]
    ).flatten()


    # ---------- Name similarity ----------

    user_name_vector = features["name_vectorizer"].transform(
        [game_name]
    )

    name_scores = cosine_similarity(
        user_name_vector,
        features["name_matrix"]
    ).flatten()


    # ---------- Description similarity ----------

    sentence_model = load_sentence_model()

    user_desc_vector = sentence_model.encode(
        [game_description],
        normalize_embeddings=True
    )

    desc_scores = cosine_similarity(
        user_desc_vector,
        features["desc_matrix"]
    ).flatten()


    # ---------- Price similarity ----------

    user_log_price = np.log1p(game_price)

    price_std = max(params["price_std"], 1e-8)

    price_scores = np.exp(
        -np.abs(
            features["log_price"] - user_log_price
        ) / price_std
    )


    # ---------- Combine similarities ----------

    S = {
        "Tags": tag_scores,
        "Desc": desc_scores,
        "Name": name_scores,
        "Price": price_scores
    }


    # ---------- Standardize similarities ----------

    # Use the same means and standard deviations
    # from Leila's original model.

    Z = {}

    for feature in S:

        mean = params["means"][feature]

        std = max(
            params["stds"][feature],
            1e-8
        )

        Z[feature] = (S[feature] - mean) / std


    # ---------- Weighted hybrid ----------

    weights = params["weights"]

    final_scores = sum(
        weights[feature] * Z[feature]
        for feature in Z
    )


    # ---------- Create recommendation table ----------

    results = df.copy()

    results["similarity_score"] = final_scores

    results["tag_similarity"] = tag_scores

    results["description_similarity"] = desc_scores

    results["name_similarity"] = name_scores

    results["price_similarity"] = price_scores


    # Do not recommend the game the user already played.

    results = results[
        results[COL_NAME].str.strip().str.casefold()
        != game_name.strip().casefold()
    ]


    # Return the 10 highest-scoring games.

    results = results.sort_values(
        "similarity_score",
        ascending=False
    ).head(k).reset_index(drop=True)

    return results


# =====================================================
# 5. LOAD DATA AND MODEL
# =====================================================

try:
    df = load_data()
    params = load_model_params()
    features = prepare_features(df)

except Exception as e:
    st.error(f"Unable to load the recommendation model: {e}")
    st.stop()


# =====================================================
# 6. STREAMLIT USER INTERFACE
# =====================================================

st.title("🎮 Steam Game Suggester")

st.write(
    "Looking for your next favorite game? "
    "Enter information about a game you've enjoyed playing, "
    "and our recommendation model will find 10 similar "
    "games from Steam."
)

st.caption(
    f"Search across {len(df):,} games in our Steam dataset."
)

st.divider()

st.subheader("Tell us about a game you've played")


# ---------- User input form ----------

with st.form("game_input"):

    col1, col2 = st.columns([3, 1])

    with col1:

        game_name = st.text_input(
            "Game Name",
            placeholder="Stardew Valley",
            help="Enter the full name of the game."
        )

        st.caption("Example: Stardew Valley")

    with col2:

        game_price = st.number_input(
            "Game Price ($ USD)",
            min_value=0.0,
            value=0.0,
            step=0.01,
            format="%.2f"
        )

        st.caption("Example: 14.99; enter 0 for free games.")


    game_description = st.text_area(
        "Short Description",
        placeholder=(
            "A relaxing farming simulation game where "
            "players grow crops, raise animals, explore "
            "caves, and build relationships with villagers."
        ),
        height=130,
        help="Describe the gameplay, setting, and objectives."
    )

    st.caption(
        "Example: A farming simulation game where players "
        "grow crops, explore caves, and build relationships "
        "with villagers."
    )


    game_tags = st.text_area(
        "Game Tags",
        placeholder=(
            "Farming Sim, Relaxing, RPG, "
            "Pixel Graphics, Multiplayer"
        ),
        height=90,
        help="Separate each tag with a comma."
    )

    st.caption(
        "Example: Farming Sim, Relaxing, RPG, "
        "Pixel Graphics, Multiplayer"
    )


    submitted = st.form_submit_button(
        "🔍 Find Similar Games",
        use_container_width=True,
        type="primary"
    )


# =====================================================
# 7. GENERATE RECOMMENDATIONS
# =====================================================

if submitted:

    if not all([
        game_name.strip(),
        game_description.strip(),
        game_tags.strip()
    ]):

        st.warning(
            "Please enter a game name, description, "
            "and at least one tag."
        )

        st.session_state.pop("recommendations", None)

    else:

        with st.spinner("Finding similar Steam games..."):

            recommendations = recommend_games(
                game_name=game_name,
                game_price=game_price,
                game_description=game_description,
                game_tags=game_tags,
                df=df,
                features=features,
                params=params,
                k=10
            )

        st.session_state["recommendations"] = recommendations
        st.session_state["query_name"] = game_name


# =====================================================
# 8. DISPLAY RECOMMENDATIONS
# =====================================================

if "recommendations" in st.session_state:

    recommendations = st.session_state["recommendations"]

    st.divider()

    st.subheader(
        f"Top {len(recommendations)} Games Similar to "
        f"{st.session_state['query_name']}"
    )

    st.write(
        "Recommendations are ranked using our weighted "
        "hybrid similarity model, which considers game "
        "tags, description, name, and price."
    )

    for i, game in recommendations.iterrows():

        with st.container(border=True):

            left, right = st.columns([4, 1])

            with left:

                st.markdown(
                    f"### {i + 1}. {game[COL_NAME]}"
                )

                st.write(
                    game[COL_DESC]
                    or "No description available."
                )

                st.markdown(
                    "**Tags:** "
                    + ", ".join(game["tag_list"])
                )

            with right:

                price = game[COL_PRICE]

                st.metric(
                    "Price",
                    "Free" if price == 0 else f"${price:.2f}"
                )

                st.metric(
                    "Hybrid Score",
                    f"{game['similarity_score']:.2f}"
                )

            with st.expander("Why was this game recommended?"):

                c1, c2, c3, c4 = st.columns(4)

                c1.metric(
                    "Tag Similarity",
                    f"{game['tag_similarity']:.1%}"
                )

                c2.metric(
                    "Description Similarity",
                    f"{game['description_similarity']:.1%}"
                )

                c3.metric(
                    "Name Similarity",
                    f"{game['name_similarity']:.1%}"
                )

                c4.metric(
                    "Price Similarity",
                    f"{game['price_similarity']:.1%}"
                )

                if COL_OWNERS in game.index:
                    st.write(
                        "**Estimated owners:**",
                        game[COL_OWNERS]
                    )

            if str(game[COL_ID]).isdigit():

                st.link_button(
                    "View on Steam",
                    f"https://store.steampowered.com/app/"
                    f"{game[COL_ID]}/"
                )


# =====================================================
# 9. ABOUT THE MODEL
# =====================================================

with st.sidebar:

    st.header("About Our Model")

    st.write(
        "Our recommendation system combines four "
        "different measures of game similarity."
    )

    st.markdown("""
    - **Tags:** Cosine similarity between game tags.
    - **Description:** Semantic similarity using MiniLM.
    - **Name:** Character-level TF-IDF cosine similarity.
    - **Price:** Exponential similarity using log prices.
    """)

    st.subheader("Optimized Model Weights")

    weights_df = pd.DataFrame({
        "Feature": list(params["weights"].keys()),
        "Weight": list(params["weights"].values())
    })

    st.bar_chart(
        weights_df.set_index("Feature")
    )

    st.caption(
        "Weights were optimized using validation games "
        "based on hidden-tag recall and same-developer "
        "hit rate."
    )

    st.caption(
        "The hybrid score is a standardized ranking score, "
        "not a probability or percentage match."
    )
