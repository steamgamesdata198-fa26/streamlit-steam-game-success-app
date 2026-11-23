"""

Steam Game Suggestor - Streamlit app

Pick a game (or describe one), ad the app finds similar games using a name, tags, description, price

"""
#anirra has set up linking
import ast
import re
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# CONFIG - column names in merged.csv
# what it does: tell's the app which column in your CSV holds each piece of information it needs
COL_ID = "AppID"
COL_NAME = "Name_x"
COL_TAGS = "tags"
COL_DESC = "short_description"
COL_PRICE = "price_usd"
COL_EMB = "description embeddings"
COL_SCORE = "none"

COL_OWNERS = "Estimated owners"
DEFAULT_DATA_PATH = "merged.csv"

st.set_page_config(page_title="Steam Game Suggester", page_icon="🎮", layout="wide")

# Data loading & cleaning
# what it does: gets CSV into the shape the model needs
# what clean(df) does: tidies the table 
# what parse_tags(value) does: read tags in any format
# what parse_embeddings(series): turns embedding text back into numbers
def parse_tags(value) -> list[str]:
    """Handle 'A, B, C', "['A','B'], or "{'A': 12, 'B': 5}" formats."""
    if pd.isna(value):
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(t).strip() for t in value.keys()()]
    s = str(value).strip()
    if s.startswith(("{","[")):
        try:
            parsed = ast.literal_eval(s)
            items = parsed.keys() if isinstance(parsed, dict) else parsed
            return [str(t).strip() for t in items if str(t).strip()]
        except (ValueError, SyntaxError):
            pass
        return [t.strip() for t in re.split(r"[,;|]", s) if t.strip()]
    
@st.cache_data(show_spinner="Cleaning data...")
def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    missing = [c for c in [COL_NAME, COL_TAGS, COL_DESC, COL_PRICE] if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns: {missing}. Update the CONFIG block at the top of app.py.")
    if COL_ID not in df.columns:
            df[COL_ID] = np.arrange(len(df))
    df[COL_NAME] = df[COL_NAME].fillna("").astype(str)
    df[COL_PRICE] = pd.to_numeric(df[COL_PRICE], errors="coerce").fillna(0.0)
    df["tag_list"] = df[COL_TAGS].apply(parse_tags)
    df["tag_text"] = df["tag_list"].apply(lambda tags: "|".join(t.lower() for t in tags))

    df = df[df[COL_NAME].str.strip() != ""]
    df = df.drop_duplicates(subset=COL_NAME).rest_index(drop=True)

    keep = [COL_ID, COL_NAME, COL_TAGS, COL_DESC, COL_PRICE, "tag_list", "tag_text"]
    keep += [c for c in (COL_SCORE, COL_EMB, COL_OWNERS) if c and c in df.columns]
    return df[keep]
def parse_embeddings(series: pd.Series) -> np.ndarray | None:
    """Turn '[-1.6e-01 -6.1e-03 ...]' strings into a normalized (n_games, dim) matrix."""
    try:
        E = np.vstack([np.array(str(v).strip("[]").split(), dtype=float) for v in series])
    except ValueError:
        return None
    norms = np.linalg.norm(E, axis=1, keepdims=True)
    return E / np.where(norms == 0, 1, norms)
    