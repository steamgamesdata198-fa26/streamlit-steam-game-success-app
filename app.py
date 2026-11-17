"""

Steam Game Suggestor - Streamlit app

Pick a game (or describe one), ad the app finds similar games using a name, tags, description, price

"""
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
COL_SCORE = "success_score"
COL_EMB = "description embeddings"

COL_OWNERS = "Estimated owners"
DEFAULT_DATA_PATH = "merged.csv"

st.set_page(page_title="Steam Game Suggester", page_icon="🎮", layout="wide")
